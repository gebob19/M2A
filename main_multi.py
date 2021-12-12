# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu


#%%
import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from ops.lr_scheduler import get_scheduler

from ops.dataset import TSNDataSet
from ops.models import TSN, Transformer
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import *
from ops.temporal_shift import make_temporal_pool
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_prec1 = 0

def cleanup():
    dist.destroy_process_group()

#%%
def main():
    global args, best_prec1
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    num_class, args.train_list, args.val_list, args.root_path, prefix = \
        dataset_config.return_dataset(args.dataset, args.modality, args.datapath)
        
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    if dist.get_rank() == 0:
        check_rootfolders(args)

    if args.model == 'tsm':
        model = TSN(num_class, args.num_segments, args.modality, args, 
                    base_model=args.arch,
                    consensus_type=args.consensus_type,
                    dropout=args.dropout,
                    dropout_type=args.dropout_type,
                    img_feature_dim=args.img_feature_dim,
                    partial_bn=not args.no_partialbn,
                    pretrain=args.pretrain,
                    temporal_module=args.temporal_module,
                    is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                    fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                    temporal_pool=args.temporal_pool,
                    non_local=args.non_local)
    elif args.model == 'transformer':
        model = Transformer(num_class, args.num_segments, args)
    else: 
        raise NotImplementedError

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = DistributedDataParallel(model.cuda(), device_ids=[args.local_rank], broadcast_buffers=True, find_unused_parameters=True)

    # papers impl
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    # get 40 frames of 8segment from video each with 3 channels (RGB) => 120 channel
    train_dataset = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, sampler=train_sampler, drop_last=True)  

    val_dataset = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, num_workers=args.workers, 
        pin_memory=True, sampler=val_sampler, drop_last=True)

    scheduler = get_scheduler(optimizer, len(train_loader), args)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    if args.trial_run: 
        args.epochs = 1 

    log_training = open(os.path.join(args.root_log, args.exp_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.exp_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    if dist.get_rank() == 0:
        tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.exp_name))

        # my custom experiment writer
        from writers import NeptuneWriter
        writer = NeptuneWriter('gebob19/something-something')
        config = vars(args)
        config['experiment_name'] = config['exp_name']
        if not args.trial_run: 
            writer.start(config, upload_source_files=['main.py', 'ops/temporal_shift.py'])

    assert args.n_batch_multiplier > 0
    print('Training with batchsize: ', args.batch_size * args.n_batch_multiplier)

    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch=epoch, scheduler=scheduler)

        if dist.get_rank() == 0:
            metrics = {
                'top1_acc': train_top1,
                'top5_acc': train_top5,
                'loss': train_loss,
                'learning_rate': optimizer.param_groups[-1]['lr'],
            }
            writer.write(metrics, epoch)
            
            tf_writer.add_scalar('loss/train', train_loss, epoch)
            tf_writer.add_scalar('acc/train_top1', train_top1, epoch)
            tf_writer.add_scalar('acc/train_top5', train_top5, epoch)
            tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_loader.sampler.set_epoch(epoch)
            prec1, prec5, val_loss = validate(val_loader, model, criterion, epoch)
            if dist.get_rank() == 0:

                metrics = {
                    'val_top1_acc': prec1,
                    'val_top5_acc': prec5,
                    'val_loss': val_loss,
                }
                writer.write(metrics, epoch)

                tf_writer.add_scalar('loss/test', val_loss, epoch)
                tf_writer.add_scalar('acc/test_top1', prec1, epoch)
                tf_writer.add_scalar('acc/test_top5', prec5, epoch)

                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

                print(("Best Prec@1: '{}'".format(best_prec1)))
                tf_writer.flush()
                save_epoch = epoch + 1
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'prec1': prec1,
                        'best_prec1': best_prec1,
                    }, epoch, is_best)

def train(train_loader, model, criterion, optimizer, epoch, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    model.train()

    end = time.time()
    loss = 0
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        
        output = model(input_var)
        loss = criterion(output, target_var) / args.n_batch_multiplier
        loss.backward()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        scheduler.step()

        if (i+1) % args.n_batch_multiplier == 0:
            if args.clip_gradient is not None:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.trial_run: 
                break 

        if False and i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                             top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))  # TODO
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            target = target.cuda()
            output = model(input)

            loss = criterion(output, target)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            loss = reduce_tensor(loss)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if args.trial_run: 
                break 

            if False and i % args.print_freq == 0:
                print(
                    ('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))
    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
        .format(top1=top1, top5=top5, loss=losses)))
    return top1.avg, top5.avg, losses.avg

def save_checkpoint(state, epoch, is_best):
    filename = '{}/{}/ckpt{}.pth.tar'.format(args.root_model, args.exp_name, epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr #* param_group['lr_mult']
        param_group['weight_decay'] = decay #* param_group['decay_mult']


def check_rootfolders(args):
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.exp_name),
                    os.path.join(args.root_model, args.exp_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
