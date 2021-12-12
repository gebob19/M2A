# BUILT OFF OF: 
# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
from einops import rearrange

import sys 
sys.path.append('./ops/')
from sota import TEA, TAM, TDN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def self_attention(q, k, v):
    """query, key, value: (B, T, E): batch, temporal, embedding
    """
    dk = q.size(-1)
    scores = torch.bmm(q, torch.transpose(k, 1, 2)) / math.sqrt(dk)
    scores = F.softmax(scores, -1)
    output = torch.bmm(scores, v)

    return output, scores

def compute_shift(x, type, amount=1):
    # x = (batch, T, ...)
    pad = torch.zeros_like(x).to(x.device)[:, :amount]
    if type == 'left':
        xt2 = torch.cat([x.clone()[:, amount:], pad], 1)
    elif type == 'right':
        xt2 = torch.cat([pad, x.clone()[:, :-amount]], 1)
    xt2.to(x.device)
    return xt2

class Motion(nn.Module):
    def __init__(self, n_segment, einops_from, einops_to):
        super().__init__()
        self.n_segment = n_segment
        self.einops_from, self.einops_to = einops_from, einops_to
        self.name = 'motion'
    
    def forward(self, x):
        x = rearrange(x, f'{self.einops_from} -> {self.einops_to}', t=self.n_segment)
        x = compute_shift(x, type='left') - x 
        x = rearrange(x, f'{self.einops_to} -> {self.einops_from}')
        return x 

class Attention(nn.Module):
    def __init__(self, n_segment, einops_to):
        super().__init__()
        self.name = 'attn'
        self.n_segment = n_segment
        self.einops_from = '(b t) c h w'
        self.einops_to = einops_to

    def forward(self, x):
        bt, c, h, w = x.size()

        x = rearrange(x, f'{self.einops_from} -> {self.einops_to}', t=self.n_segment, c=c, h=h, w=w)
        xn = F.layer_norm(x, [x.size(-1)])
        x = self_attention(xn, xn, xn)[0] + x
        x = rearrange(x, f'{self.einops_to} -> {self.einops_from}', t=self.n_segment, c=c, h=h, w=w)
                
        return x

class SpatialPatchMotionAttention(nn.Module):
    def __init__(self, n_segment, einops_to, patch_size):
        super().__init__()
        self.name = 'attn'
        self.n_segment = n_segment
        self.einops_from = '(b t) c h w'
        self.einops_to = einops_to
        
        self.p = patch_size # patch size
    
    def forward(self, x):
        bt, c, h, w = x.size()
        b = bt // self.n_segment 
        p = self.p
        ph, pw = (h // p), (w // p)

        patchable = ((h / p) % 1) == 0 # can it be split into patches? 
        if patchable:
            # temporal attention 
            x = rearrange(x, '(b t) c (ph p1) (pw p2) -> (b ph pw) t (p1 p2 c)', p1=p, p2=p, b=b) 
            xn = F.layer_norm(x, [x.size(-1)])
            # motion info 
            xn = compute_shift(xn, 'left') - xn
            x = self_attention(xn, xn, xn)[0] + x 
            
            # spatial attention
            x = rearrange(x, '(b ph pw) t (p1 p2 c) -> (b t) (ph pw) (p1 p2 c)', p1=p, p2=p, ph=ph, b=b)
            xn = F.layer_norm(x, [x.size(-1)])
            x = self_attention(xn, xn, xn)[0] + x 

            x = rearrange(x, '(b t) (ph pw) (p1 p2 c) -> (b t) c (ph p1) (pw p2)', p1=p, p2=p, ph=ph, b=b)

        else: # spatial is small enough for per pixel (OOM otherwise)
            x = rearrange(x, '(b t) c h w -> b (t h w) c', b=b) 
            xn = F.layer_norm(x, [x.size(-1)])
            xn = compute_shift(xn, 'left') - xn
            x = self_attention(xn, xn, xn)[0] + x 
            x = rearrange(x, 'b (t h w) c -> (b t) c h w', t=self.n_segment, h=h, w=w) 
    
        return x

class CustomMotionAttention(nn.Module):
    def __init__(self, n_segment, einops_to, motion_mod, in_channels):
        super().__init__()
        self.name = 'attn'
        self.n_segment = n_segment
        self.einops_from = '(b t) c h w'
        self.einops_to = einops_to
        self.motion_mod = motion_mod
        self.attn_activations = None
        self.in_channels = in_channels

    def forward(self, x):
        bt, c, h, w = x.size()

        # layer norm 
        xn = rearrange(x, f'{self.einops_from} -> {self.einops_to}', t=self.n_segment, c=c, h=h, w=w)
        xn = F.layer_norm(xn, [xn.size(-1)])
        xn = rearrange(xn, f'{self.einops_to} -> {self.einops_from}', t=self.n_segment, c=c, h=h, w=w)

        # motion 
        xn = self.motion_mod(xn)  

        # attention -- don't use Attention class bc of layernorm
        xn = rearrange(xn, f'{self.einops_from} -> {self.einops_to}', t=self.n_segment, c=c, h=h, w=w)
        x_attn, _ = self_attention(xn, xn, xn)
        x_attn = rearrange(x_attn, f'{self.einops_to} -> {self.einops_from}', t=self.n_segment, c=c, h=h, w=w)
        
        self.attn_activations = x_attn

        # skip connection
        x = x_attn + x

        return x 

class CustomMotionTAM(nn.Module):
    def __init__(self, n_segment, einops_to, rsize, motion_mod):
        super().__init__()
        self.name = 'attn'
        self.n_segment = n_segment
        self.einops_from = '(b t) c h w'
        self.einops_to = einops_to
        self.motion_mod = motion_mod
        self.attn = TAM(rsize, n_segment=n_segment)
        
        self.prod_dims = None
        
    def set_dims(self, size):
        _, c, h, w = size
        dims = self.einops_to[self.einops_to.rfind('(')+1:].split(')')[0].split(' ')
        kv = {'c': c, 'h': h, 'w': w, 't': self.n_segment}
        self.dims = [kv[d] for d in dims]
        self.prod_dims = np.prod(self.dims)

    def forward(self, x):
        bt, c, h, w = x.size()
        if self.prod_dims is None: 
            self.set_dims(x.size())

        # layer norm 
        x_norm = F.layer_norm(x, self.dims)
        # motion 
        x_norm = self.motion_mod(x_norm)
        # attention 
        x_attn = self.attn(x_norm) 
        # skip connection
        x = x_attn + x
        return x

class M2A(nn.Module):
    def __init__(self, args, in_channels, n_segment, n_div, blocks):
        super().__init__()
        self.n_segment = n_segment
        self.rsize = in_channels // n_div

        def block2mod(block):
            name, attn_shape = block
            if name == 'motion': 
                m = Motion(n_segment, '(b t) c h w', 'b t c h w')
            elif name == 'tdn':
                m = TDN(self.rsize, n_segment=n_segment)
            elif name == 'tam':
                m = TAM(self.rsize, n_segment=n_segment)
            elif name == 'tea':
                m = TEA(self.rsize, reduction=1, n_segment=n_segment)
            elif name == 'attn': 
                m = Attention(n_segment, attn_shape)
            elif name == 'patch_mattn': 
                m = SpatialPatchMotionAttention(n_segment, attn_shape, args.patch_size)
            elif '+' in name: # combination of motion & attention modules (motion+attn)
                mods = name.split('+')
                motion_m, attn_m = mods[0], mods[1]
                
                # get the motion module
                motion_mod = block2mod((motion_m, ''))

                # get the attention module
                if attn_m == 'attn':
                    m = CustomMotionAttention(n_segment, attn_shape, motion_mod, self.rsize)
                elif attn_m == 'tam':
                    m = CustomMotionTAM(n_segment, attn_shape, self.rsize, motion_mod)
                else: 
                    raise NotImplementedError
            else: 
                raise NotImplementedError
            return m 
        
        self.modules_list = nn.ModuleList([])
        for block in blocks:
            m = block2mod(block)
            self.modules_list.append(m)

        self.conv_down = nn.Conv2d(in_channels, self.rsize, 1)
        self.conv_up = nn.Conv2d(self.rsize, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        shortcut = x
        x = self.conv_down(x)
        
        for m in self.modules_list:
            x = m(x)

        x = self.conv_up(x)
        x = self.sigmoid(x) 

        x = x * shortcut + shortcut
        return x

class M2A_3DCNN_Wrapper(nn.Module):
    def __init__(self, args, in_channels, n_segment, n_div, blocks):
        super().__init__()
        self.m2a = M2A(args, in_channels, n_segment, n_div, blocks)
        self.blocks = blocks

    def forward(self, x):
        b, c, t, h, w = x.size()

        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.m2a(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x 

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x

def make_temporal_shift(net, temporal_module, n_segment, args, i3d=False, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    if place == 'block':
        def make_block_temporal(stage, this_segment):
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks'.format(len(blocks)))
            for i, b in enumerate(blocks):
                if temporal_module == 'tsm':
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
                else: 
                    print('Warning: Not Implemented')
                    raise NotImplementedError
            return nn.Sequential(*(blocks))

        net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
        net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
        net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
        net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

    elif 'blockres' in place:
        n_round = 1
        if len(list(net.layer3.children())) >= 23:
            n_round = 2
            print('=> Using n_round {} to insert temporal shift'.format(n_round))

        def make_block_temporal(x, stage, this_segment, index):
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks residual'.format(len(blocks)))
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    embed_dim = x.shape[-1] * x.shape[-2]
                    if temporal_module == 'tsm':
                        blocks[i].conv1 = TemporalShift(b.conv1, 
                            n_segment=this_segment, n_div=n_div)
                    else: 
                        ### M2A EXPERIMENTS
                        t = this_segment if not i3d else x.size(2)
                        modules = [
                          get_temporal_module(temporal_module,
                            blocks[i].conv1.in_channels,
                            t,
                            n_div=n_div,
                            i3d=i3d, args=args),
                        ]
                        ## TSM + M2A
                        if args.use_tsm: 
                            modules.append(TemporalShift(nn.Sequential(), t, 8))
                        modules.append(blocks[i].conv1)
                        blocks[i].conv1 = nn.Sequential(*modules)
                x = blocks[i](x)

            return nn.Sequential(*blocks), x
        
        # track the size of the tensors
        # (batch * time, channel, height, width) -- 2D networks 
        x = torch.randn(2 * n_segment, 3, 224, 224)
        if i3d: 
            # (batch, channel, time, height, width) -- 3D networks
            x = torch.randn(2, 3, n_segment, 224, 224)

        x = net.conv1(x)
        x = net.maxpool(x)

        net.layer1, x = make_block_temporal(x, net.layer1, n_segment_list[0], 0)
        net.layer2, x = make_block_temporal(x, net.layer2, n_segment_list[1], 1)
        net.layer3, x = make_block_temporal(x, net.layer3, n_segment_list[2], 2)
        net.layer4, _ = make_block_temporal(x, net.layer4, n_segment_list[3], 3)


def get_temporal_module(temporal_module, in_channels, this_segment,
    n_div=8, i3d=False, args=None):

    if temporal_module == 'attn':
        network_blocks = [
            ('attn', 'b t (c h w)'),
        ]
    elif '+' in temporal_module: ## tdn+attn OR motion+tam etc.
        network_blocks = [
            (temporal_module, 'b t (c h w)'),
        ]
    elif temporal_module in ['tdn', 'tam', 'tea', 'motion', 'patch_mattn']:
        network_blocks = [
            (temporal_module, '')
        ]
    else: 
        print(f'Warning: {temporal_module} not implemented...')
        raise NotImplementedError

    if i3d: 
        mod = M2A_3DCNN_Wrapper(args, in_channels, this_segment,
            n_div, network_blocks)    
    else: 
        mod = M2A(args, in_channels, this_segment,
            n_div, network_blocks)

    return mod

def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError

