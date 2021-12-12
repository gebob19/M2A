import torch.nn as nn
import torch.nn.functional as F

class TDN(nn.Module):
    def __init__(self, channel, n_segment=8, index=1):
        super().__init__()
        self.name = 'tdn'
        self.channel = channel
        self.reduction = 1
        self.n_segment = n_segment
        self.stride = 2**(index-1)
        self.conv1 = nn.Conv2d(in_channels=self.channel,
                out_channels=self.channel//self.reduction,
                kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        self.conv2 = nn.Conv2d(in_channels=self.channel//self.reduction,
                out_channels=self.channel//self.reduction,
                kernel_size=3, padding=1, groups=self.channel//self.reduction, bias=False)

        self.avg_pool_forward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_forward4 = nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.sigmoid_forward = nn.Sigmoid()

        self.avg_pool_backward2 = nn.AvgPool2d(kernel_size=2, stride=2)#nn.AdaptiveMaxPool2d(1)
        self.avg_pool_backward4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.sigmoid_backward = nn.Sigmoid()

        self.pad1_forward = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad1_backward = (0, 0, 0, 0, 0, 0, 1, 0)

        self.conv3 = nn.Conv2d(in_channels=self.channel//self.reduction,
                 out_channels=self.channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.conv3_smallscale2 = nn.Conv2d(in_channels=self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn3_smallscale2 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        
        self.conv3_smallscale4 = nn.Conv2d(in_channels = self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn3_smallscale4 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

    def spatial_pool(self, x):
        nt, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(nt, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(nt, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.view(nt,1,height,width)
        return context_mask


    def forward(self, x):
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:]) # n, t, c//r, h, w
        
        t_fea_forward, _ = reshape_bottleneck.split([self.n_segment -1, 1], dim=1) # n, t-1, c//r, h, w
        _, t_fea_backward = reshape_bottleneck.split([1, self.n_segment -1], dim=1) # n, t-1, c//r, h, w
        
        conv_bottleneck = self.conv2(bottleneck) # nt, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:]) # n, t, c//r, h, w
        _, tPlusone_fea_forward = reshape_conv_bottleneck.split([1, self.n_segment-1], dim=1) # n, t-1, c//r, h, w
        tPlusone_fea_backward ,_ = reshape_conv_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w
        diff_fea_forward = tPlusone_fea_forward - t_fea_forward # n, t-1, c//r, h, w
        diff_fea_backward = tPlusone_fea_backward - t_fea_backward# n, t-1, c//r, h, w
        diff_fea_pluszero_forward = F.pad(diff_fea_forward, self.pad1_forward, mode="constant", value=0) # n, t, c//r, h, w
        diff_fea_pluszero_forward = diff_fea_pluszero_forward.view((-1,) + diff_fea_pluszero_forward.size()[2:]) #nt, c//r, h, w
        diff_fea_pluszero_backward = F.pad(diff_fea_backward, self.pad1_backward, mode="constant", value=0) # n, t, c//r, h, w
        diff_fea_pluszero_backward = diff_fea_pluszero_backward.view((-1,) + diff_fea_pluszero_backward.size()[2:]) #nt, c//r, h, w
        y_forward_smallscale2 = self.avg_pool_forward2(diff_fea_pluszero_forward) # nt, c//r, 1, 1
        y_backward_smallscale2 = self.avg_pool_backward2(diff_fea_pluszero_backward) # nt, c//r, 1, 1

        y_forward_smallscale4 = diff_fea_pluszero_forward
        y_backward_smallscale4 = diff_fea_pluszero_backward
        y_forward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_forward_smallscale2))
        y_backward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_backward_smallscale2))

        y_forward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_forward_smallscale4))
        y_backward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_forward_smallscale4))
        
        y_forward_smallscale2 = F.interpolate(y_forward_smallscale2, diff_fea_pluszero_forward.size()[2:])
        y_backward_smallscale2 = F.interpolate(y_backward_smallscale2, diff_fea_pluszero_backward.size()[2:])
        
        y_forward = self.bn3(self.conv3(1.0/3.0*diff_fea_pluszero_forward + 1.0/3.0*y_forward_smallscale2 + 1.0/3.0*y_forward_smallscale4))# nt, c, 1, 1
        y_backward = self.bn3(self.conv3(1.0/3.0*diff_fea_pluszero_backward + 1.0/3.0*y_backward_smallscale2 + 1.0/3.0*y_backward_smallscale4)) # nt, c, 1, 1

        y_forward = self.sigmoid_forward(y_forward) - 0.5
        y_backward = self.sigmoid_backward(y_backward) - 0.5

        y = 0.5*y_forward + 0.5*y_backward
        return y

class TAM(nn.Module):
    def __init__(self,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 reduction=1,
                 padding=1):
        super().__init__()
        self.name = 'tam'
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        print('TAM with kernel_size {}.'.format(kernel_size))
        self.red = reduction

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False), nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // self.red,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // self.red),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // self.red, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        # x.size = N*C*T*(H*W)
        nt, c, h, w = x.size()
        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3,
                                                     4).contiguous()
        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))
        out = out.view(-1, t)
        conv_kernel = self.G(out.view(-1, t)).view(n_batch * c, 1, -1, 1)
        local_activation = self.L(out.view(n_batch, c,
                                           t)).view(n_batch, c, t, 1, 1)
        new_x = new_x * local_activation
        out = F.conv2d(new_x.view(1, n_batch * c, t, h * w),
                       conv_kernel,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n_batch * c)
        out = out.view(n_batch, c, t, h, w)
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)

        return out

class TEA(nn.Module):
    """ Motion exciation module
    
    :param reduction=16
    :param n_segment=8/16
    """
    def __init__(self, channel, reduction=16, n_segment=8):
        super().__init__()
        self.name = 'tea'
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        self.conv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
        self.conv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)
        self.identity = nn.Identity()

    def forward(self, x):
        nt, c, h, w = x.size()
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w
        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
        t_fea, __ = reshape_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w
        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment-1], dim=1)  # n, t-1, c//r, h, w
        
        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
        diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  #nt, c//r, h, w
        # y = self.avg_pool(diff_fea_pluszero)  
        y = self.conv3(diff_fea_pluszero)  
        y = self.bn3(y)  
        return y

# ---------------