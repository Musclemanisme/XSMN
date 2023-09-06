import torch
import torch.nn.functional as F

import torch.nn as nn

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveInstanceNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.ins_norm = nn.InstanceNorm2d(n, momentum=0.999, eps=0.001, affine=True)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.ins_norm(x)

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h




class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

def amp_Torch(x,m=0.25):  # m=0.3也是一个不错的值
    b,c,h,w = x.shape
    mean = x.sum(dim=3,keepdim=True).sum(dim=2,keepdim=True).sum(dim=1,keepdim=True) / (c * h * w)
    out = m * x / mean
    return torch.clamp(out,0,1)

class Amplifier(nn.Module):
    def __init__(self):
        super(Amplifier, self).__init__()
        self.requires_grad = False

    def forward(self,x):
        return amp_Torch(x)


        
class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class GRUnet(torch.nn.Module):
    def __init__(self, x_c,h_c,depth):
        self.x_c = x_c
        self.h_c = h_c
        self.depth = depth
        super(GRUnet, self).__init__()
        self.gate0=nn.Sequential(AtrousSeparableConvolution(h_c+x_c, h_c,kernel_size=3,padding=1),
                                 nn.Sigmoid()
                                 )
        self.gate1 = nn.Sequential(AtrousSeparableConvolution(h_c + x_c, h_c, kernel_size=3, padding=1),
                                   nn.Sigmoid()
                                   )
        self.gate2 = nn.Sequential(AtrousSeparableConvolution(h_c+x_c, h_c, kernel_size=3, padding=1),
                                   nn.Tanh()
                                   )

    def forward(self,x):
        b,_,h,w=x.shape
        h=torch.zeros([b,self.h_c,h,w]).to(x.device)
        for i in range(self.depth):
            fea0=torch.cat([h,x],dim=1)
            fea1=torch.cat([self.gate0(fea0)*h,x],dim=1)
            fea2=self.gate1(fea0)
            h=h*(1-fea2)+self.gate2(fea1)*fea2
        return h

class GCRDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer=6):
        super(GCRDB, self).__init__()
        _in_channels = in_channels
        modules = []

        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels))
            _in_channels += 64

        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)
        self.final_att = SE_net(in_channels=in_channels,out_channels=in_channels,reduction=16)

    def forward(self, x):
        out_rdb = self.residual_dense_layers(x)
        out_rdb = self.conv_1x1(out_rdb)
        out_rdb = self.final_att(out_rdb)
        out = out_rdb + x
        return out


class MakeDense(nn.Module):
    def __init__(self, in_channels):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=(3 - 1) // 2)
        # self.gcblock = ContextBlock2d(inplanes=in_channels, planes=in_channels)
        # self.norm_layer = nn.BatchNorm2d(growth_rate)
        self.norm_layer = AdaptiveInstanceNorm(64)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.norm_layer(out)
        out = torch.cat((x, out), 1)
        return out


class SE_net(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, attention=True):
        super().__init__()
        self.attention = attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_mid = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.conv_out = nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1, padding=0)

        self.x_red = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):

        if self.attention is True:
            y = self.avg_pool(x)
            y = F.relu(self.conv_in(y))
            y = F.relu(self.conv_mid(y))
            y = torch.sigmoid(self.conv_out(y))
            x = self.x_red(x)
            return x * y
        else:
            return x


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes=9, planes=32, pool='att', fusions=['channel_add'], ratio=4):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)  # context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.PReLU(),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.PReLU(),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


# class GCWTResDown(nn.Module):
#     def __init__(self, in_channels, norm_layer=AdaptiveInstanceNorm):
#         super().__init__()
#         self.dwt = DWT()
#         if norm_layer:
#             self.stem = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
#                                       norm_layer(in_channels),
#                                       nn.PReLU(),
#                                       nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
#                                       norm_layer(in_channels),
#                                       nn.PReLU())
#         else:
#             self.stem = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
#                                       nn.PReLU(),
#                                       nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
#                                       nn.PReLU())
#         self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
#         self.conv_down = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)
#
#     def forward(self, x):
#         stem = self.stem(x)
#         xLL, dwt = self.dwt(x)
#         res = self.conv1x1(xLL)
#         out = torch.cat([stem, res], dim=1)
#         return out, dwt
#
#
# class GCIWTResUp(nn.Module):
#
#     def __init__(self, in_channels, norm_layer=None):
#         super().__init__()
#         if norm_layer:
#             self.stem = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                 nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
#                 norm_layer(in_channels // 2),
#                 nn.PReLU(),
#                 nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
#                 norm_layer(in_channels // 2),
#                 nn.PReLU(),
#             )
#         else:
#             self.stem = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                 nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
#                 nn.PReLU(),
#                 nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
#                 nn.PReLU(),
#             )
#
#         self.pre_conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, padding=0)
#         self.prelu = nn.PReLU()
#         self.conv1x1 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1, padding=0)
#         self.iwt = IWT()
#
#     def forward(self, x, x_dwt):
#         stem = self.stem(x)
#         x_dwt = self.prelu(self.pre_conv(x_dwt))
#         x_iwt = self.iwt(x_dwt)
#         x_iwt = self.conv1x1(x_iwt)
#         out = torch.cat([stem, x_iwt], dim=1)
#         return out
#
#
# class shortcutblock(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.se = SE_net(in_channels, in_channels)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         return self.se(self.relu(self.conv2(self.relu(self.conv1(x)))))


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)
