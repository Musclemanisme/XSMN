import torch.nn as nn
import torch
from .modules import DWT,IWT
from torch.nn import functional as F
import pdb
'''
对应的训练文件夹是hratt_ml
'''
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y




class XSMB(nn.Module):
    def __init__(self, in_channels=48, distillation_rate= 1 / 3):
        super(XSMB, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.head = conv_layer(in_channels, in_channels, 3, dilation=1)

        self.c1 = conv_layer(in_channels,in_channels, 3, dilation=6, groups=3)
        self.d1 = conv_layer(in_channels, self.distilled_channels, 1)

        self.c2 = conv_layer(in_channels, in_channels, 3, dilation=12, groups=3)
        self.d2 = conv_layer(in_channels, self.distilled_channels, 1)

        self.c3 = conv_layer(in_channels, in_channels, 3, dilation=18, groups=3)
        self.d3 = conv_layer(in_channels, self.distilled_channels, 1)


        self.att1 = Modulation()
        self.att2 = Modulation()
        self.att3 = Modulation()

        self.act = activation('lrelu', neg_slope=0.05)  # attention part
        self.out_layer = conv_layer(self.distilled_channels * 3, in_channels, 1)

    def forward(self, input):
        in_ = self.act(self.head(input))

        c1_out = self.act(self.c1(in_))
        c1_out = self.d1(c1_out)

        c2_out = self.act(self.c2(in_))
        c2_out = self.d2(c2_out)

        c3_out = self.act(self.c3(in_))
        c3_out = self.d3(c3_out)

        #########attention部分####################
        att1 = self.att1(c2_out,c3_out,c1_out)
        att2 = self.att2(c1_out,c3_out,c2_out)
        att3 = self.att3(c1_out,c2_out,c3_out)
        out = torch.cat([att1, att2, att3], dim=1)
        ##############无attention部分####################
        # out = torch.cat([c1_out,c2_out,c3_out],dim=1)

        out_fused = self.out_layer(out) + input
        return out_fused



class XSMN(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super(XSMN, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=3, stride=1, padding=1)
        self.final1 = nn.Conv2d(48, out_channels, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(in_channels, 48, kernel_size=3, stride=1, padding=1)
        self.final2 = nn.Conv2d(48, out_channels, kernel_size=3, padding=1)
        self.co_conv = nn.Conv2d(15, out_channels, kernel_size=1, padding=0)
        self.net1 = nn.Sequential(*[XSMB() for _ in range(1)])
        self.net2 = nn.Sequential(*[XSMB() for _ in range(5)])
    def forward(self, x):
        _, _, pad_h, pad_w = x.size()
        if pad_h % 2 != 0 or pad_w % 2 != 0:
            h_pad_len = 2 - pad_h % 2
            w_pad_len = 2 - pad_w % 2
            x = F.pad(x, (0, w_pad_len, 0, h_pad_len), mode='reflect')

        x_ll, x_dwt = self.dwt(x)
        x_ll2,x_dwt2 = self.dwt(x_ll)
        # pdb.set_trace()
        x2 = self.conv2(x_dwt2)
        x2_out = self.net2(x2)
        x2_out = self.final2(x2_out)
        x2_out = self.iwt(x2_out)

        x1 = self.conv1(x_dwt)
        x1_out = self.final1(self.net1(x1))

        x_final = torch.cat((x2_out, x1_out), dim=1)

        x_out = self.co_conv(x_final)
        out = self.iwt(x_out) + x
        return out[:,:,:pad_h,:pad_w]




class Modulation(nn.Module):
    def __init__(self,in_channels=32,mid_channels=32, out_channels=16):
        super(Modulation, self).__init__()
        self.att = nn.Sequential(nn.Conv2d(in_channels,mid_channels,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(inplace=True),   # 相比之前加了relu
                                 nn.Conv2d(mid_channels, out_channels, 1, 1, 0),
                                 nn.Sigmoid(),)



    def forward(self, q, k, v):
        att_input = torch.cat((q,k), dim=1)
        att_map = self.att(att_input)
        assert (att_map.dim() == 4)
        #########放大att_map的值让他们不至于全在(0,1]###########
        att_map_spatial_sum = att_map.sum(3, keepdim=True).sum(2,keepdim=True)
        att_map = (att_map * att_map.size(2) * att_map.size(3)) / att_map_spatial_sum
        out = torch.mul(v,att_map)
        return out


