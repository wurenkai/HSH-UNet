import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from functools import reduce
import os
import sys
import torch.fft
import math

import traceback


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM


        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)
        # print('Using Megvii large kernel dw conv impl')
    except:
        print(traceback.format_exc())


        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

        # print('[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl')
else:
    def get_dwconv(dim, kernel, bias):
        return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

    # print('Using PyTorch large kernel dw conv impl')


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s

        print('[gconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class Block(nn.Module):
    r""" HorNet block
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        #self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        #att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            #att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            #att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4):
        t_list = [t1, t2, t3, t4]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4):
        r1, r2, r3, r4 = t1, t2, t3, t4

        satt1, satt2, satt3, satt4 = self.satt(t1, t2, t3, t4)
        t1, t2, t3, t4 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4

        r1_, r2_, r3_, r4_ = t1, t2, t3, t4
        t1, t2, t3, t4 = t1 + r1, t2 + r2, t3 + r3, t4 + r4

        catt1, catt2, catt3, catt4 = self.catt(t1, t2, t3, t4)
        t1, t2, t3, t4 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_


class HSH_UNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, gnconv=gnconv, block=Block,
                 pretrained=None,
                 use_checkpoint=False, c_list=[32, 64, 128, 256, 512],
                 split_att='fc', bridge=True):
        super().__init__()
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )


        if not isinstance(gnconv, list):
            gnconv = [partial(gnconv, order=2, s=1 / 3),
                      partial(gnconv, order=3, s=1 / 3),
                      partial(gnconv, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                      partial(gnconv, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)


        self.encoder2 = nn.Sequential(
            S2345nConv(in_channels=c_list[0], out_channels=c_list[0]),
            nn.Conv2d(c_list[0], c_list[1], 1, stride=1, padding=0),
        )

        self.encoder3 = nn.Sequential(
            S2345nConv(in_channels=c_list[1], out_channels=c_list[1]),
            nn.Conv2d(c_list[1], c_list[2], 1, stride=1, padding=0),
        )

        self.encoder4 = nn.Sequential(
            S2345nConv(in_channels=c_list[2], out_channels=c_list[2]),
            nn.Conv2d(c_list[2], c_list[3], 1, stride=1, padding=0),
        )

        self.encoder5 = nn.Sequential(
            S2345nConv(in_channels=c_list[3], out_channels=c_list[3]),
            nn.Conv2d(c_list[3], c_list[4], 1, stride=1, padding=0),
        )

        # build Bottleneck layers
        self.ConvMixer = ConvMixerBlock(dim=c_list[4], depth=7, k=7)

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')


        self.decoder1 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[3], 1, stride=1, padding=0),
            S2345nConv(in_channels=c_list[3], out_channels=c_list[3]),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[2], 1, stride=1, padding=0),
            S2345nConv(in_channels=c_list[2], out_channels=c_list[2]),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 1, stride=1, padding=0),
            S2345nConv(in_channels=c_list[1], out_channels=c_list[1]),
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 1, stride=1, padding=0),
            S2345nConv(in_channels=c_list[0], out_channels=c_list[0]),
        )


        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[3])
        self.dbn2 = nn.GroupNorm(4, c_list[2])
        self.dbn3 = nn.GroupNorm(4, c_list[1])
        self.dbn4 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        if self.bridge: t1, t2, t3, t4 = self.scab(t1, t2, t3, t4)
        out = F.gelu((self.ebn5(self.encoder5(out))))# b, c5, H/64, W/64
        out = self.ConvMixer(out)

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t4)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t3)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t2)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t1)  # b, c1, H/4, W/4

        out0 = F.interpolate(self.final(out2), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)


class EAblock(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, in_c, 1)

        self.k = in_c * 4
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.norm_layer = nn.GroupNorm(4, in_c)

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.norm_layer(self.conv2(x))
        x = x + idn
        x = F.gelu(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# bottleneck
class ConvMixerBlock(nn.Module):
    def __init__(self, dim=256, depth=7, k=7):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)
        return x


class g5conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, gnconv=gnconv, block=Block):  # ch_in, ch_out, kernel, stride, padding, groups
        super(g5conv, self).__init__()


        if not isinstance(gnconv, list):
            gnconv = [partial(gnconv, order=2, s=1 / 3),
                      partial(gnconv, order=3, s=1 / 3),
                      partial(gnconv, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                      partial(gnconv, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        self.gnconv = nn.Sequential(
            EAblock(c1),
            *[block(dim=c1, drop_path=0.5,
                    layer_scale_init_value=1e-6, gnconv=gnconv[3]) for j in range(1)],
            #ConvMixerBlock(dim=c1, depth=7, k=7),
        )
    def forward(self, x):
        return self.gnconv(x)

    def forward_fuse(self, x):
        return self.gnconv(x)


class g4conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, gnconv=gnconv, block=Block):  # ch_in, ch_out, kernel, stride, padding, groups
        super(g4conv, self).__init__()


        if not isinstance(gnconv, list):
            gnconv = [partial(gnconv, order=2, s=1 / 3),
                      partial(gnconv, order=3, s=1 / 3),
                      partial(gnconv, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                      partial(gnconv, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        self.gnconv = nn.Sequential(
            EAblock(c1),
            *[block(dim=c1, drop_path=0.5,
                    layer_scale_init_value=1e-6, gnconv=gnconv[2]) for j in range(1)],
            #ConvMixerBlock(dim=c1, depth=7, k=7),
        )
    def forward(self, x):
        return self.gnconv(x)

    def forward_fuse(self, x):
        return self.gnconv(x)


class g3conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, gnconv=gnconv, block=Block):  # ch_in, ch_out, kernel, stride, padding, groups
        super(g3conv, self).__init__()


        if not isinstance(gnconv, list):
            gnconv = [partial(gnconv, order=2, s=1 / 3),
                      partial(gnconv, order=3, s=1 / 3),
                      partial(gnconv, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                      partial(gnconv, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        self.gnconv = nn.Sequential(
            EAblock(c1),
            *[block(dim=c1, drop_path=0.5,
                    layer_scale_init_value=1e-6, gnconv=gnconv[1]) for j in range(1)],
            #ConvMixerBlock(dim=c1, depth=7, k=7),
        )
    def forward(self, x):
        return self.gnconv(x)

    def forward_fuse(self, x):
        return self.gnconv(x)


class g2conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, gnconv=gnconv, block=Block):  # ch_in, ch_out, kernel, stride, padding, groups
        super(g2conv, self).__init__()


        if not isinstance(gnconv, list):
            gnconv = [partial(gnconv, order=2, s=1 / 3),
                      partial(gnconv, order=3, s=1 / 3),
                      partial(gnconv, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                      partial(gnconv, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        self.gnconv = nn.Sequential(
            EAblock(c1),
            *[block(dim=c1, drop_path=0.5,
                    layer_scale_init_value=1e-6, gnconv=gnconv[0]) for j in range(1)],
            #ConvMixerBlock(dim=c1, depth=7, k=7),
        )
    def forward(self, x):
        return self.gnconv(x)

    def forward_fuse(self, x):
        return self.gnconv(x)



class S2345nConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=4,r=16,L=32):

        super(S2345nConv,self).__init__()
        d=max(in_channels//r,L)   
        self.M=M
        self.out_channels=out_channels
        self.gnconv_list = [g2conv,g3conv,g4conv,g5conv]  #g2conv,g3conv,g4conv,g5conv
        self.conv=nn.ModuleList()  
        for i in range(M):
            self.conv.append(nn.Sequential(self.gnconv_list[i](in_channels),
                                            nn.BatchNorm2d(in_channels),
                                            nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(output_size = 1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) 
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            output.append(conv(input))    #[batch_size,out_channels,H,W]
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) 
        s=self.global_pool(U)     # [batch_size,channel,1,1]
        z=self.fc1(s)  # S->Z降维   # [batch_size,32,1,1]
        a_b=self.fc2(z) # Z->a，b 升维 
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]
        a_b=self.softmax(a_b) # 使得四个全连接层对应位置进行softmax [batch_size,M,out_channels,1]
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) 
        V=list(map(lambda x,y:x*y,output,a_b)) 
        V=reduce(lambda x,y:x+y,V) 
        return V    
