from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from functools import partial
from einops import rearrange
from torch import Tensor
from config import Config
from mamba_ssm import Mamba
#from .transformer import Transformer
config = Config()
from .mamba2d import VSSBlock,Block2D


class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResBlk(nn.Module):
    def __init__(self, channel_in, channel_out, groups=0):
        super(ResBlk, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, 64, 3, 1, 1)
        self.relu_in = nn.ReLU(inplace=True)
        if config.dec_att == 'ASPP':
            self.dec_att = ASPP(channel_in=64)
        self.conv_out = nn.Conv2d(64, channel_out, 3, 1, 1)
        if config.use_bn:
            self.bn_in = nn.BatchNorm2d(64)
            self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        if config.use_bn:
            x = self.bn_in(x)
        x = self.relu_in(x)
        if config.dec_att:
            x = self.dec_att(x)
        x = self.conv_out(x)
        if config.use_bn:
            x = self.bn_out(x)
        return x


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type=False,
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        # if self.norm_type is not None:
        #     x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x



class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div=2,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.mlp(x)
        #x = self.up(x)
        return x
        

class AttentionModule(nn.Module):
    def __init__(self, in_dim):
        super(AttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (in_dim ** 0.5)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)

        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        
        drop_path=0.
        depth=4
        #build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(Block2D(
                hidden_dim=256,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=16,
                expand=4,
                input_resolution=(16,16),is_light_sr=False))
        
        self.conv_in_dim_128 = nn.Conv2d(in_dim, 128, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x4, x3):
        # x4----x4
        B, C, H, W = x4.size()
        a_value = self.value_conv(x4)
        a_query = self.query_conv(x4).view(B, -1, W * H).permute(0, 2, 1)  # [B,HW,C]
        a_key = self.key_conv(x4).view(B, -1, W * H).permute(1, 0, 2)  # [B,C,HW]
        a_query = a_query.contiguous().view(-1, C)  # [BHW,C]
        a_key = a_key.contiguous().view(C, -1)
        a_x = torch.matmul(a_query, a_key)  # [BHW,BHW] x4 Q*K
        a_x = a_x.view(B * H * W, B, H * W)  # [BHW, B, HW]
        a_max = torch.max(a_x, dim=-1)[0]  # [BHW, B]
        a_mean = torch.mean(a_max, dim=-1)  # [BHW, B]
        # a_w = a_mean.sum(-1)  # [BWH]

        a_w = a_mean.view(B, -1) * self.scale
        a_w = F.softmax(a_w, dim=-1)  # [B,HW]
        a_w = a_w.view(B, H, W).unsqueeze(1)  # [B,1,16,16]
        a_x4 = self.conv_in_dim_128(a_w * a_value)  # [128,16,16]

        # x3----x3
        B, C, H, W = x3.size()
        b_value = self.value_conv(x3)
        b_query = self.query_conv(x3).view(B, -1, W * H).permute(0, 2, 1)  # [B,HW,C]
        b_key = self.key_conv(x3).view(B, -1, W * H).permute(1, 0, 2)  # [B,C,HW]
        b_query = b_query.contiguous().view(-1, C)  # [BHW,C]
        b_key = b_key.contiguous().view(C, -1)
        b_x = torch.matmul(b_query, b_key)  # [BHW,BHW] x3 Q*K
        b_x = b_x.view(B * H * W, B, H * W)  # [BHW, B, HW]
        b_max = torch.max(b_x, dim=-1)[0]  # [BHW, B]
        b_mean = torch.mean(b_max, dim=-1)  # [BHW, B]
        # b_w = b_mean.sum(-1)  # [BWH]
        b_w = b_mean.view(B, -1) * self.scale
        b_w = F.softmax(b_w, dim=-1)  # [B,HW]
        b_w = b_w.view(B, H, W).unsqueeze(1)  # [B,1,16,16]
        b_x3 = self.conv_in_dim_128(b_w * b_value)  # [128,16,16]
        # x4----x3
        a_query = self.query_conv(x4).view(B, -1, W * H).permute(0, 2, 1)  # [B,HW,C]

        a_query = a_query.contiguous().view(-1, C)  # [BHW,C]
        b_key = self.key_conv(x3).view(B, -1, W * H).permute(1, 0, 2)  # [C,HW,B]
        b_key = b_key.contiguous().view(C, -1)  #

        mab = torch.matmul(a_query, b_key)  # [BHW,BHW] x43 Q*K 4-q k-kv

        ab_x = mab.view(B * W * H, B, W * H)  # [BHW, B, HW]
        ab_max = torch.max(ab_x, dim=-1)[0]  # [BHW, B]
        ab_mean = torch.mean(ab_max, dim=-1)  # [BHW, B]
        # ab_w = ab_mean.sum(-1)  # [BWH]
        ab_w = ab_mean.view(B, -1) * self.scale
        ab_w = F.softmax(ab_w, dim=-1)  # [B,HW]
        ab_w = ab_w.view(B, H, W).unsqueeze(1)  # [B,1,16,16]
        x43 = self.conv_in_dim_128(ab_w * b_value)  # [128,16,16]
        # x3----x4
        b_query = self.query_conv(x3).view(B, -1, W * H).permute(0, 2, 1)  # [B,HW,C]
        b_query = b_query.contiguous().view(-1, C)  # [BHW,C]
        a_key = self.key_conv(x4).view(B, -1, W * H).permute(1, 0, 2)  # [B,C,HW]
        a_key = a_key.contiguous().view(C, -1)
        ba_x = torch.matmul(b_query, a_key)  # [BHW,BHW] x43 Q*K 4-q k-kv
        ba_x = ba_x.view(B * H * W, B, H * W)  # [BHW, B, HW]
        ba_max = torch.max(ba_x, dim=-1)[0]  # [BHW, B]
        ba_mean = torch.mean(ba_max, dim=-1)  # [BHW, B]
        # ba_w = ba_mean.sum(-1)  # [BWH]
        ba_w = ba_mean.view(B, -1) * self.scale
        ba_w = F.softmax(ba_w, dim=-1)  # [B,HW]
        ba_w = ba_w.view(B, H, W).unsqueeze(1)  # [B,1,16,16]
        x34 = self.conv_in_dim_128(ba_w * a_value)  # [128,16,16]
        # Cat
        xa_cat = torch.cat([x34, a_x4], 1)
        xb_cat = torch.cat([x43, b_x3], 1)
        ab_cat = torch.cat([xa_cat, xb_cat], 1)

        for block in self.blocks:
            ab1, ab2 = torch.chunk(ab_cat, 2, dim=1)
            ab12 = block(ab1,ab2)
        ab12 = ab_cat * ab12
        ab_avg = self.avgpool(ab12)
        ab = F.softmax(ab_avg, dim=-1)
        final = ab * ab_cat
        return final


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True) 
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SAM(nn.Module):
    def __init__(self,channels):
        super(SAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=-1)
        self.conv512_64 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.conv128_64 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.branch0 = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1, bias=False),
        )
    def forward(self, x1, x2):
        x12 = x1 * x2
        x12_avg = self.avg_pool(x12)
        x12_max = self.max_pool(x12)
        x12_avg = self.branch0(x12_avg)
        x12_max = self.branch0(x12_max)
        x12 = x12_avg + x12_max
        max_x1, _ = torch.max(x12 * x1, dim=1, keepdim=True)
        mean_x1 = torch.mean(x12 * x1, dim=1, keepdim=True)
        x1_c = self.conv(torch.cat([max_x1, mean_x1], dim=1))
        x1_m = x1_c * x1
        max_x2, _ = torch.max(x12 * x2, dim=1, keepdim=True)
        mean_x2 = torch.mean(x12 * x2, dim=1, keepdim=True)
        x2_c = self.conv(torch.cat([max_x2, mean_x2], dim=1))
        x2_m = x2_c * x2
        y = torch.cat([x1_m, x2_m],dim=1)
        return y
        
class Contrast(nn.Module):
    def __init__(self):
        super(Contrast, self).__init__()
        self.conv1 = conv_block(in_features=512,
                                out_features=512,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type=False,
                                activation=True)
        self.conv3 = conv_block(in_features=512,
                                out_features=512,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type=False,
                                activation=True)
        self.small_decoder = nn.Sequential(nn.Conv2d(512, 128, 3, stride=1, padding=1),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(128, 1, 1, stride=1, padding=0))
    def forward(self, g1, g2):
            g_1 = self.conv3(self.conv1(g1) * g1)
            g_2 = self.conv3(self.conv1(g2) * g2)
            po = torch.cat([g_1,g_2],dim=0)
            neg = torch.cat([g2 * g_1, g1 * g_2],dim=0)
            pred_contrast = torch.cat([po, neg],dim=0)
            pred_contrast = self.small_decoder(pred_contrast)
            pred_contrast = F.interpolate(pred_contrast, size=(256, 256), mode='bilinear', align_corners=True)
            return pred_contrast

            
class CoAttLayer(nn.Module):
    def __init__(self):
        super(CoAttLayer, self).__init__()

        self.H_attention = AttentionModule(256)
        self.h_attention = AttentionModule(320)
        self.contrast = Contrast()
        self.backatt4 = BackAttention(512)
        self.backatt3 = BackAttention(320)
        drop_path=0.
        depth=4
        #build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=256,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=16,
                expand=4,
                input_resolution=(16,16),is_light_sr=False))
        
        self.conv512_320 = nn.Conv2d(512, 320, kernel_size=3, stride=1, padding=1)
        self.conv640_128 = nn.Conv2d(640, 128, kernel_size=3, stride=1, padding=1)
        self.conv512_128 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.conv512_64 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.conv64_512 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.conv64_128 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv128_64 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv256_64 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv128_512 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1)
        self.conv256_512 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv256_64 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv320_512 = nn.Conv2d(320, 512, kernel_size=3, stride=1, padding=1)
        self.conv1 = conv_block(in_features=512,
                                out_features=512,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type=False,
                                activation=True)
        self.conv3 = conv_block(in_features=512,
                                out_features=512,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type=False,
                                activation=True)

    def forward(self, x4, x3, gts):
        f4_begin = 0
        f4_end = int(x4.shape[0] / 2)
        s4_begin = f4_end
        s4_end = int(x4.shape[0])
        x4_1 = x4[f4_begin: f4_end]
        x4_2 = x4[s4_begin: s4_end]

        f3_begin = 0
        f3_end = int(x3.shape[0] / 2)
        s3_begin = f3_end
        s3_end = int(x3.shape[0])
        x3_1 = x3[f3_begin: f3_end]
        x3_2 = x3[s3_begin: s3_end]
        
        s41_1, s42_1 = torch.chunk(x4_1, 2, dim=1)
        s41_2, s42_2 = torch.chunk(x4_2, 2, dim=1)
        H_new_1 = self.H_attention(s41_1, s42_1)  # GAM
        H_new_2 = self.H_attention(s41_2, s42_2)  # GAM
        H_new_1 = F.interpolate(self.conv512_320(H_new_1), size=x3_1.shape[2:], mode='bilinear', align_corners=False)
        H_new_2 = F.interpolate(self.conv512_320(H_new_2), size=x3_2.shape[2:], mode='bilinear', align_corners=False)
        H_new1 = self.h_attention(x3_1,H_new_1)
        H_new2 = self.h_attention(x3_2,H_new_2)
        new = torch.cat([H_new1, H_new2], dim=0)

        att4 = self.backatt4(x4)
        att4 = F.interpolate(self.conv512_320(att4), size=x3.shape[2:], mode='bilinear', align_corners=False)
        att43 = (att4 * x3) + att4
        att = self.backatt3(att43)
        att = self.conv320_512(att)
        new = att * new
        for block in self.blocks:
            new1,new2 = torch.chunk(new, 2, dim=1)
            new12 = block(new1,new2)
        new12 = att * new12
        if self.training:
            pred_contrast = self.contrast(H_new1,H_new2)
            gts = F.interpolate(gts, size=new.shape[2:], mode='bilinear', align_corners=False)
            new_p = self.conv3(self.conv1(new))
            new_gt = self.conv3(self.conv1(new * gts))
            new_n = self.conv3(self.conv1(new * (1 - gts)))
            return new_p, new_gt, new_n, new12, pred_contrast
        return new12

class BackAttention(nn.Module):
    def __init__(self, input_channels=512):
        super(BackAttention, self).__init__()
        self.query_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.key_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.value_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Conv2d(input_channels, input_channels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv(x)
        x_query = self.query_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        # x_key: C,BHW
        x_key = self.key_transform(x).view(B, C, -1)  # B, C,HW
        # x_value: BHW, C
        x_value = self.value_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        attention_bmm = torch.bmm(x_query, x_key)*self.scale # B, HW, HW
        attention = F.softmax(attention_bmm, dim=-1)
        attention_sort = torch.sort(attention_bmm, dim=-1, descending=True)[1]
        attention_sort = torch.sort(attention_sort, dim=-1)[1]
        #####
        attention_positive_num = torch.ones_like(attention).cuda()
        attention_positive_num[attention_bmm < 0] = 0
        att_pos_mask = attention_positive_num.clone()
        attention_positive_num = torch.sum(attention_positive_num, dim=-1, keepdim=True).expand_as(attention_sort)
        attention_sort_pos = attention_sort.float().clone()
        apn = attention_positive_num-1
        attention_sort_pos[attention_sort > apn] = 0
        attention_mask = ((attention_sort_pos+1)**3)*att_pos_mask + (1-att_pos_mask)
        out = torch.bmm(attention*attention_mask, x_value) #b,hw,c
        out = out.permute(1, 0, 2)
        out_max = torch.max(out, dim=-1)[0]
        out_avg = torch.mean(out, dim=-1)
        out_co = out_max + out_avg
        #####
        x_co = out_co.view(B, -1) * self.scale
        x_co = F.softmax(x_co, dim=-1)
        x_co = x_co.view(B, H, W).unsqueeze(1)
        out = x * x_co  # b 2048 7 7
        atten = self.avgpool(out)
        atten_1 = torch.sigmoid(self.conv_atten(atten)) * x
        atten_3 = atten_1 + out
        return atten_3
        
# FFM
def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential, nn.ModuleList)):
            weight_init(m)
        elif isinstance(m, (
                nn.ReLU, nn.Sigmoid, nn.SiLU, nn.GELU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.Conv1d,
                nn.Upsample,
                nn.MaxPool2d, nn.Identity, nn.Dropout, nn.ZeroPad2d)):
            pass
        else:
            m.initialize()


class BaseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, bias=True) -> None:
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.basicconv(x)

    def initialize(self):
        weight_init(self)

