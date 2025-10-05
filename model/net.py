import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from typing import Tuple


from utils.tools import PROJECT_ROOT


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # x: [4, 16384, 64], H: 128, W: 128
    def forward(self, x, H, W):
        # x: [4, 16384, 256]
        x = self.fc1(x)  # 全连接层对embedding进行操作。
        # x: [4, 16384, 256]
        x = self.dwconv(x, H, W)
        x = self.act(x)  # 激活函数
        x = self.drop(x)
        # x: [4, 16384, 64]
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 注意力的每个头对应embedding维数
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # 一个线性层，用于生成q矩阵。
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # x: [4, 16384, 64], H: 128, W: 128
    # x: [4, 4096, 128], H: 64, W: 64
    def forward(self, x, H, W):
        # B(batch_size): 4, N(number of patch): 16384, C(embedding length): 64
        # B: 4, N: 4096, C: 128
        B, N, C = x.shape
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x)
        # q: [4, 16384, 1, 64]
        # q: [4, 4096, 2, 64]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads)
        # q: [4, 1, 16384, 64]
        # q: [4, 2, 4096, 64]
        q = q.permute(0, 2, 1, 3)

        # sr_ratio: 8 将图片缩小sr_ratio倍
        # sr_ratio: 4
        if self.sr_ratio > 1:
            # x_: [4, 64, 128, 128]
            # x_: [4, 128, 64, 64]
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # x_: [4, 256, 64]
            # x_: [4, 256, 128]
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            # kv: [2, 4, 1, 256, 64]
            # kv: [2, 4, 2, 256, 64]
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k: [4, 1, 256, 64], v: [4, 1, 256, 64]
        # k: [4, 2, 256, 64], v: [4, 2, 256, 64]
        k, v = kv[0], kv[1]

        # attn: [4, 1, 16384, 256]
        # attn: [4, 2, 4096, 256]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 最后一个维度做softmax
        attn = self.attn_drop(attn)

        # x: [4, 16384, 64]
        # x: [4, 4096, 128]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # 有概率将整条残差即子网络的输出丢弃。
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))  # 经历MLP层特征形状不会发生改变。

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class FCT(nn.Module):
    def __init__(self, dim=64, decode_dim=1024, hw=128 * 128):
        super(FCT, self).__init__()
        self.dim_o = dim
        a = dim
        dim, decode_dim = hw, hw
        hw = a
        self.decode_dim = decode_dim
        self.weight_q = nn.Linear(dim, decode_dim, bias=False)
        self.weight_k = nn.Linear(dim, decode_dim, bias=False)
        self.weight_alpha = nn.Parameter(torch.randn(hw // 2 + 1, hw // 2 + 1) * 0.02)
        self.proj = nn.Linear(hw, hw)
        self.ac_bn_2 = torch.nn.Sequential(torch.nn.ReLU(), nn.BatchNorm2d(self.dim_o))

    def forward(self, x):  # 【B，C，N】
        raw = x
        B, C, H, W = x.shape
        N = H * W
        x = x.reshape(B, C, N)
        q = self.weight_q(x).transpose(-2, -1)
        k = self.weight_k(x).transpose(-2, -1)
        q = torch.fft.rfft2(q, dim=(-2, -1), norm='ortho')
        k = torch.fft.rfft2(k, dim=(-2, -1), norm='ortho')

        '''
        [B,N,C//2+1]
        '''
        q_r, q_i = q.real.transpose(-2, -1), q.imag.transpose(-2, -1)  # 1, 33,16384
        attn_r = q_r @ k.real  # [N,N] 1,33,33
        attn_i = q_i @ k.imag  # [N,N] 1,33,33
        attn_r = self.weight_alpha * attn_r  # 1025,1025  * 1,33,33
        attn_i = self.weight_alpha * attn_i
        # aa = torch.softmax(attn_r,dim=-1)
        x_r = torch.softmax(attn_r, dim=-1) @ q_i  # [B, N, C] 无softmax 95.7
        x_i = torch.softmax(attn_i, dim=-1) @ q_r  # [B, N, C]
        x = torch.view_as_complex(torch.stack([x_r, x_i], dim=-1)).transpose(-2, -1)
        x = torch.fft.irfft2(x, dim=(-2, -1), norm='ortho')
        x = self.proj(x)
        x = x.reshape(B, C, H, W)
        x = self.ac_bn_2(x)
        return raw + x


def logmax(X, axis=-1):
    X_log = torch.log(X)
    return X_log / X_log.sum(axis, keepdim=True)


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=(64, 128, 256, 512),
                 num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1)):
        super(MixVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def get_embed_dims(self):
        return self.embed_dims

    # x: [4, 3, 512, 512]
    def forward_features(self, x):
        # B: 4
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class MLPDecoder(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, num_classes: int = 2, feature_channels: Tuple[int, ...] = (64, 128, 320, 512),
                 embedding_dim: int = 64):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = feature_channels
        self.linear_c4 = MLPDecoder(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLPDecoder(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLPDecoder(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLPDecoder(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.conv_fuse = nn.Conv2d(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1
        )
        self.dropout = nn.Dropout(p=0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        n, _, h, w = c4.shape
        size = c1.shape[2:]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=size, mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=size, mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=size, mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c = self.conv_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class Channel_Attention_Module_Conv(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(Channel_Attention_Module_Conv, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x)
        max_x = self.max_pooling(x)
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(avg_out + max_out)
        return x * v


class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size=(k, k), stride=(1, 1), padding=((k - 1) // 2, (k - 1) // 2),
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim=1, keepdim=True)
        max_x, _ = self.max_pooling(x, dim=1, keepdim=True)
        v = self.conv(torch.cat((max_x, avg_x), dim=1))
        v = self.sigmoid(v)
        return x * v


class CBAMBlock(nn.Module):
    def __init__(self, channels: int = None, spatial_attention_kernel_size=5):
        super(CBAMBlock, self).__init__()
        self.channel_attention_block = Channel_Attention_Module_Conv(channels=channels, gamma=2, b=1)
        self.spatial_attention_block = Spatial_Attention_Module(k=spatial_attention_kernel_size)

    def forward(self, x):
        x = self.channel_attention_block(x)
        x = self.spatial_attention_block(x)
        return x


class Dblock(nn.Module):
    def __init__(self, channel: int, p: float = 0.2):
        super(Dblock, self).__init__()
        self.nonlinearity = partial(F.relu, inplace=True)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        self.cbam = CBAMBlock(channel * 4)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel * 4, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
        )
        self.dropout_ = nn.Dropout(p=p)
        self.relu3x3 = self.nonlinearity

    def forward(self, x):
        dilate1_out = self.nonlinearity(self.dilate1(x))
        dilate2_out = self.nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = self.nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = self.nonlinearity(self.dilate4(dilate3_out))

        out = torch.cat((dilate1_out, dilate2_out, dilate3_out, dilate4_out), dim=1)
        out = self.cbam(out)
        out = self.conv3x3(out)
        out = self.dropout_(out)
        out = self.relu3x3(out)

        return out


class SegHead_ASPP_MC(nn.Module):
    def __init__(self, num_classes: int = 2, feature_channels: Tuple[int, ...] = (64, 128, 320, 512),
                 embedding_dim: int = 64, p: float = 0.2):
        super(SegHead_ASPP_MC, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = feature_channels
        self.linear_c4 = MLPDecoder(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLPDecoder(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLPDecoder(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLPDecoder(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.conv_fuse = nn.Conv2d(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1
        )

        self.dblock = Dblock(channel=embedding_dim)
        self.dropout_1 = nn.Dropout(p=p)
        self.dropout_2 = nn.Dropout(p=p)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        n, _, h, w = c4.shape
        size = c1.shape[2:]
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=size, mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=size, mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=size, mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        x = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        x = self.conv_fuse(x)
        x = self.dropout_1(x)  # first dropout

        x = self.dblock(x)
        x = self.dropout_2(x)

        out = self.linear_pred(x)
        return out


class Net(nn.Module):
    def __init__(self, num_classes: int = 2, init_weights_path: str = './pretrained/mit_b4.pth'):
        super(Net, self).__init__()
        self.backbone = mit_b4()
        feature_channels = self.backbone.get_embed_dims()
        self.seg_head = SegHead_ASPP_MC(
            num_classes=num_classes,
            feature_channels=feature_channels,
            embedding_dim=64
        )
        try:
            self._init_weights(init_weights_path)
        except FileNotFoundError:
            print("Init weight not found!")

    def get_features(self, x):
        features = self.backbone(x)
        return features

    def get_output(self, features):
        out = self.seg_head(features)
        return out

    def forward(self, x):
        features = self.backbone(x)
        out = self.seg_head(features)
        return out

    def _init_weights(self, path):
        pretrained_dict = torch.load(path)
        model_dict1 = self.backbone.state_dict()
        pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict1}
        model_dict1.update(pretrained_dict1)
        self.backbone.load_state_dict(model_dict1)
        print("successfully loaded!!!!")