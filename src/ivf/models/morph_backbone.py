"""
Multi-scale morphology backbone with CBAM-lite attention.
"""

from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise separable block with optional multi-scale (3x3 + 5x5) mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        multi_scale: bool = True,
    ) -> None:
        super().__init__()
        self.dw3 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.dw5 = None
        if multi_scale:
            self.dw5 = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                groups=in_channels,
                bias=False,
            )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if (
            in_channels != out_channels or stride != 1
        ) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.dw3(x)
        if self.dw5 is not None:
            out = out + self.dw5(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.pw(out)
        out = self.bn2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        return self.act(out)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.mlp(self.pool(x)))
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        maxv = x.max(dim=1, keepdim=True).values
        scale = torch.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))
        return x * scale


class CBAMLite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        return self.spatial(x)


class ECAAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) without spatial attention.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = torch.sigmoid(y).transpose(1, 2).unsqueeze(-1)
        return x * y


class MorphologyBackbone(nn.Module):
    """
    Multi-scale CNN backbone producing a global embedding vector.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dims: Iterable[int] = (32, 64, 128),
        feature_dim: int = 256,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        blocks_per_stage: int = 2,
        scale_feature_dim: bool = True,
        fusion_mode: str = "concat",
        attention_reduction: int = 8,
        attention_type: str = "eca",
        attention_kernel: int = 3,
        multi_scale: bool = True,
    ) -> None:
        super().__init__()
        width_mult = float(width_mult)
        depth_mult = float(depth_mult)
        dims = [max(8, int(round(d * width_mult))) for d in list(dims)]
        if scale_feature_dim:
            feature_dim = max(8, int(round(feature_dim * width_mult)))
        self.feature_dim = feature_dim

        stage_blocks = max(1, int(round(blocks_per_stage * depth_mult)))
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        self.stages = nn.ModuleList()
        self.blocks = []
        in_dim = dims[0]
        for out_dim in dims:
            blocks = [DepthwiseSeparableBlock(in_dim, out_dim, stride=2, multi_scale=multi_scale)]
            for _ in range(stage_blocks - 1):
                blocks.append(DepthwiseSeparableBlock(out_dim, out_dim, stride=1, multi_scale=multi_scale))
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            self.blocks.extend(blocks)
            in_dim = out_dim

        mid_channels = dims[-2] if len(dims) >= 2 else dims[-1]
        high_channels = dims[-1]
        if attention_type not in {"cbam", "eca", "none"}:
            raise ValueError(f"Unsupported attention_type: {attention_type}")
        if attention_type == "cbam":
            self.attn_mid = CBAMLite(mid_channels, reduction=attention_reduction) if len(dims) >= 2 else None
            self.attn_high = CBAMLite(high_channels, reduction=attention_reduction)
        elif attention_type == "eca":
            self.attn_mid = ECAAttention(mid_channels, kernel_size=attention_kernel) if len(dims) >= 2 else None
            self.attn_high = ECAAttention(high_channels, kernel_size=attention_kernel)
        else:
            self.attn_mid = None
            self.attn_high = None

        self.fusion_mode = fusion_mode
        if fusion_mode not in {"concat", "sum"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        if fusion_mode == "sum":
            self.f3_align = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
            self.f4_align = nn.Conv2d(high_channels, mid_channels, kernel_size=1, bias=False) if (
                high_channels != mid_channels
            ) else None
            self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float))
            fused_channels = mid_channels
        else:
            self.f3_align = None
            self.f4_align = None
            self.fusion_weights = None
            fused_channels = mid_channels + high_channels

        embed_in = max(mid_channels, high_channels)
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(fused_channels, embed_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_in),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(embed_in, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        features: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        f4 = features[-1]
        f3 = features[-2] if len(features) >= 2 else f4
        if self.attn_mid is not None:
            f3 = self.attn_mid(f3)
        if self.attn_high is not None:
            f4 = self.attn_high(f4)

        if self.fusion_mode == "sum":
            f3_aligned = self.f3_align(f3) if self.f3_align is not None else f3
            f4_aligned = self.f4_align(f4) if self.f4_align is not None else f4
            f4_up = F.interpolate(f4_aligned, size=f3_aligned.shape[-2:], mode="bilinear", align_corners=False)
            weights = torch.softmax(self.fusion_weights, dim=0)
            fused = weights[0] * f3_aligned + weights[1] * f4_up
        else:
            f4_up = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
            fused = torch.cat([f3, f4_up], dim=1)

        fused = self.fusion_proj(fused)
        pooled = self.pool(fused).flatten(1)
        emb = self.proj(pooled)
        return F.normalize(emb, dim=-1)
