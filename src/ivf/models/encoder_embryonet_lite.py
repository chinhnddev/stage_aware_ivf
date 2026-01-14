"""
EmbryoNetLite: lightweight MBConv + SE encoder trained from scratch.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _make_divisible(value: float, divisor: int = 8) -> int:
    if divisor <= 0:
        return int(value)
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def _scale_channels(channels: int, width_mult: float, divisor: int = 8) -> int:
    if width_mult == 1.0:
        return channels
    return _make_divisible(channels * width_mult, divisor=divisor)


def _gn_groups(num_channels: int, requested_groups: int) -> int:
    groups = min(requested_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        reduced = max(1, int(in_channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, in_channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        norm_layer: Callable[[int], nn.Module],
        se_ratio: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels

        mid_channels = _make_divisible(in_channels * expand_ratio, divisor=8)
        self.expand = None
        if mid_channels != in_channels:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                norm_layer(mid_channels),
                nn.SiLU(inplace=True),
            )

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=mid_channels,
                bias=False,
            ),
            norm_layer(mid_channels),
            nn.SiLU(inplace=True),
        )

        self.se = SqueezeExcite(mid_channels, se_ratio=se_ratio) if se_ratio and se_ratio > 0 else nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.expand is not None:
            x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = identity + self.drop_path(x)
        return x


class EmbryoNetLite(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        width_mult: float = 1.0,
        feature_dim: int = 512,
        use_head_conv: bool = True,
        norm: str = "bn",
        gn_groups: int = 8,
        drop_path_rate: float = 0.05,
        se_ratio: float = 0.25,
        stages: Optional[Iterable[dict]] = None,
    ) -> None:
        super().__init__()
        self.width_mult = float(width_mult)
        self.norm = norm
        self.drop_path_rate = float(drop_path_rate)
        self.se_ratio = float(se_ratio)

        def _norm_layer(num_channels: int) -> nn.Module:
            if norm == "gn":
                groups = _gn_groups(num_channels, gn_groups)
                return nn.GroupNorm(groups, num_channels)
            if norm != "bn":
                raise ValueError(f"Unsupported norm: {norm}")
            return nn.BatchNorm2d(num_channels)

        stage_cfgs = list(stages) if stages is not None else [
            {"out_channels": 32, "kernel": 3, "stride": 1, "repeats": 2, "expand": 2},
            {"out_channels": 64, "kernel": 5, "stride": 2, "repeats": 3, "expand": 4},
            {"out_channels": 128, "kernel": 5, "stride": 2, "repeats": 4, "expand": 4},
            {"out_channels": 256, "kernel": 3, "stride": 2, "repeats": 2, "expand": 6},
        ]

        stem_out = _scale_channels(32, self.width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_out, kernel_size=3, stride=2, padding=1, bias=False),
            _norm_layer(stem_out),
            nn.SiLU(inplace=True),
        )

        total_blocks = sum(int(stage.get("repeats", 0)) for stage in stage_cfgs)
        if total_blocks <= 1:
            drop_rates = [0.0 for _ in range(max(total_blocks, 1))]
        else:
            drop_rates = [self.drop_path_rate * idx / (total_blocks - 1) for idx in range(total_blocks)]

        blocks = []
        in_ch = stem_out
        block_idx = 0
        for stage in stage_cfgs:
            out_ch = _scale_channels(int(stage["out_channels"]), self.width_mult)
            repeats = int(stage.get("repeats", 1))
            kernel = int(stage.get("kernel", 3))
            stride = int(stage.get("stride", 1))
            expand = float(stage.get("expand", 1))
            for rep in range(repeats):
                block_stride = stride if rep == 0 else 1
                drop_rate = drop_rates[block_idx] if block_idx < len(drop_rates) else 0.0
                blocks.append(
                    MBConv(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=block_stride,
                        expand_ratio=expand,
                        norm_layer=_norm_layer,
                        se_ratio=self.se_ratio,
                        drop_path=drop_rate,
                    )
                )
                in_ch = out_ch
                block_idx += 1

        self.blocks = nn.ModuleList(blocks)
        self.use_head_conv = bool(use_head_conv)
        if self.use_head_conv:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, feature_dim, kernel_size=1, bias=False),
                _norm_layer(feature_dim),
                nn.SiLU(inplace=True),
            )
            self.out_dim = int(feature_dim)
        else:
            self.proj = nn.Identity()
            self.out_dim = int(in_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj(x)
        x = self.pool(x).flatten(1)
        return F.normalize(x, dim=-1)
