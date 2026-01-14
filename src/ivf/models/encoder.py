"""
Lightweight ConvNeXt-inspired encoder (scratch-trained by default).
"""

from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ivf.models.encoder_embryonet_lite import EmbryoNetLite
from ivf.utils.logging import get_logger

class ConvBlock(nn.Module):
    """
    Simple residual conv block with depthwise-style mixing.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if (
            in_channels != out_channels or stride != 1
        ) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.proj is not None:
            identity = self.proj(identity)
        x = x + identity
        return self.act(x)


class ConvNeXtMini(nn.Module):
    """
    Small convolutional encoder producing a pooled feature vector.

    Default initialization is random; optional weights_path can be provided to load
    pretrained weights, but this is off by default.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dims: Iterable[int] = (32, 64, 128),
        feature_dim: int = 256,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        dims = list(dims)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        blocks = []
        in_dim = dims[0]
        for out_dim in dims:
            stride = 2 if out_dim != in_dim else 1
            blocks.append(ConvBlock(in_dim, out_dim, stride=stride))
            blocks.append(ConvBlock(out_dim, out_dim, stride=1))
            in_dim = out_dim
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(dims[-1], feature_dim)
        self.out_dim = int(feature_dim)

        if weights_path:
            self.load_pretrained(weights_path)

    def load_pretrained(self, weights_path: str) -> None:
        state = torch.load(weights_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys when loading pretrained encoder: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys when loading pretrained encoder: {unexpected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


ENCODER_REGISTRY = {
    "convnext_mini": ConvNeXtMini,
    "embryonet_lite": EmbryoNetLite,
}


def _log_embryonet_lite(encoder: EmbryoNetLite, cfg) -> None:
    logger = get_logger("ivf")
    params = sum(p.numel() for p in encoder.parameters())
    width_mult = getattr(cfg, "width_mult", 1.0)
    norm = getattr(cfg, "norm", "bn")
    logger.info("Encoder=EmbryoNetLite (from scratch). Pretrained weights disabled.")
    logger.info(
        "EmbryoNetLite params=%s width_mult=%s norm=%s out_dim=%s",
        params,
        width_mult,
        norm,
        encoder.out_dim,
    )


def build_encoder(cfg) -> nn.Module:
    name = getattr(cfg, "name", None)
    if not name:
        raise ValueError("Encoder name missing in config. Set model.encoder.name explicitly.")
    if name == "convnext_mini":
        return ConvNeXtMini(
            in_channels=cfg.in_channels,
            dims=cfg.dims,
            feature_dim=cfg.feature_dim,
            weights_path=getattr(cfg, "weights_path", None),
        )
    if name == "embryonet_lite":
        weights_path = getattr(cfg, "weights_path", None)
        if weights_path:
            raise RuntimeError("EmbryoNetLite must train from scratch; remove weights_path from config.")
        encoder = EmbryoNetLite(
            in_channels=cfg.in_channels,
            width_mult=getattr(cfg, "width_mult", 1.0),
            feature_dim=getattr(cfg, "feature_dim", 512),
            use_head_conv=getattr(cfg, "use_head_conv", True),
            norm=getattr(cfg, "norm", "bn"),
            gn_groups=getattr(cfg, "gn_groups", 8),
            drop_path_rate=getattr(cfg, "drop_path_rate", 0.05),
            se_ratio=getattr(cfg, "se_ratio", 0.25),
            stages=getattr(cfg, "stages", None),
        )
        _log_embryonet_lite(encoder, cfg)
        return encoder
    valid = ", ".join(sorted(ENCODER_REGISTRY.keys()))
    raise ValueError(f"Unsupported encoder name: {name}. Valid options: {valid}")
