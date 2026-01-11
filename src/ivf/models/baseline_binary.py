"""
Baseline binary classifier for target-only supervised training.
"""

from typing import Iterable, Optional

import torch
from torch import nn
from torchvision import models

from ivf.models.encoder import ConvNeXtMini


def _build_head(in_dim: int, head: str, mlp_hidden: int, dropout: float) -> nn.Module:
    if head == "linear":
        return nn.Linear(in_dim, 1)
    if head == "mlp":
        layers = [nn.Linear(in_dim, mlp_hidden), nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(mlp_hidden, 1))
        return nn.Sequential(*layers)
    raise ValueError(f"Unsupported head: {head}")


class BaselineBinaryClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_mini",
        in_channels: int = 3,
        dims: Optional[Iterable[int]] = None,
        feature_dim: int = 256,
        weights_path: Optional[str] = None,
        head: str = "linear",
        mlp_hidden: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if backbone == "convnext_mini":
            dims = list(dims) if dims is not None else [32, 64, 128]
            self.encoder = ConvNeXtMini(
                in_channels=in_channels,
                dims=dims,
                feature_dim=feature_dim,
                weights_path=weights_path,
            )
            encoder_dim = feature_dim
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=None)
            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(
                    in_channels,
                    resnet.conv1.out_channels,
                    kernel_size=resnet.conv1.kernel_size,
                    stride=resnet.conv1.stride,
                    padding=resnet.conv1.padding,
                    bias=False,
                )
            encoder_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.encoder = resnet
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.head = _build_head(encoder_dim, head=head, mlp_hidden=mlp_hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.head(features)
        return logits.squeeze(-1)
