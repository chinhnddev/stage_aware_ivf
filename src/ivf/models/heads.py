"""
Heads for multitask IVF model.
"""

from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ivf.data.label_schema import EXPANSION_CLASSES, ICM_CLASSES, STAGE_CLASSES, TE_CLASSES


class MorphologyHeads(nn.Module):
    """
    Three parallel classifiers for EXP, ICM, TE.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        def _make_head(num_classes: int):
            if hidden_dim > 0:
                return nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, num_classes),
                )
            return nn.Linear(in_dim, num_classes)

        self.exp = _make_head(len(EXPANSION_CLASSES))
        self.icm = _make_head(len(ICM_CLASSES))
        self.te = _make_head(len(TE_CLASSES))

    def forward(self, features: torch.Tensor):
        return {
            "exp": self.exp(features),
            "icm": self.icm(features),
            "te": self.te(features),
        }


class StageHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(STAGE_CLASSES))] if hidden_dim > 0 else [nn.Linear(in_dim, len(STAGE_CLASSES))]
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class StageConditionedQualityHead(nn.Module):
    """
    Quality head conditioned on stage probabilities.
    """

    def __init__(
        self,
        in_dim: int,
        stage_classes: int = len(STAGE_CLASSES),
        hidden_dim: int = 128,
        mode: Literal["concat", "film"] = "concat",
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.net = nn.Sequential(
                nn.Linear(in_dim + stage_classes, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        elif mode == "film":
            self.stage_embed = nn.Linear(stage_classes, in_dim * 2)
            self.out = nn.Linear(in_dim, 2)
        else:
            raise ValueError(f"Unsupported quality head mode: {mode}")

    def forward(
        self,
        features: torch.Tensor,
        stage_logits: Optional[torch.Tensor] = None,
        stage_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if stage_probs is None:
            if stage_logits is None:
                raise ValueError("Stage probabilities or logits required for conditioned quality head.")
            stage_probs = F.softmax(stage_logits, dim=-1)

        if self.mode == "concat":
            x = torch.cat([features, stage_probs], dim=-1)
            return self.net(x)

        # FiLM-style modulation
        gamma_beta = self.stage_embed(stage_probs)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gated = (1 + gamma) * features + beta
        return self.out(F.gelu(gated))
