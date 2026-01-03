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


class QualityHead(nn.Module):
    """
    Quality head without conditioning.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ConditionedQualityHead(nn.Module):
    """
    Quality head conditioned on a generic context vector.
    """

    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        hidden_dim: int = 128,
        mode: Literal["concat", "film"] = "concat",
    ) -> None:
        super().__init__()
        if cond_dim <= 0:
            raise ValueError("cond_dim must be > 0 for conditioned quality head.")
        self.mode = mode
        if mode == "concat":
            self.net = nn.Sequential(
                nn.Linear(in_dim + cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        elif mode == "film":
            self.cond_embed = nn.Linear(cond_dim, in_dim * 2)
            self.out = nn.Linear(in_dim, 2)
        else:
            raise ValueError(f"Unsupported quality head mode: {mode}")

    def forward(self, features: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            x = torch.cat([features, cond_vec], dim=-1)
            return self.net(x)

        gamma_beta = self.cond_embed(cond_vec)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gated = (1 + gamma) * features + beta
        return self.out(F.gelu(gated))


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
        self._head = ConditionedQualityHead(
            in_dim=in_dim,
            cond_dim=stage_classes,
            hidden_dim=hidden_dim,
            mode=mode,
        )

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

        return self._head(features, stage_probs)
