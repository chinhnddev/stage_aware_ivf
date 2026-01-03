import torch
import torch.nn.functional as F
from torch import nn

import pytest

from ivf.data.label_schema import (
    EXPANSION_CLASSES,
    ICM_CLASSES,
    TE_CLASSES,
    gardner_to_morphology_targets,
)
from ivf.data.datamodule import _log_morphology_train_stats
from ivf.train.lightning_module import MultiTaskLightningModule


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.morph = nn.Identity()
        self.stage = nn.Identity()
        self.quality = nn.Identity()


def _tensorize_targets(targets):
    return {key: torch.tensor([value]) for key, value in targets.items()}


def test_exp_lt3_masks_disable_icm_te_losses():
    targets = gardner_to_morphology_targets("2AA")
    assert targets["exp_mask"] == 1
    assert targets["icm_mask"] == 0
    assert targets["te_mask"] == 0

    targets_tensor = _tensorize_targets(targets)
    outputs = {
        "features": torch.zeros(1, 1),
        "morph": {
            "exp": torch.randn(1, len(EXPANSION_CLASSES)),
            "icm": torch.randn(1, len(ICM_CLASSES)),
            "te": torch.randn(1, len(TE_CLASSES)),
        },
        "stage": torch.zeros(1, 3),
        "quality": torch.zeros(1, 2),
    }

    module = MultiTaskLightningModule(model=_DummyModel(), phase="morph")
    losses = module._compute_losses(outputs, targets_tensor)
    exp_loss = F.cross_entropy(outputs["morph"]["exp"], targets_tensor["exp"])
    assert torch.allclose(losses["morphology"], exp_loss)


def test_missing_icm_te_are_masked():
    targets = gardner_to_morphology_targets(None, exp_value=4, icm_value="ND", te_value="0")
    assert targets["exp_mask"] == 1
    assert targets["icm_mask"] == 0
    assert targets["te_mask"] == 0
    assert targets["icm"] == -1
    assert targets["te"] == -1


def test_range_label_is_rejected():
    with pytest.raises(ValueError):
        gardner_to_morphology_targets("3-4AB")


def test_morph_label_index_guardrail():
    records = [
        {
            "targets": {"icm": 2, "icm_mask": 1, "te": 0, "te_mask": 1},
            "meta": {"icm": "A", "te": "A"},
        }
    ]
    with pytest.raises(ValueError, match="targets out of range"):
        _log_morphology_train_stats(records, context="morph_train")
