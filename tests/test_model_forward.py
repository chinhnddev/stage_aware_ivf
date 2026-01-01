import inspect

import torch

from ivf.data.label_schema import EXPANSION_CLASSES, ICM_CLASSES, STAGE_CLASSES, TE_CLASSES
from ivf.models.multitask import MultiTaskEmbryoNet


def test_forward_shapes():
    model = MultiTaskEmbryoNet(feature_dim=128)
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)

    assert "features" in outputs and outputs["features"].shape == (2, 128)
    morph = outputs["morph"]
    assert morph["exp"].shape == (2, len(EXPANSION_CLASSES))
    assert morph["icm"].shape == (2, len(ICM_CLASSES))
    assert morph["te"].shape == (2, len(TE_CLASSES))
    assert outputs["stage"].shape == (2, len(STAGE_CLASSES))
    assert outputs["quality"].shape == (2, 2)


def test_forward_signature_has_no_day_param():
    sig = inspect.signature(MultiTaskEmbryoNet.forward)
    assert "day" not in sig.parameters
