"""
Utilities for freezing encoder parameters.
"""

from typing import Iterable, Tuple


def _iter_encoder_params(model):
    if not hasattr(model, "encoder"):
        raise AttributeError("Model has no encoder attribute to freeze.")
    return list(model.encoder.parameters())


def freeze_encoder(model, ratio: float = 1.0) -> None:
    """
    Freeze a fraction of encoder parameters (prefix order).
    ratio=1.0 freezes all encoder params; ratio=0 leaves encoder trainable.
    """
    params = _iter_encoder_params(model)
    cutoff = int(len(params) * ratio)
    for i, p in enumerate(params):
        p.requires_grad = i >= cutoff


def progressive_unfreeze(model, epoch: int, schedule: Iterable[Tuple[int, float]]) -> float:
    """
    Progressive unfreeze based on epoch thresholds.
    schedule: iterable of (start_epoch, freeze_ratio)
    Returns the applied freeze ratio.
    """
    applied_ratio = None
    for start_epoch, ratio in sorted(schedule, key=lambda x: x[0]):
        if epoch >= start_epoch:
            applied_ratio = ratio
    if applied_ratio is None:
        applied_ratio = 1.0
    freeze_encoder(model, ratio=applied_ratio)
    return applied_ratio
