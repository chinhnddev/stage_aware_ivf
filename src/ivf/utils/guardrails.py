"""
Guardrails to enforce data constraints:
- No day metadata can be used as model input.
- No segmentation or mask inputs are permitted.
"""

from typing import Iterable, Mapping

from ivf.config.constants import DAY_FEATURE_KEYS, SEGMENTATION_KEYS


def _find_blocked_keys(batch_dict: Mapping, blocked_exact: set, blocked_substrings: set) -> set:
    found = set()
    for key in batch_dict.keys():
        lower_key = str(key).lower()
        if lower_key in blocked_exact:
            found.add(key)
            continue
        for substring in blocked_substrings:
            if substring in lower_key:
                found.add(key)
                break
    return found


def assert_no_day_feature(batch_dict: Mapping) -> None:
    """
    Raise if batch contains any day-related feature keys.
    """
    if batch_dict is None:
        return

    blocked = _find_blocked_keys(batch_dict, {k.lower() for k in DAY_FEATURE_KEYS}, {"day"})
    if blocked:
        raise ValueError(f"Day metadata is not allowed in model inputs: {sorted(blocked)}")


def assert_no_segmentation_inputs(batch_dict: Mapping) -> None:
    """
    Raise if batch contains segmentation or mask inputs.
    """
    if batch_dict is None:
        return

    blocked = _find_blocked_keys(
        batch_dict,
        {k.lower() for k in SEGMENTATION_KEYS},
        {"mask", "seg"},
    )
    if blocked:
        raise ValueError(f"Segmentation inputs are not allowed: {sorted(blocked)}")


def assert_no_hungvuong_training(dataset_types: Iterable[str]) -> None:
    """
    Raise if any training dataset is Hung Vuong (external-only).
    """
    for name in dataset_types:
        if str(name).strip().lower() == "hungvuong":
            raise ValueError("Hung Vuong dataset is external-test only and cannot be used for training.")
