import pytest

from ivf.utils.guardrails import assert_no_day_feature, assert_no_segmentation_inputs


def test_day_key_triggers_guardrail():
    with pytest.raises(ValueError):
        assert_no_day_feature({"image": "x", "day": 5})


def test_mask_key_triggers_guardrail():
    with pytest.raises(ValueError):
        assert_no_segmentation_inputs({"image": "x", "segmentation_mask": "mask"})


def test_clean_batch_passes():
    batch = {"image": "x", "label": 1}
    assert assert_no_day_feature(batch) is None
    assert assert_no_segmentation_inputs(batch) is None
