import pytest

from ivf.utils.guardrails import assert_no_hungvuong_training


def test_hungvuong_training_blocked():
    with pytest.raises(ValueError):
        assert_no_hungvuong_training(["blastocyst", "hungvuong"])
