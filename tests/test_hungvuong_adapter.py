import pytest

from ivf.data.adapters.hungvuong import HungVuongDataset


def test_hungvuong_training_mode_raises():
    with pytest.raises(ValueError):
        HungVuongDataset(
            root_dir=".",
            csv_path="dummy.csv",
            image_col="image",
            id_col="id",
            split="train",
        )


def test_hungvuong_training_mode_raises_with_mode_flag():
    with pytest.raises(ValueError):
        HungVuongDataset(
            root_dir=".",
            csv_path="dummy.csv",
            image_col="image",
            id_col="id",
            mode="train",
        )
