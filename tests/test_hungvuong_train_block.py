import pytest

from ivf.utils.guardrails import assert_no_hungvuong_training
from ivf.data.datamodule import IVFDataModule


def test_hungvuong_training_blocked():
    with pytest.raises(ValueError):
        assert_no_hungvuong_training(["blastocyst", "hungvuong"])


def test_datamodule_blocks_hungvuong_in_training():
    splits = {"hungvuong": "data/processed/splits/hungvuong"}
    datamodule = IVFDataModule(
        phase="morph",
        splits=splits,
        batch_size=2,
        num_workers=0,
        train_transform_level="light",
        include_meta_day=False,
    )
    with pytest.raises(ValueError):
        datamodule.setup()
