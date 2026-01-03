import logging
import pytest

from ivf.utils.guardrails import assert_no_hungvuong_training
from ivf.data.datamodule import IVFDataModule


def test_hungvuong_training_blocked():
    with pytest.raises(ValueError):
        assert_no_hungvuong_training(["blastocyst", "hungvuong"])


def test_datamodule_warns_hungvuong_in_training(tmp_path, caplog):
    split_dir = tmp_path / "blastocyst"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.csv").write_text("image_path,grade\n", encoding="utf-8")
    (split_dir / "val.csv").write_text("image_path,grade\n", encoding="utf-8")
    splits = {"blastocyst": split_dir, "hungvuong": tmp_path / "hungvuong"}
    datamodule = IVFDataModule(
        phase="morph",
        splits=splits,
        batch_size=2,
        num_workers=0,
        train_transform_level="light",
        include_meta_day=False,
    )
    with caplog.at_level(logging.WARNING):
        datamodule.setup()
    assert "Hung Vuong" in caplog.text
