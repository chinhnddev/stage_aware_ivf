import pandas as pd
import torch
from PIL import Image

from ivf.data.datamodule import IVFDataModule


def _write_image(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(128, 128, 128))
    image.save(path)


def test_joint_dataloader_shape(tmp_path):
    image_size = 64
    blast_root = tmp_path / "blast"
    human_root = tmp_path / "human"

    _write_image(blast_root / "img1.png", (80, 50))
    _write_image(blast_root / "img2.png", (120, 90))
    _write_image(human_root / "img3.png", (100, 140))
    _write_image(human_root / "img4.png", (60, 110))

    blast_split_dir = tmp_path / "splits" / "blastocyst"
    human_split_dir = tmp_path / "splits" / "humanembryo2"
    blast_split_dir.mkdir(parents=True, exist_ok=True)
    human_split_dir.mkdir(parents=True, exist_ok=True)

    blast_df = pd.DataFrame(
        [
            {"image_path": "blast/img1.png", "grade": "4AA", "id": "b1", "dataset": "blastocyst"},
            {"image_path": "blast/img2.png", "grade": "3AB", "id": "b2", "dataset": "blastocyst"},
        ]
    )
    human_df = pd.DataFrame(
        [
            {"image_path": "human/img3.png", "stage": "cleavage", "id": "h1", "dataset": "humanembryo2"},
            {"image_path": "human/img4.png", "stage": "morula", "id": "h2", "dataset": "humanembryo2"},
        ]
    )

    blast_df.to_csv(blast_split_dir / "train.csv", index=False)
    blast_df.to_csv(blast_split_dir / "val.csv", index=False)
    human_df.to_csv(human_split_dir / "train.csv", index=False)
    human_df.to_csv(human_split_dir / "val.csv", index=False)

    datamodule = IVFDataModule(
        phase="joint",
        splits={"blastocyst": blast_split_dir, "humanembryo2": human_split_dir, "quality": blast_split_dir},
        batch_size=2,
        num_workers=0,
        train_transform_level="light",
        include_meta_day=False,
        root_dirs={"blastocyst": str(tmp_path), "humanembryo2": str(tmp_path)},
        image_size=image_size,
    )
    datamodule.setup()
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    assert isinstance(batch["image"], torch.Tensor)
    assert batch["image"].shape[1:] == (3, image_size, image_size)
