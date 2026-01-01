"""
Lightning DataModule for multi-phase IVF training.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader

from ivf.data.datasets import BaseImageDataset, collate_batch, make_full_target_dict
from ivf.data.label_schema import (
    QUALITY_TO_ID,
    STAGE_TO_ID,
    QualityLabel,
    StageLabel,
    gardner_to_morphology_targets,
    map_gardner_to_quality,
)
from ivf.data.transforms import assert_no_augmentation, get_eval_transforms, get_train_transforms
from ivf.utils.logging import get_logger


def _normalize_stage(stage_value) -> Optional[StageLabel]:
    if stage_value is None:
        return None
    text = str(stage_value).strip().lower()
    for stage in StageLabel:
        if stage.value == text:
            return stage
    return None


def _normalize_quality(quality_value) -> Optional[QualityLabel]:
    if quality_value is None:
        return None
    text = str(quality_value).strip().lower()
    for label in QualityLabel:
        if label.value == text:
            return label
    return None


def _load_split_df(split_dir: Path, split_name: str) -> pd.DataFrame:
    split_path = split_dir / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_path}")
    return pd.read_csv(split_path)


def _build_morphology_records(df: pd.DataFrame, include_meta_day: bool) -> list:
    records = []
    for _, row in df.iterrows():
        grade = row.get("grade")
        try:
            morph = gardner_to_morphology_targets(grade)
        except ValueError:
            continue

        targets = make_full_target_dict(
            exp=morph["exp"],
            icm=morph["icm"],
            te=morph["te"],
        )
        meta = {
            "id": row.get("id"),
            "dataset": row.get("dataset", "blastocyst"),
            "grade": grade,
        }
        if include_meta_day and "day" in row:
            meta["day"] = row.get("day")
        records.append(
            {
                "image_path": row.get("image_path"),
                "targets": targets,
                "meta": meta,
            }
        )
    return records


def _build_stage_records(df: pd.DataFrame, include_meta_day: bool) -> list:
    records = []
    for _, row in df.iterrows():
        stage_label = _normalize_stage(row.get("stage"))
        if stage_label is None:
            continue
        targets = make_full_target_dict(stage=STAGE_TO_ID[stage_label])
        meta = {
            "id": row.get("id"),
            "dataset": row.get("dataset", "humanembryo2"),
            "stage": stage_label.value,
        }
        if include_meta_day and "day" in row:
            meta["day"] = row.get("day")
        records.append(
            {
                "image_path": row.get("image_path"),
                "targets": targets,
                "meta": meta,
            }
        )
    return records


def _build_quality_records(df: pd.DataFrame, include_meta_day: bool) -> list:
    records = []
    for _, row in df.iterrows():
        quality_label = _normalize_quality(row.get("quality"))
        if quality_label is None:
            quality_label = map_gardner_to_quality(row.get("grade"))
        if quality_label is None:
            continue

        targets = make_full_target_dict(quality=QUALITY_TO_ID[quality_label])
        meta = {
            "id": row.get("id"),
            "dataset": row.get("dataset", "quality_public"),
            "quality": quality_label.value,
            "grade": row.get("grade"),
        }
        if include_meta_day and "day" in row:
            meta["day"] = row.get("day")
        records.append(
            {
                "image_path": row.get("image_path"),
                "targets": targets,
                "meta": meta,
            }
        )
    return records


class IVFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        phase: str,
        splits: Dict[str, Path],
        batch_size: int = 16,
        num_workers: int = 4,
        train_transform_level: str = "light",
        include_meta_day: bool = True,
    ) -> None:
        super().__init__()
        self.phase = phase
        self.splits = splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform_level = train_transform_level
        self.include_meta_day = include_meta_day

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_tf = get_train_transforms(self.train_transform_level)
        eval_tf = get_eval_transforms()
        assert_no_augmentation(eval_tf)
        logger = get_logger("ivf")
        if "hungvuong" in self.splits and self.phase in {"morph", "stage", "joint", "quality"}:
            raise ValueError("Hung Vuong dataset is external-only and cannot be used in training phases.")

        if self.phase == "morph":
            train_df = _load_split_df(self.splits["blastocyst"], "train")
            val_df = _load_split_df(self.splits["blastocyst"], "val")
            self.train_dataset = BaseImageDataset(_build_morphology_records(train_df, self.include_meta_day), transform=train_tf, include_meta_day=self.include_meta_day)
            self.val_dataset = BaseImageDataset(_build_morphology_records(val_df, self.include_meta_day), transform=eval_tf, include_meta_day=self.include_meta_day)
            logger.info("Morphology train size=%s val size=%s", len(self.train_dataset), len(self.val_dataset))
        elif self.phase == "stage":
            train_df = _load_split_df(self.splits["humanembryo2"], "train")
            val_df = _load_split_df(self.splits["humanembryo2"], "val")
            self.train_dataset = BaseImageDataset(_build_stage_records(train_df, self.include_meta_day), transform=train_tf, include_meta_day=self.include_meta_day)
            self.val_dataset = BaseImageDataset(_build_stage_records(val_df, self.include_meta_day), transform=eval_tf, include_meta_day=self.include_meta_day)
            logger.info("Stage train size=%s val size=%s", len(self.train_dataset), len(self.val_dataset))
        elif self.phase == "joint":
            blast_train = _load_split_df(self.splits["blastocyst"], "train")
            blast_val = _load_split_df(self.splits["blastocyst"], "val")
            human_train = _load_split_df(self.splits["humanembryo2"], "train")
            human_val = _load_split_df(self.splits["humanembryo2"], "val")
            train_sets = [
                BaseImageDataset(_build_morphology_records(blast_train, self.include_meta_day), transform=train_tf, include_meta_day=self.include_meta_day),
                BaseImageDataset(_build_stage_records(human_train, self.include_meta_day), transform=train_tf, include_meta_day=self.include_meta_day),
            ]
            val_sets = [
                BaseImageDataset(_build_morphology_records(blast_val, self.include_meta_day), transform=eval_tf, include_meta_day=self.include_meta_day),
                BaseImageDataset(_build_stage_records(human_val, self.include_meta_day), transform=eval_tf, include_meta_day=self.include_meta_day),
            ]
            self.train_dataset = ConcatDataset(train_sets)
            self.val_dataset = ConcatDataset(val_sets)
            logger.info("Joint train size=%s val size=%s", len(self.train_dataset), len(self.val_dataset))
        elif self.phase == "quality":
            train_df = _load_split_df(self.splits["quality"], "train")
            val_df = _load_split_df(self.splits["quality"], "val")
            self.train_dataset = BaseImageDataset(_build_quality_records(train_df, self.include_meta_day), transform=train_tf, include_meta_day=self.include_meta_day)
            self.val_dataset = BaseImageDataset(_build_quality_records(val_df, self.include_meta_day), transform=eval_tf, include_meta_day=self.include_meta_day)
            test_path = self.splits["quality"] / "test.csv"
            if test_path.exists():
                test_df = _load_split_df(self.splits["quality"], "test")
                self.test_dataset = BaseImageDataset(_build_quality_records(test_df, self.include_meta_day), transform=eval_tf, include_meta_day=self.include_meta_day)
            logger.info("Quality train size=%s val size=%s test size=%s", len(self.train_dataset), len(self.val_dataset), len(self.test_dataset) if self.test_dataset else 0)
        else:
            raise ValueError(f"Unsupported phase: {self.phase}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )
