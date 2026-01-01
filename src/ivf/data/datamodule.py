"""
Lightning DataModule for multi-phase IVF training.
"""

from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ivf.data.datasets import BaseImageDataset, collate_batch, make_full_target_dict
from ivf.data.label_schema import (
    QUALITY_TO_ID,
    STAGE_TO_ID,
    QualityLabel,
    StageLabel,
    UNSET,
    gardner_to_morphology_targets,
    is_gardner_range_label,
    map_gardner_to_quality,
    normalize_gardner_exp,
    normalize_gardner_grade,
    parse_gardner_components,
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
    if isinstance(quality_value, (int, float)) and quality_value in {0, 1}:
        return QualityLabel.GOOD if int(quality_value) == 1 else QualityLabel.POOR
    text = str(quality_value).strip().lower()
    if text in {"0", "1"}:
        return QualityLabel.GOOD if text == "1" else QualityLabel.POOR
    for label in QualityLabel:
        if label.value == text:
            return label
    return None


def _load_split_df(split_dir: Union[Path, Mapping[str, str]], split_name: str) -> pd.DataFrame:
    if isinstance(split_dir, Mapping):
        split_path = Path(split_dir[split_name])
    else:
        split_path = split_dir / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_path}")
    return pd.read_csv(split_path)


def _is_missing_token(value) -> bool:
    if value is None or pd.isna(value):
        return True
    if isinstance(value, (int, float)) and value == 0:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.upper() in {"", "0", "ND", "NA", "N/A"}


def _build_morphology_records(df: pd.DataFrame, include_meta_day: bool, context: str) -> list:
    records = []
    has_exp_col = "exp" in df.columns
    has_icm_col = "icm" in df.columns
    has_te_col = "te" in df.columns
    stats = {
        "total": 0,
        "kept": 0,
        "dropped_range_label": 0,
        "missing_exp": 0,
        "invalid_exp": 0,
        "exp_lt3": 0,
        "missing_icm": 0,
        "invalid_icm": 0,
        "missing_te": 0,
        "invalid_te": 0,
    }
    for _, row in df.iterrows():
        stats["total"] += 1
        grade = row.get("grade")
        if is_gardner_range_label(grade):
            stats["dropped_range_label"] += 1
            continue
        exp_raw = row.get("exp") if has_exp_col else None
        icm_raw = row.get("icm") if has_icm_col else None
        te_raw = row.get("te") if has_te_col else None
        components = parse_gardner_components(grade)
        exp = normalize_gardner_exp(exp_raw)
        if exp is None and not has_exp_col and components is not None:
            exp = components[0]
        if exp is None:
            if _is_missing_token(exp_raw) or (exp_raw is None and components is None):
                stats["missing_exp"] += 1
            else:
                stats["invalid_exp"] += 1
            continue
        if exp < 3:
            stats["exp_lt3"] += 1
        if exp >= 3:
            icm_norm = normalize_gardner_grade(icm_raw) if has_icm_col else (components[1] if components else None)
            te_norm = normalize_gardner_grade(te_raw) if has_te_col else (components[2] if components else None)
            if has_icm_col:
                if _is_missing_token(icm_raw):
                    stats["missing_icm"] += 1
                elif icm_norm is None:
                    stats["invalid_icm"] += 1
            else:
                if components is None or components[1] is None:
                    stats["missing_icm"] += 1
            if has_te_col:
                if _is_missing_token(te_raw):
                    stats["missing_te"] += 1
                elif te_norm is None:
                    stats["invalid_te"] += 1
            else:
                if components is None or components[2] is None:
                    stats["missing_te"] += 1
        try:
            morph = gardner_to_morphology_targets(
                grade,
                exp_value=exp,
                icm_value=icm_raw if has_icm_col else UNSET,
                te_value=te_raw if has_te_col else UNSET,
            )
        except ValueError:
            stats["invalid_exp"] += 1
            continue

        targets = make_full_target_dict(
            exp=morph["exp"],
            icm=morph["icm"],
            te=morph["te"],
            exp_mask=morph.get("exp_mask"),
            icm_mask=morph.get("icm_mask"),
            te_mask=morph.get("te_mask"),
        )
        icm_meta = normalize_gardner_grade(icm_raw) if icm_raw is not None else (components[1] if components else None)
        te_meta = normalize_gardner_grade(te_raw) if te_raw is not None else (components[2] if components else None)
        if exp < 3:
            icm_meta = None
            te_meta = None
        meta = {
            "id": row.get("id"),
            "dataset": row.get("dataset", "blastocyst"),
            "grade": grade,
            "exp": exp,
            "icm": icm_meta,
            "te": te_meta,
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
        stats["kept"] += 1
    get_logger("ivf").info("Blastocyst Gardner parsing stats (%s): %s", context, stats)
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


def _coerce_quality_component(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)) and value == 0:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"ND", "NA"}:
        return None
    try:
        num = int(float(text))
    except ValueError:
        return None
    if num == 0:
        return None
    return num


def _derive_quality_from_components(row: pd.Series) -> Optional[QualityLabel]:
    exp = _coerce_quality_component(row.get("exp"))
    icm = _coerce_quality_component(row.get("icm"))
    te = _coerce_quality_component(row.get("te"))
    if exp is None or icm not in {1, 2, 3} or te not in {1, 2, 3}:
        return None
    if exp >= 3 and icm in {1, 2} and te in {1, 2}:
        return QualityLabel.GOOD
    return QualityLabel.POOR


def _build_quality_records(df: pd.DataFrame, include_meta_day: bool, context: str) -> list:
    records = []
    stats = {"total": 0, "dropped": 0, "good": 0, "poor": 0}
    for _, row in df.iterrows():
        stats["total"] += 1
        quality_label = None
        if "quality_label" in df.columns and pd.notna(row.get("quality_label")):
            quality_label = _normalize_quality(row.get("quality_label"))
        if quality_label is None:
            quality_label = _normalize_quality(row.get("quality"))
        if quality_label is None and {"exp", "icm", "te"}.issubset(df.columns):
            quality_label = _derive_quality_from_components(row)
        if quality_label is None and "grade" in df.columns:
            quality_label = map_gardner_to_quality(row.get("grade"))
        if quality_label is None:
            stats["dropped"] += 1
            continue
        if quality_label == QualityLabel.GOOD:
            stats["good"] += 1
        else:
            stats["poor"] += 1

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
    get_logger("ivf").info("Quality records (%s): %s", context, stats)
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
        root_dirs: Optional[Dict[str, str]] = None,
        preflight_samples: int = 5,
        image_size: int = 256,
        normalize: bool = False,
        mean: Optional[list] = None,
        std: Optional[list] = None,
        joint_sampling: str = "balanced",
        quality_sampling: str = "proportional",
    ) -> None:
        super().__init__()
        self.phase = phase
        self.splits = splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform_level = train_transform_level
        self.include_meta_day = include_meta_day
        self.root_dirs = root_dirs or {}
        self.preflight_samples = preflight_samples
        self.image_size = image_size
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.joint_sampling = joint_sampling
        self.quality_sampling = quality_sampling

        if self.joint_sampling not in {"balanced", "proportional"}:
            raise ValueError(f"Unsupported joint_sampling: {self.joint_sampling}")
        if self.quality_sampling not in {"balanced", "proportional"}:
            raise ValueError(f"Unsupported quality_sampling: {self.quality_sampling}")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _root_dir(self, key: str) -> Optional[str]:
        return self.root_dirs.get(key)

    def _preflight_check_dataset(self, dataset, name: str) -> None:
        if dataset is None:
            return
        samples_checked = 0
        missing = []

        def _iter_samples(ds):
            if isinstance(ds, BaseImageDataset):
                for sample in ds.samples:
                    yield sample.get("image_path")
            elif isinstance(ds, ConcatDataset):
                for subset in ds.datasets:
                    yield from _iter_samples(subset)

        for path in _iter_samples(dataset):
            if path is None:
                continue
            samples_checked += 1
            if not Path(path).exists():
                missing.append(str(path))
            if samples_checked >= self.preflight_samples:
                break

        if missing:
            raise ValueError(
                f"Missing image files in {name} dataset (showing up to 3): {missing[:3]}. "
                "Check that image_path entries are POSIX-style and rooted correctly."
            )

    def _preflight_check_shapes(self, dataset, name: str) -> None:
        if dataset is None:
            return
        expected = (3, self.image_size, self.image_size)
        samples_checked = 0
        bad_shapes = []

        def _iter_items(ds):
            if isinstance(ds, BaseImageDataset):
                for i in range(min(self.preflight_samples, len(ds))):
                    yield ds[i]
            elif isinstance(ds, ConcatDataset):
                for subset in ds.datasets:
                    yield from _iter_items(subset)

        for item in _iter_items(dataset):
            tensor = item.get("image")
            shape = tuple(tensor.shape) if hasattr(tensor, "shape") else None
            if shape != expected:
                bad_shapes.append(shape)
            samples_checked += 1
            if samples_checked >= self.preflight_samples:
                break

        if bad_shapes:
            raise ValueError(
                f"Unexpected image tensor shapes in {name} dataset (expected {expected}): {bad_shapes[:3]}"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        train_tf = get_train_transforms(
            self.train_transform_level,
            image_size=self.image_size,
            normalize=self.normalize,
            mean=self.mean,
            std=self.std,
        )
        eval_tf = get_eval_transforms(
            image_size=self.image_size,
            normalize=self.normalize,
            mean=self.mean,
            std=self.std,
        )
        assert_no_augmentation(eval_tf)
        logger = get_logger("ivf")
        if "hungvuong" in self.splits and self.phase in {"morph", "stage", "joint", "quality"}:
            raise ValueError("Hung Vuong dataset is external-only and cannot be used in training phases.")

        if self.phase == "morph":
            train_df = _load_split_df(self.splits["blastocyst"], "train")
            val_df = _load_split_df(self.splits["blastocyst"], "val")
            self.train_dataset = BaseImageDataset(
                _build_morphology_records(train_df, self.include_meta_day, context="morph_train"),
                transform=train_tf,
                include_meta_day=self.include_meta_day,
                root_dir=self._root_dir("blastocyst"),
            )
            self.val_dataset = BaseImageDataset(
                _build_morphology_records(val_df, self.include_meta_day, context="morph_val"),
                transform=eval_tf,
                include_meta_day=self.include_meta_day,
                root_dir=self._root_dir("blastocyst"),
            )
            logger.info("Morphology train size=%s val size=%s", len(self.train_dataset), len(self.val_dataset))
        elif self.phase == "stage":
            train_df = _load_split_df(self.splits["humanembryo2"], "train")
            val_df = _load_split_df(self.splits["humanembryo2"], "val")
            self.train_dataset = BaseImageDataset(
                _build_stage_records(train_df, self.include_meta_day),
                transform=train_tf,
                include_meta_day=self.include_meta_day,
                root_dir=self._root_dir("humanembryo2"),
            )
            self.val_dataset = BaseImageDataset(
                _build_stage_records(val_df, self.include_meta_day),
                transform=eval_tf,
                include_meta_day=self.include_meta_day,
                root_dir=self._root_dir("humanembryo2"),
            )
            logger.info("Stage train size=%s val size=%s", len(self.train_dataset), len(self.val_dataset))
        elif self.phase == "joint":
            blast_train = _load_split_df(self.splits["blastocyst"], "train")
            blast_val = _load_split_df(self.splits["blastocyst"], "val")
            human_train = _load_split_df(self.splits["humanembryo2"], "train")
            human_val = _load_split_df(self.splits["humanembryo2"], "val")
            train_sets = [
                BaseImageDataset(
                    _build_morphology_records(blast_train, self.include_meta_day, context="joint_blast_train"),
                    transform=train_tf,
                    include_meta_day=self.include_meta_day,
                    root_dir=self._root_dir("blastocyst"),
                ),
                BaseImageDataset(
                    _build_stage_records(human_train, self.include_meta_day),
                    transform=train_tf,
                    include_meta_day=self.include_meta_day,
                    root_dir=self._root_dir("humanembryo2"),
                ),
            ]
            val_sets = [
                BaseImageDataset(
                    _build_morphology_records(blast_val, self.include_meta_day, context="joint_blast_val"),
                    transform=eval_tf,
                    include_meta_day=self.include_meta_day,
                    root_dir=self._root_dir("blastocyst"),
                ),
                BaseImageDataset(
                    _build_stage_records(human_val, self.include_meta_day),
                    transform=eval_tf,
                    include_meta_day=self.include_meta_day,
                    root_dir=self._root_dir("humanembryo2"),
                ),
            ]
            self.train_dataset = ConcatDataset(train_sets)
            self.val_dataset = ConcatDataset(val_sets)
            logger.info("Joint train size=%s val size=%s", len(self.train_dataset), len(self.val_dataset))
        elif self.phase == "quality":
            train_df = _load_split_df(self.splits["quality"], "train")
            val_df = _load_split_df(self.splits["quality"], "val")
            self.train_dataset = BaseImageDataset(
                _build_quality_records(train_df, self.include_meta_day, context="quality_train"),
                transform=train_tf,
                include_meta_day=self.include_meta_day,
                root_dir=self._root_dir("quality"),
            )
            self.val_dataset = BaseImageDataset(
                _build_quality_records(val_df, self.include_meta_day, context="quality_val"),
                transform=eval_tf,
                include_meta_day=self.include_meta_day,
                root_dir=self._root_dir("quality"),
            )
            try:
                test_df = _load_split_df(self.splits["quality"], "test")
            except FileNotFoundError:
                test_df = None
            if test_df is not None:
                self.test_dataset = BaseImageDataset(
                    _build_quality_records(test_df, self.include_meta_day, context="quality_test"),
                    transform=eval_tf,
                    include_meta_day=self.include_meta_day,
                    root_dir=self._root_dir("quality"),
                )
            logger.info("Quality train size=%s val size=%s test size=%s", len(self.train_dataset), len(self.val_dataset), len(self.test_dataset) if self.test_dataset else 0)
        else:
            raise ValueError(f"Unsupported phase: {self.phase}")

        self._preflight_check_dataset(self.train_dataset, "train")
        self._preflight_check_dataset(self.val_dataset, "val")
        self._preflight_check_shapes(self.train_dataset, "train")
        self._preflight_check_shapes(self.val_dataset, "val")

    def train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True
        if (
            self.phase == "joint"
            and self.joint_sampling == "balanced"
            and isinstance(self.train_dataset, ConcatDataset)
        ):
            lengths = [len(ds) for ds in self.train_dataset.datasets]
            if all(length > 0 for length in lengths):
                weights = []
                for length in lengths:
                    weights.extend([1.0 / length] * length)
                sampler = WeightedRandomSampler(weights, num_samples=sum(lengths), replacement=True)
                shuffle = False
                get_logger("ivf").info("Joint sampling balanced across datasets: %s", lengths)
        if self.phase == "quality" and self.quality_sampling == "balanced" and isinstance(self.train_dataset, BaseImageDataset):
            labels = [sample.get("targets", {}).get("quality") for sample in self.train_dataset.samples]
            labels = [label for label in labels if label is not None and label >= 0]
            if labels:
                counts = {0: labels.count(0), 1: labels.count(1)}
                if counts.get(0, 0) > 0 and counts.get(1, 0) > 0:
                    weights = []
                    for sample in self.train_dataset.samples:
                        label = sample.get("targets", {}).get("quality")
                        if label is None or label < 0:
                            weights.append(0.0)
                        else:
                            weights.append(1.0 / counts[label])
                    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                    shuffle = False
                    get_logger("ivf").info("Quality sampling balanced: %s", counts)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
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
