"""
Lightning DataModule for multi-phase IVF training.
"""

from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ivf.data.datasets import BaseImageDataset, IGNORE_INDEX, collate_batch, make_full_target_dict
from ivf.data.label_schema import (
    EXPANSION_CLASSES,
    ICM_CLASSES,
    QUALITY_TO_ID,
    STAGE_TO_ID,
    TE_CLASSES,
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


def _resolve_group_col(df: pd.DataFrame, split_entry, candidates) -> Optional[str]:
    if isinstance(split_entry, Mapping):
        group_col = split_entry.get("group_col")
        if group_col:
            return group_col
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _assert_no_group_overlap_dfs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    group_col: Optional[str],
    context: str,
) -> None:
    logger = get_logger("ivf")
    if not group_col:
        logger.warning("%s group_col not configured; leakage check skipped.", context)
        return
    missing = []
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if df is None or df.empty:
            continue
        if group_col not in df.columns:
            missing.append(name)
    if missing:
        logger.warning("%s group_col=%s missing in %s split(s); leakage check skipped.", context, group_col, ", ".join(missing))
        return

    def _groups(df: pd.DataFrame) -> set:
        return set(df[group_col].dropna().astype(str))

    groups = {
        "train": _groups(train_df),
        "val": _groups(val_df),
    }
    if test_df is not None and not test_df.empty:
        groups["test"] = _groups(test_df)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if left not in groups or right not in groups:
            continue
        overlap = groups[left].intersection(groups[right])
        if overlap:
            raise ValueError(
                f"{context} group leakage detected between {left}/{right}: {sorted(list(overlap))[:3]}"
            )
    logger.info("%s group leakage check passed for group_col=%s", context, group_col)


def _is_missing_token(value) -> bool:
    if value is None or pd.isna(value):
        return True
    if isinstance(value, (int, float)) and value == 0:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.upper() in {"", "0", "ND", "NA", "N/A"}


def _log_morphology_train_stats(records: list, context: str) -> None:
    logger = get_logger("ivf")
    label_counts = {
        "exp": {exp: 0 for exp in EXPANSION_CLASSES},
        "icm": {cls: 0 for cls in ICM_CLASSES},
        "te": {cls: 0 for cls in TE_CLASSES},
    }
    meta_counts = {
        "icm": {cls: 0 for cls in ICM_CLASSES},
        "te": {cls: 0 for cls in TE_CLASSES},
    }
    mask_counts = {"exp": 0, "icm": 0, "te": 0}
    unique_targets = {"exp": set(), "icm": set(), "te": set()}
    for record in records:
        targets = record.get("targets", {})
        meta = record.get("meta", {})
        exp_mask = targets.get("exp_mask", 0)
        exp_label = targets.get("exp", IGNORE_INDEX)
        if exp_mask:
            mask_counts["exp"] += 1
            if exp_label is not None and exp_label >= 0 and exp_label < len(EXPANSION_CLASSES):
                label_counts["exp"][EXPANSION_CLASSES[int(exp_label)]] += 1
                unique_targets["exp"].add(int(exp_label))
        for head, classes in (("icm", ICM_CLASSES), ("te", TE_CLASSES)):
            meta_value = meta.get(head)
            if meta_value in classes:
                meta_counts[head][meta_value] += 1
            mask = targets.get(f"{head}_mask", 0)
            if not mask:
                continue
            mask_counts[head] += 1
            label = targets.get(head)
            if label is None:
                continue
            try:
                label_int = int(label)
            except (TypeError, ValueError):
                continue
            if label_int < 0:
                continue
            unique_targets[head].add(label_int)
            if 0 <= label_int < len(classes):
                label_counts[head][classes[label_int]] += 1

    logger.info(
        "Morph %s label counts: exp=%s icm=%s te=%s exp_masked=%s icm_masked=%s te_masked=%s",
        context,
        label_counts["exp"],
        label_counts["icm"],
        label_counts["te"],
        mask_counts["exp"],
        mask_counts["icm"],
        mask_counts["te"],
    )
    logger.info(
        "Morph %s unique target indices: exp=%s icm=%s te=%s",
        context,
        sorted(unique_targets["exp"]),
        sorted(unique_targets["icm"]),
        sorted(unique_targets["te"]),
    )
    total = len(records)
    if total > 0:
        logger.info(
            "Morph %s missing rates: exp=%.3f icm=%.3f te=%.3f",
            context,
            1.0 - (mask_counts["exp"] / total),
            1.0 - (mask_counts["icm"] / total),
            1.0 - (mask_counts["te"] / total),
        )

    for head in ("icm", "te"):
        if not unique_targets[head]:
            continue
        expected_num_classes = 3 if meta_counts[head].get("C", 0) > 0 else 2
        max_label = max(unique_targets[head])
        if max_label >= expected_num_classes:
            raise ValueError(
                f"Morph {context} {head} targets out of range: {sorted(unique_targets[head])} "
                f"with num_classes={expected_num_classes}. Check label encoding (e.g., B encoded as 2)."
            )


def _build_morphology_records(
    df: pd.DataFrame,
    include_meta_day: bool,
    context: str,
    drop_missing_icm_te: bool = False,
) -> list:
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
        "dropped_missing_icm_te": 0,
    }
    exp_counts = {exp: 0 for exp in EXPANSION_CLASSES}
    icm_counts = {"A": 0, "B": 0, "C": 0}
    te_counts = {"A": 0, "B": 0, "C": 0}
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
        if exp is None and components is not None:
            exp = components[0]
        if exp is None:
            if _is_missing_token(exp_raw) or (exp_raw is None and components is None):
                stats["missing_exp"] += 1
            else:
                stats["invalid_exp"] += 1
            targets = make_full_target_dict(
                exp=IGNORE_INDEX,
                icm=IGNORE_INDEX,
                te=IGNORE_INDEX,
                exp_mask=0,
                icm_mask=0,
                te_mask=0,
            )
            meta = {
                "id": row.get("id"),
                "dataset": row.get("dataset", "blastocyst"),
                "grade": grade,
                "exp": None,
                "icm": None,
                "te": None,
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
            continue
        exp_counts[exp] += 1
        if exp < 3:
            stats["exp_lt3"] += 1
        icm_norm = normalize_gardner_grade(icm_raw) if has_icm_col else (components[1] if components else None)
        te_norm = normalize_gardner_grade(te_raw) if has_te_col else (components[2] if components else None)
        if exp >= 3:
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
            if drop_missing_icm_te and (icm_norm is None or te_norm is None):
                stats["dropped_missing_icm_te"] += 1
                continue
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
        icm_meta = icm_norm if exp >= 3 and icm_norm in ICM_CLASSES else None
        te_meta = te_norm if exp >= 3 and te_norm in TE_CLASSES else None
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
        if icm_meta in icm_counts:
            icm_counts[icm_meta] += 1
        if te_meta in te_counts:
            te_counts[te_meta] += 1
        if exp in exp_counts:
            exp_counts[exp] += 1
        records.append(
            {
                "image_path": row.get("image_path"),
                "targets": targets,
                "meta": meta,
            }
        )
        stats["kept"] += 1
    stats["exp_label_counts"] = exp_counts
    stats["icm_label_counts"] = icm_counts
    stats["te_label_counts"] = te_counts
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
        morph_labeled_oversample_ratio: float = 0.5,
        morph_balance_icm_te: bool = False,
        morph_labeled_mix_ratio: float = 0.5,
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
        self.morph_labeled_oversample_ratio = morph_labeled_oversample_ratio
        self.morph_balance_icm_te = morph_balance_icm_te
        self.morph_labeled_mix_ratio = morph_labeled_mix_ratio
        self.morph_labeled_idx = []
        self.morph_icm_counts = None
        self.morph_te_counts = None
        self.morph_sample_info = []

        if self.joint_sampling not in {"balanced", "proportional"}:
            raise ValueError(f"Unsupported joint_sampling: {self.joint_sampling}")
        if self.quality_sampling not in {"balanced", "proportional"}:
            raise ValueError(f"Unsupported quality_sampling: {self.quality_sampling}")
        if not 0 <= self.morph_labeled_oversample_ratio <= 1:
            raise ValueError("morph_labeled_oversample_ratio must be in [0,1].")
        if not 0 <= self.morph_labeled_mix_ratio <= 1:
            raise ValueError("morph_labeled_mix_ratio must be in [0,1].")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._train_loader = None
        self._setup_done = False

    def _root_dir(self, key: str) -> Optional[str]:
        return self.root_dirs.get(key)

    def _preflight_check_dataset(self, dataset, name: str) -> None:
        if dataset is None:
            return
        samples_checked = 0
        missing = []

        def _resolve_path(path_value, root_dir):
            if path_value is None:
                return None
            path_str = str(path_value).replace("\\", "/")
            path = Path(path_str)
            if root_dir:
                root = Path(str(root_dir).replace("\\", "/"))
                if not path.is_absolute():
                    if path.parts[: len(root.parts)] == root.parts:
                        return path
                    path = root / path
            return path

        def _iter_samples(ds):
            if isinstance(ds, BaseImageDataset):
                root_dir = getattr(ds, "root_dir", None)
                for sample in ds.samples:
                    yield sample.get("image_path"), root_dir
            elif isinstance(ds, ConcatDataset):
                for subset in ds.datasets:
                    yield from _iter_samples(subset)

        for path_value, root_dir in _iter_samples(dataset):
            if path_value is None:
                continue
            samples_checked += 1
            resolved = _resolve_path(path_value, root_dir)
            if resolved is None:
                continue
            if not resolved.exists():
                missing.append(str(resolved))
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
        if self._setup_done:
            return
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
            logger.warning("Hung Vuong splits present during phase=%s; ignored for training.", self.phase)

        if self.phase == "morph":
            train_df = _load_split_df(self.splits["blastocyst"], "train")
            val_df = _load_split_df(self.splits["blastocyst"], "val")
            try:
                test_df = _load_split_df(self.splits["blastocyst"], "test")
            except FileNotFoundError:
                test_df = None
            group_col = _resolve_group_col(train_df, self.splits["blastocyst"], ("patient_id", "embryo_id"))
            _assert_no_group_overlap_dfs(train_df, val_df, test_df, group_col, context="blastocyst")
            morph_train_records = _build_morphology_records(
                train_df,
                self.include_meta_day,
                context="morph_train",
                drop_missing_icm_te=False,
            )
            labeled_idx = []
            icm_counts = torch.zeros(len(ICM_CLASSES), dtype=torch.long)
            te_counts = torch.zeros(len(TE_CLASSES), dtype=torch.long)
            sample_info = []
            for idx, record in enumerate(morph_train_records):
                targets = record.get("targets", {})
                meta = record.get("meta", {})
                exp_value = meta.get("exp")
                icm_mask = targets.get("icm_mask", 0)
                te_mask = targets.get("te_mask", 0)
                icm_label = targets.get("icm", IGNORE_INDEX)
                te_label = targets.get("te", IGNORE_INDEX)
                if targets.get("exp_mask", 0) == 1 and exp_value is not None and exp_value >= 3:
                    if icm_mask or te_mask:
                        labeled_idx.append(idx)
                if icm_mask and icm_label is not None and icm_label >= 0 and icm_label < icm_counts.numel():
                    icm_counts[int(icm_label)] += 1
                if te_mask and te_label is not None and te_label >= 0 and te_label < te_counts.numel():
                    te_counts[int(te_label)] += 1
                sample_info.append(
                    {
                        "icm_label": icm_label,
                        "icm_mask": bool(icm_mask),
                        "te_label": te_label,
                        "te_mask": bool(te_mask),
                    }
                )
            self.morph_labeled_idx = labeled_idx
            self.morph_icm_counts = icm_counts
            self.morph_te_counts = te_counts
            self.morph_sample_info = sample_info
            logger.info("Morph labeled_idx size=%s", len(labeled_idx))
            _log_morphology_train_stats(morph_train_records, context="morph_train")
            self.train_dataset = BaseImageDataset(
                morph_train_records,
                transform=train_tf,
                include_meta_day=self.include_meta_day,
                root_dir=self._root_dir("blastocyst"),
            )
            self.val_dataset = BaseImageDataset(
                _build_morphology_records(
                    val_df,
                    self.include_meta_day,
                    context="morph_val",
                ),
                transform=eval_tf,
                include_meta_day=self.include_meta_day,
                root_dir=self._root_dir("blastocyst"),
            )
            logger.info("Morphology train size=%s val size=%s", len(self.train_dataset), len(self.val_dataset))
        elif self.phase == "stage":
            train_df = _load_split_df(self.splits["humanembryo2"], "train")
            val_df = _load_split_df(self.splits["humanembryo2"], "val")
            try:
                test_df = _load_split_df(self.splits["humanembryo2"], "test")
            except FileNotFoundError:
                test_df = None
            group_col = _resolve_group_col(train_df, self.splits["humanembryo2"], ("patient_id", "embryo_id"))
            _assert_no_group_overlap_dfs(train_df, val_df, test_df, group_col, context="humanembryo2")
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
            try:
                blast_test = _load_split_df(self.splits["blastocyst"], "test")
            except FileNotFoundError:
                blast_test = None
            group_col = _resolve_group_col(blast_train, self.splits["blastocyst"], ("patient_id", "embryo_id"))
            _assert_no_group_overlap_dfs(blast_train, blast_val, blast_test, group_col, context="blastocyst")
            human_train = _load_split_df(self.splits["humanembryo2"], "train")
            human_val = _load_split_df(self.splits["humanembryo2"], "val")
            try:
                human_test = _load_split_df(self.splits["humanembryo2"], "test")
            except FileNotFoundError:
                human_test = None
            group_col = _resolve_group_col(human_train, self.splits["humanembryo2"], ("patient_id", "embryo_id"))
            _assert_no_group_overlap_dfs(human_train, human_val, human_test, group_col, context="humanembryo2")
            joint_blast_records = _build_morphology_records(
                blast_train,
                self.include_meta_day,
                context="joint_blast_train",
            )
            _log_morphology_train_stats(joint_blast_records, context="joint_blast_train")
            train_sets = [
                BaseImageDataset(
                    joint_blast_records,
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
            try:
                test_df = _load_split_df(self.splits["quality"], "test")
            except FileNotFoundError:
                test_df = None
            group_col = _resolve_group_col(train_df, self.splits["quality"], ("embryo_id", "patient_id"))
            _assert_no_group_overlap_dfs(train_df, val_df, test_df, group_col, context="quality")
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
        self._setup_done = True

    def train_dataloader(self) -> DataLoader:
        if self._train_loader is not None:
            return self._train_loader
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
        if (
            self.phase == "morph"
            and isinstance(self.train_dataset, BaseImageDataset)
            and self.morph_labeled_idx
        ):
            total = len(self.train_dataset)
            labeled = len(self.morph_labeled_idx)
            unlabeled = total - labeled
            if labeled > 0 and unlabeled > 0:
                if self.morph_balance_icm_te and 0 < self.morph_labeled_mix_ratio < 1:
                    ratio = self.morph_labeled_mix_ratio
                    w_labeled = (ratio * unlabeled) / ((1.0 - ratio) * labeled)
                    labeled_set = set(self.morph_labeled_idx)
                    icm_weights = None
                    te_weights = None
                    if self.morph_icm_counts is not None:
                        num_classes = 2 if int(self.morph_icm_counts[2].item()) == 0 else 3
                        head_counts = self.morph_icm_counts[:num_classes].float()
                        total_icm = head_counts.sum().item()
                        if total_icm > 0:
                            icm_weights = total_icm / (num_classes * head_counts)
                            nonzero = icm_weights[icm_weights > 0]
                            if nonzero.numel() > 0:
                                icm_weights = icm_weights / nonzero.mean()
                    if self.morph_te_counts is not None:
                        num_classes = 2 if int(self.morph_te_counts[2].item()) == 0 else 3
                        head_counts = self.morph_te_counts[:num_classes].float()
                        total_te = head_counts.sum().item()
                        if total_te > 0:
                            te_weights = total_te / (num_classes * head_counts)
                            nonzero = te_weights[te_weights > 0]
                            if nonzero.numel() > 0:
                                te_weights = te_weights / nonzero.mean()

                    weights = []
                    for idx in range(total):
                        if idx not in labeled_set:
                            weights.append(1.0)
                            continue
                        info = self.morph_sample_info[idx] if idx < len(self.morph_sample_info) else {}
                        parts = []
                        if icm_weights is not None and info.get("icm_mask") and isinstance(info.get("icm_label"), int):
                            label = info.get("icm_label")
                            if 0 <= label < len(icm_weights):
                                parts.append(float(icm_weights[label]))
                        if te_weights is not None and info.get("te_mask") and isinstance(info.get("te_label"), int):
                            label = info.get("te_label")
                            if 0 <= label < len(te_weights):
                                parts.append(float(te_weights[label]))
                        if parts:
                            weight_factor = sum(parts) / len(parts)
                        else:
                            weight_factor = 1.0
                        weights.append(w_labeled * weight_factor)

                    sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)
                    shuffle = False
                    get_logger("ivf").info(
                        "Morph sampling balanced mix_ratio=%.2f labeled=%s total=%s w_labeled=%.3f",
                        ratio,
                        labeled,
                        total,
                        w_labeled,
                    )
                elif 0 < self.morph_labeled_oversample_ratio < 1:
                    ratio = self.morph_labeled_oversample_ratio
                    w_labeled = (ratio * unlabeled) / ((1.0 - ratio) * labeled)
                    labeled_set = set(self.morph_labeled_idx)
                    weights = [
                        w_labeled if idx in labeled_set else 1.0
                        for idx in range(total)
                    ]
                    sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)
                    shuffle = False
                    get_logger("ivf").info(
                        "Morph sampling oversample_ratio=%.2f labeled=%s total=%s w_labeled=%.3f",
                        ratio,
                        labeled,
                        total,
                        w_labeled,
                    )
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
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )
        self._train_loader = loader
        return loader

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
