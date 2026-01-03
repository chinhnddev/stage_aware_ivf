"""
Train a specific phase of the IVF multitask pipeline.

Usage:
    python scripts/train_phase.py --phase morph --config configs/experiment/base.yaml
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
import random
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import ConcatDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from omegaconf import OmegaConf

from ivf.config import load_experiment_config, resolve_config_dict
from ivf.data.adapters import load_blastocyst_records, records_to_dataframe
from ivf.data.datasets import BaseImageDataset, IGNORE_INDEX
from ivf.data.datamodule import IVFDataModule
from ivf.data.label_schema import (
    EXPANSION_CLASSES,
    ICM_CLASSES,
    TE_CLASSES,
    normalize_gardner_exp,
    normalize_gardner_grade,
    parse_gardner_components,
)
from ivf.data.splits import save_splits, split_by_group
from ivf.models.encoder import ConvNeXtMini
from ivf.models.multitask import MultiTaskEmbryoNet
from ivf.eval import compute_metrics, predict
from ivf.train.callbacks import BestMetricCheckpoint, StepProgressLogger
from ivf.train.lightning_module import MultiTaskLightningModule
from ivf.utils.guardrails import assert_no_hungvuong_training
from ivf.utils.logging import configure_logging
from ivf.utils.paths import ensure_outputs_dir
from ivf.utils.seed import set_global_seed

MISSINGNESS_THRESHOLD = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="Train a phase of the IVF pipeline.")
    parser.add_argument("--phase", required=True, choices=["morph", "stage", "joint", "quality"])
    parser.add_argument("--config", default="configs/experiment/base.yaml", help="Experiment config path.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max training steps.")
    parser.add_argument("--enable_progress_bar", action="store_true", help="Enable progress bar output.")
    parser.add_argument("--disable_progress_bar", action="store_true", help="Disable progress bar output.")
    parser.add_argument("--live_epoch_line", action="store_true", help="Show live epoch progress on a single line.")
    parser.add_argument("--allow_missing_ckpt", action="store_true", help="Allow missing previous checkpoint for later phases.")
    parser.add_argument("--dry_run", action="store_true", help="Validate pipeline without training.")
    return parser.parse_args()


def _split_dir(splits_base_dir: str, config_path: str) -> Path:
    return Path(splits_base_dir) / Path(config_path).stem


def _resolve_split_path(split_entry, split_name: str) -> Path:
    if isinstance(split_entry, dict):
        return Path(split_entry[split_name])
    return Path(split_entry) / f"{split_name}.csv"


def _log_transforms(transforms_cfg, train_transform_level: str, logger) -> None:
    mean = list(transforms_cfg.mean) if transforms_cfg.mean is not None else None
    std = list(transforms_cfg.std) if transforms_cfg.std is not None else None
    logger.info(
        "Transforms: train_level=%s image_size=%s normalize=%s mean=%s std=%s",
        train_transform_level,
        transforms_cfg.image_size,
        transforms_cfg.normalize,
        mean,
        std,
    )


def _safe_load_checkpoint(lightning_module, ckpt_path: Path, logger) -> None:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    current = lightning_module.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    lightning_module.load_state_dict(filtered, strict=False)
    if skipped:
        logger.warning(
            "Loaded checkpoint with %s keys; skipped %s mismatched keys (e.g., %s).",
            len(filtered),
            len(skipped),
            ", ".join(skipped[:3]),
        )


def _summarize_morph_counts(dataset):
    counts = {
        "icm": torch.zeros(len(ICM_CLASSES), dtype=torch.long),
        "te": torch.zeros(len(TE_CLASSES), dtype=torch.long),
    }

    def _iter_samples(ds):
        if isinstance(ds, BaseImageDataset):
            for sample in ds.samples:
                yield sample
        elif isinstance(ds, ConcatDataset):
            for subset in ds.datasets:
                yield from _iter_samples(subset)

    for sample in _iter_samples(dataset):
        targets = sample.get("targets", {})
        for head in ("icm", "te"):
            label = targets.get(head, IGNORE_INDEX)
            mask = targets.get(f"{head}_mask", 0)
            if mask and label is not None and label >= 0:
                if label < counts[head].numel():
                    counts[head][int(label)] += 1

    return counts


def _compute_inverse_freq_weights(counts: torch.Tensor, num_classes: int):
    counts = counts[:num_classes].float()
    total = counts.sum().item()
    if total <= 0:
        return None
    weights = torch.ones_like(counts)
    for i, c in enumerate(counts):
        if c > 0:
            weights[i] = total / (num_classes * c)
        else:
            weights[i] = 0.0
    return weights


def _normalize_split_paths(df: pd.DataFrame, root_dir: Optional[str]) -> pd.DataFrame:
    if "image_path" not in df.columns or not root_dir:
        return df
    root = Path(root_dir)

    def _normalize(value):
        if pd.isna(value):
            return value
        path = Path(str(value).replace("\\", "/"))
        if path.is_absolute():
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        else:
            if path.parts[: len(root.parts)] == root.parts:
                path = Path(*path.parts[len(root.parts) :])
        return path.as_posix()

    df = df.copy()
    df["image_path"] = df["image_path"].apply(_normalize)
    return df


def _resolve_blastocyst_group_col(df: pd.DataFrame, split_cfg: dict) -> Optional[str]:
    group_col = split_cfg.get("group_col")
    if not group_col:
        raise ValueError("Blastocyst split.group_col is required for leakage-safe grouping.")
    if group_col not in df.columns:
        raise ValueError(f"Blastocyst split.group_col '{group_col}' missing from metadata.")
    missing_frac = float(df[group_col].isna().mean())
    if missing_frac > MISSINGNESS_THRESHOLD:
        raise ValueError(
            f"Blastocyst split.group_col '{group_col}' missingness {missing_frac:.1%} exceeds threshold {MISSINGNESS_THRESHOLD:.1%}."
        )
    return group_col


def _extract_morph_labels(row: pd.Series):
    grade = row.get("grade")
    if grade is None:
        grade = row.get("gardner")
    components = parse_gardner_components(grade) if grade is not None else None
    exp = normalize_gardner_exp(row.get("exp"))
    if exp is None and components is not None:
        exp = components[0]
    icm = normalize_gardner_grade(row.get("icm"))
    te = normalize_gardner_grade(row.get("te"))
    if icm is None and components is not None:
        icm = components[1]
    if te is None and components is not None:
        te = components[2]
    return exp, icm, te


def _exp_bucket_mode(exp_counts: Dict[int, int], val_ratio: float, min_val_per_class: int) -> str:
    for exp, count in exp_counts.items():
        if count <= 0:
            continue
        if count * val_ratio < min_val_per_class:
            return "bucket"
    return "exact"


def _split_by_group_stratified(
    df: pd.DataFrame,
    group_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_val_per_class: int,
    logger,
):
    if group_col not in df.columns:
        raise ValueError(f"group_col={group_col} missing from blastocyst metadata.")

    group_ids = []
    for idx, value in df[group_col].items():
        text = str(value).strip()
        if not text or text.lower() == "nan":
            group_ids.append(f"missing_{idx}")
        else:
            group_ids.append(text)

    exp_values = []
    for _, row in df.iterrows():
        exp, _, _ = _extract_morph_labels(row)
        exp_values.append(exp)
    exp_counts = {exp: exp_values.count(exp) for exp in EXPANSION_CLASSES if exp in exp_values}
    exp_mode = _exp_bucket_mode(exp_counts, val_ratio, min_val_per_class)
    if exp_mode == "exact":
        exp_labels_set = [str(exp) for exp in EXPANSION_CLASSES if exp in exp_counts]
    else:
        exp_labels_set = ["1-2", "3-6"]

    exp_labels = []
    icm_labels = []
    te_labels = []
    for _, row in df.iterrows():
        exp, icm, te = _extract_morph_labels(row)
        if exp is None:
            exp_label = "missing"
        elif exp_mode == "exact":
            exp_label = str(exp)
        else:
            exp_label = "1-2" if exp < 3 else "3-6"
        exp_labels.append(exp_label)

        if exp is None or exp < 3:
            icm_label = "missing"
            te_label = "missing"
        else:
            icm_label = icm if icm in ICM_CLASSES else "missing"
            te_label = te if te in TE_CLASSES else "missing"
        icm_labels.append(icm_label)
        te_labels.append(te_label)

    def _counts(values):
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return counts

    exp_counts = _counts(exp_labels)
    icm_counts = _counts(icm_labels)
    te_counts = _counts(te_labels)
    if "missing" in exp_counts and "missing" not in exp_labels_set:
        exp_labels_set.append("missing")
    icm_labels_set = [cls for cls in ICM_CLASSES if cls in icm_counts]
    if "missing" in icm_counts:
        icm_labels_set.append("missing")
    te_labels_set = [cls for cls in TE_CLASSES if cls in te_counts]
    if "missing" in te_counts:
        te_labels_set.append("missing")

    total_counts = _morph_label_distribution(df)
    for head in ("icm", "te"):
        for cls in ("A", "B"):
            total = total_counts[head].get(cls, 0)
            if total > 0 and total < min_val_per_class:
                logger.warning(
                    "Blastocyst split: not enough %s=%s samples for min_val_per_class=%s (total=%s).",
                    head,
                    cls,
                    min_val_per_class,
                    total,
                )

    group_info = {}
    for idx, gid in enumerate(group_ids):
        info = group_info.setdefault(
            gid,
            {
                "rows": 0,
                "exp": {label: 0 for label in exp_labels_set},
                "icm": {label: 0 for label in icm_labels_set},
                "te": {label: 0 for label in te_labels_set},
            },
        )
        info["rows"] += 1
        info["exp"][exp_labels[idx]] = info["exp"].get(exp_labels[idx], 0) + 1
        info["icm"][icm_labels[idx]] = info["icm"].get(icm_labels[idx], 0) + 1
        info["te"][te_labels[idx]] = info["te"].get(te_labels[idx], 0) + 1

    def _scale_counts(counts, ratio):
        return {label: int(round(value * ratio)) for label, value in counts.items()}

    target_val = int(round(len(df) * val_ratio))
    target_test = int(round(len(df) * test_ratio))
    target_counts_val = {
        "exp": _scale_counts(exp_counts, val_ratio),
        "icm": _scale_counts(icm_counts, val_ratio),
        "te": _scale_counts(te_counts, val_ratio),
    }
    target_counts_test = {
        "exp": _scale_counts(exp_counts, test_ratio),
        "icm": _scale_counts(icm_counts, test_ratio),
        "te": _scale_counts(te_counts, test_ratio),
    }

    def _init_counts():
        return {
            "exp": {label: 0 for label in exp_labels_set},
            "icm": {label: 0 for label in icm_labels_set},
            "te": {label: 0 for label in te_labels_set},
        }

    def _score(current, group_counts, target, weights, target_rows, next_rows):
        score = 0.0
        for key in current:
            weight = weights.get(key, 1.0)
            for label, value in current[key].items():
                new_val = value + group_counts[key].get(label, 0)
                score += weight * abs(new_val - target[key].get(label, 0))
        if target_rows > 0:
            score += abs(next_rows - target_rows) / target_rows
        return score

    def _select_groups(target_rows, target_counts, candidates, enforce_min):
        selected = set()
        current = _init_counts()
        current_rows = 0
        weights = {"exp": 1.0, "icm": 2.0, "te": 2.0}

        def _add_group(gid):
            nonlocal current_rows
            selected.add(gid)
            current_rows += group_info[gid]["rows"]
            for key in current:
                for label in current[key]:
                    current[key][label] += group_info[gid][key].get(label, 0)

        remaining = set(candidates)
        if enforce_min:
            for head in ("icm", "te"):
                for cls in ("A", "B"):
                    if total_counts[head].get(cls, 0) < min_val_per_class:
                        continue
                    if current[head].get(cls, 0) >= min_val_per_class:
                        continue
                    sorted_groups = sorted(
                        remaining,
                        key=lambda gid: group_info[gid][head].get(cls, 0),
                        reverse=True,
                    )
                    for gid in sorted_groups:
                        if group_info[gid][head].get(cls, 0) == 0:
                            break
                        if gid in remaining:
                            _add_group(gid)
                            remaining.discard(gid)
                        if current[head].get(cls, 0) >= min_val_per_class:
                            break

        candidate_list = sorted(remaining)
        while candidate_list and current_rows < target_rows:
            best_gid = None
            best_score = None
            for gid in candidate_list:
                next_rows = current_rows + group_info[gid]["rows"]
                score = _score(current, group_info[gid], target_counts, weights, target_rows, next_rows)
                if best_score is None or score < best_score:
                    best_score = score
                    best_gid = gid
            if best_gid is None:
                break
            _add_group(best_gid)
            candidate_list.remove(best_gid)

        return selected, current

    group_ids_unique = list(dict.fromkeys(group_ids))
    rng = random.Random(seed)
    rng.shuffle(group_ids_unique)

    val_groups, _ = _select_groups(target_val, target_counts_val, group_ids_unique, enforce_min=True)
    remaining = [gid for gid in group_ids_unique if gid not in val_groups]
    test_groups = set()
    if test_ratio > 0 and remaining:
        test_groups, _ = _select_groups(target_test, target_counts_test, remaining, enforce_min=False)
        remaining = [gid for gid in remaining if gid not in test_groups]

    df = df.copy()
    df["_group_id"] = group_ids
    val_df = df[df["_group_id"].isin(val_groups)].copy()
    test_df = df[df["_group_id"].isin(test_groups)].copy() if test_ratio > 0 else df.iloc[0:0].copy()
    train_df = df[~df["_group_id"].isin(val_groups.union(test_groups))].copy()
    for split_df in (train_df, val_df, test_df):
        if "_group_id" in split_df.columns:
            split_df.drop(columns=["_group_id"], inplace=True)

    val_counts = _morph_label_distribution(val_df)
    unmet = []
    for head in ("icm", "te"):
        for cls in ("A", "B"):
            total = total_counts[head].get(cls, 0)
            if total >= min_val_per_class and val_counts[head].get(cls, 0) < min_val_per_class:
                unmet.append(f"{head}={cls}")
    if unmet:
        logger.warning(
            "Stratified split could not satisfy min_val_per_class=%s for %s.",
            min_val_per_class,
            ", ".join(unmet),
        )
        return None

    logger.info("Stratified split mode=%s val=%s test=%s", exp_mode, len(val_df), len(test_df))
    return {"train": train_df, "val": val_df, "test": test_df} if test_ratio > 0 else {"train": train_df, "val": val_df}


def _ensure_blastocyst_splits(blast_cfg, split_entry, seed: int, logger):
    split_cfg = blast_cfg.get("split") or {}
    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    test_ratio = float(split_cfg.get("test_ratio", 0.0))
    split_seed = int(split_cfg.get("seed", seed))
    stratified = bool(split_cfg.get("stratified", False))
    min_val_per_class = int(split_cfg.get("min_val_per_class", 3))
    required = ["train", "val"]
    if test_ratio > 0:
        required.append("test")

    if isinstance(split_entry, dict):
        split_entry = {name: str(path) for name, path in split_entry.items()}
        missing = [name for name in required if name not in split_entry]
        if missing:
            raise ValueError(f"Blastocyst split_files missing entries: {missing}")
        for name, path_str in split_entry.items():
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"Blastocyst split file not found: {path}")
        logger.info("Using fixed blastocyst split files: %s", split_entry)
        return split_entry

    split_dir = Path(split_entry)
    split_paths = {name: split_dir / f"{name}.csv" for name in required}
    existing = {name: path.exists() for name, path in split_paths.items()}
    if all(existing[name] for name in required):
        logger.info("Using existing blastocyst splits at %s", split_dir)
        return split_dir
    if any(existing[name] for name in required):
        missing = [name for name in required if not existing[name]]
        raise FileNotFoundError(
            f"Blastocyst split files missing ({missing}) in {split_dir}; "
            "remove existing files to regenerate a full set."
        )

    logger.info("Generating blastocyst splits at %s", split_dir)
    day_col = blast_cfg.get("day_col") if blast_cfg.get("include_meta_day", True) else None
    records, stats = load_blastocyst_records(
        Path(blast_cfg["root_dir"]),
        Path(blast_cfg["csv_path"]),
        image_col=blast_cfg["image_col"],
        grade_col=blast_cfg["label_col"],
        id_col=blast_cfg["id_col"],
        day_col=day_col,
        return_stats=True,
        log_stats=False,
    )
    logger.info("Blastocyst split generation stats: %s", stats.as_dict())
    df = records_to_dataframe(records)
    df = _normalize_split_paths(df, blast_cfg.get("root_dir"))
    group_col = _resolve_blastocyst_group_col(df, split_cfg)
    splits = None
    if stratified:
        logger.info("Using stratified blastocyst split (min_val_per_class=%s).", min_val_per_class)
        try:
            splits = _split_by_group_stratified(
                df,
                group_col=group_col,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=split_seed,
                min_val_per_class=min_val_per_class,
                logger=logger,
            )
        except ValueError as exc:
            logger.warning("Stratified split failed (%s); falling back to split_by_group.", exc)
            splits = None
        if splits is None:
            logger.warning("Falling back to non-stratified group split.")

    if splits is None:
        splits = split_by_group(
            df,
            group_col=group_col,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed,
        )
    save_splits(splits, output_dir=split_dir)
    logger.info("Saved blastocyst splits to %s", split_dir)
    return split_dir


def _morph_label_distribution(df: pd.DataFrame) -> dict:
    exp_counts = {exp: 0 for exp in EXPANSION_CLASSES}
    icm_counts = {cls: 0 for cls in ICM_CLASSES}
    te_counts = {cls: 0 for cls in TE_CLASSES}
    n_exp_labeled = 0
    n_icm_labeled = 0
    n_te_labeled = 0
    for _, row in df.iterrows():
        grade = row.get("grade")
        components = parse_gardner_components(grade) if grade is not None else None
        exp = normalize_gardner_exp(row.get("exp"))
        if exp is None and components is not None:
            exp = components[0]
        if exp is None:
            continue
        n_exp_labeled += 1
        if exp in exp_counts:
            exp_counts[exp] += 1
        if exp < 3:
            continue
        icm = normalize_gardner_grade(row.get("icm"))
        te = normalize_gardner_grade(row.get("te"))
        if icm is None and components is not None:
            icm = components[1]
        if te is None and components is not None:
            te = components[2]
        if icm in icm_counts:
            icm_counts[icm] += 1
            n_icm_labeled += 1
        if te in te_counts:
            te_counts[te] += 1
            n_te_labeled += 1
    return {
        "exp": exp_counts,
        "icm": icm_counts,
        "te": te_counts,
        "n_exp_labeled": n_exp_labeled,
        "n_icm_labeled": n_icm_labeled,
        "n_te_labeled": n_te_labeled,
    }


def _log_blastocyst_split_stats(split_entry, logger) -> None:
    for split_name in ("train", "val", "test"):
        split_path = _resolve_split_path(split_entry, split_name)
        if not split_path.exists():
            if split_name == "test":
                logger.info("Blastocyst test split not found; skipping %s", split_path)
                continue
            raise FileNotFoundError(f"Missing blastocyst split: {split_path}")
        df = pd.read_csv(split_path)
        counts = _morph_label_distribution(df)
        logger.info(
            "Blastocyst %s split path=%s rows=%s n_exp_labeled=%s n_icm_labeled=%s n_te_labeled=%s",
            split_name,
            split_path,
            len(df),
            counts["n_exp_labeled"],
            counts["n_icm_labeled"],
            counts["n_te_labeled"],
        )
        logger.info(
            "Blastocyst %s label distribution exp=%s icm=%s te=%s",
            split_name,
            counts["exp"],
            counts["icm"],
            counts["te"],
        )


def _coerce_quality_label(value):
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, (int, float)) and value in {0, 1}:
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "good"}:
        return 1
    if text in {"0", "poor"}:
        return 0
    return None


def _coerce_quality_component(value):
    if value is None or (isinstance(value, float) and value != value):
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


def _derive_quality_label_from_row(row) -> Optional[int]:
    exp = _coerce_quality_component(row.get("exp"))
    icm = _coerce_quality_component(row.get("icm"))
    te = _coerce_quality_component(row.get("te"))
    if exp is None or icm not in {1, 2, 3} or te not in {1, 2, 3}:
        return None
    return 1 if exp >= 3 and icm in {1, 2} and te in {1, 2} else 0


def _derive_quality_labels(df: pd.DataFrame, logger) -> pd.DataFrame:
    if not {"exp", "icm", "te"}.issubset(df.columns):
        raise ValueError("Quality CSV must contain exp, icm, te columns to derive labels.")
    exp_vals = df["exp"].apply(_coerce_quality_component)
    icm_vals = df["icm"].apply(_coerce_quality_component)
    te_vals = df["te"].apply(_coerce_quality_component)
    valid_mask = exp_vals.notna() & icm_vals.notna() & te_vals.notna()
    df = df.copy()
    df["quality_label"] = np.nan
    good_mask = (exp_vals >= 3) & icm_vals.isin([1, 2]) & te_vals.isin([1, 2])
    df.loc[valid_mask, "quality_label"] = np.where(good_mask[valid_mask], 1, 0)
    df.loc[valid_mask, "quality"] = np.where(good_mask[valid_mask], "good", "poor")

    dropped = int((~valid_mask).sum())
    logger.info(
        "Derived quality_label from exp/icm/te. Dropped missing: exp=%s icm=%s te=%s total=%s",
        int(exp_vals.isna().sum()),
        int(icm_vals.isna().sum()),
        int(te_vals.isna().sum()),
        dropped,
    )
    return df[df["quality_label"].notna()].copy()


def _quality_counts(split_path: Path):
    df = pd.read_csv(split_path)
    if "quality_label" in df.columns:
        labels = df["quality_label"].apply(_coerce_quality_label)
    elif "quality" in df.columns:
        labels = df["quality"].apply(_coerce_quality_label)
    elif {"exp", "icm", "te"}.issubset(df.columns):
        labels = df.apply(_derive_quality_label_from_row, axis=1)
    else:
        labels = pd.Series([], dtype=float)
    labels = labels.dropna().astype(int)
    counts = {0: int((labels == 0).sum()), 1: int((labels == 1).sum())}
    return counts


def _split_random(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0
    n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
    if val_ratio > 0 and n_val == 0:
        n_val = 1
    if test_ratio > 0 and n_test == 0:
        n_test = 1
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough samples to create train/val/test splits.")
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy() if test_ratio > 0 else df.iloc[0:0].copy()
    return train_df, val_df, test_df


def _split_grouped(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    groups = df[group_col].astype(str)
    group_ids = []
    for idx, value in groups.items():
        text = str(value).strip()
        if not text or text.lower() == "nan":
            group_ids.append(f"missing_{idx}")
        else:
            group_ids.append(text)
    df = df.copy()
    df["_group_id"] = group_ids
    unique_groups = list(dict.fromkeys(group_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    n_groups = len(unique_groups)
    n_val = int(round(n_groups * val_ratio)) if val_ratio > 0 else 0
    n_test = int(round(n_groups * test_ratio)) if test_ratio > 0 else 0
    if val_ratio > 0 and n_val == 0:
        n_val = 1
    if test_ratio > 0 and n_test == 0:
        n_test = 1
    n_train = n_groups - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough groups to create train split.")
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])
    train_df = df[df["_group_id"].isin(train_groups)].copy()
    val_df = df[df["_group_id"].isin(val_groups)].copy()
    test_df = df[df["_group_id"].isin(test_groups)].copy() if test_ratio > 0 else df.iloc[0:0].copy()
    for split_df in (train_df, val_df, test_df):
        if "_group_id" in split_df.columns:
            split_df.drop(columns=["_group_id"], inplace=True)
    return train_df, val_df, test_df


def _assert_no_group_overlap(splits, group_col: str) -> None:
    groups = []
    for df in splits:
        if df is None or df.empty or group_col not in df.columns:
            groups.append(set())
            continue
        groups.append(set(df[group_col].dropna().astype(str)))
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlap = groups[i].intersection(groups[j])
            if overlap:
                raise ValueError(f"Group leakage detected between splits: {sorted(list(overlap))[:3]}")


def _ensure_quality_splits(quality_cfg, split_entry, seed: int, logger):
    logger.info("EXP-4 quality labels are proxy labels derived from morphology (exp/icm/te) when available.")
    split_cfg = quality_cfg.get("split") or {}
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0:
        raise ValueError("train_ratio must be > 0 for quality splits.")

    if isinstance(split_entry, dict):
        split_paths = {name: Path(path) for name, path in split_entry.items()}
    else:
        split_paths = {
            "train": Path(split_entry) / "train.csv",
            "val": Path(split_entry) / "val.csv",
            "test": Path(split_entry) / "test.csv",
        }

    missing = [name for name, path in split_paths.items() if name != "test" and not path.exists()]
    if missing:
        logger.warning("Quality split files missing (%s); regenerating splits.", ", ".join(missing))
    else:
        logger.info("Regenerating quality splits from %s for EXP-4.", quality_cfg.csv_path)

    df = pd.read_csv(quality_cfg.csv_path)
    total_before = len(df)
    if "image_path" not in df.columns:
        image_col = quality_cfg.get("image_col", "image_path")
        if image_col not in df.columns:
            raise ValueError("Quality CSV must include image_path or the configured image_col.")
        df = df.copy()
        df["image_path"] = df[image_col].astype(str)
    has_components = {"exp", "icm", "te"}.issubset(df.columns)
    has_quality = "quality_label" in df.columns or "quality" in df.columns
    if has_components:
        df = _derive_quality_labels(df, logger)
        if df.empty:
            raise ValueError("No samples remain after deriving quality labels.")
        total_after = len(df)
        counts = df["quality_label"].value_counts(dropna=False).to_dict()
        logger.info(
            "Quality derive summary: before=%s after=%s good=%s poor=%s",
            total_before,
            total_after,
            counts.get(1, 0),
            counts.get(0, 0),
        )
    elif has_quality:
        logger.info("Using existing quality labels (not derived) from %s.", quality_cfg.csv_path)
        df = df.copy()
        if "quality_label" not in df.columns:
            df["quality_label"] = df["quality"].apply(_coerce_quality_label)
        df = df[df["quality_label"].notna()].copy()
        total_after = len(df)
        if total_after == 0:
            raise ValueError("No samples remain after filtering existing quality labels.")
        counts = df["quality_label"].value_counts(dropna=False).to_dict()
        logger.info(
            "Quality label summary: before=%s after=%s good=%s poor=%s",
            total_before,
            total_after,
            counts.get(1, 0),
            counts.get(0, 0),
        )
    else:
        raise ValueError("Quality CSV must contain exp/icm/te or explicit quality labels.")

    group_col = split_cfg.get("group_col")
    if group_col and group_col in df.columns and df[group_col].notna().any():
        train_df, val_df, test_df = _split_grouped(df, group_col, train_ratio, val_ratio, test_ratio, seed)
        _assert_no_group_overlap([train_df, val_df, test_df], group_col)
    else:
        if group_col:
            logger.warning("group_col=%s missing or empty; falling back to random split.", group_col)
        train_df, val_df, test_df = _split_random(df, train_ratio, val_ratio, test_ratio, seed)
    if train_df.empty or val_df.empty or (test_ratio > 0 and test_df.empty):
        raise ValueError("Quality split generation produced empty train/val/test splits.")

    for name, split_df in {"train": train_df, "val": val_df, "test": test_df}.items():
        path = split_paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(path, index=False)
        counts = split_df["quality_label"].value_counts(dropna=False).to_dict()
        logger.info("Quality %s label distribution: %s", name, counts)
        if counts.get(1, 0) == 0:
            logger.warning("Quality %s split has zero positive samples.", name)

    logger.info("Generated quality splits at %s", split_paths)
    return split_paths


def _tune_quality_threshold(model, dataloader, device: torch.device, reports_dir: Path, logger):
    if dataloader is None:
        logger.warning("No val dataloader found; skipping threshold tuning.")
        return None
    preds = predict(model, dataloader, device)
    probs = np.array(preds["prob_good"], dtype=float)
    y_true = np.array(preds["y_true"], dtype=int)
    if probs.size == 0:
        logger.warning("No predictions available for threshold tuning.")
        return None

    best = {"threshold": 0.5, "f1": -1.0}
    thresholds = np.arange(0.05, 0.96, 0.01)
    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best["f1"]:
            best = {"threshold": float(thresh), "f1": float(f1)}

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "exp04_best_threshold.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    logger.info("EXP-4 best threshold: %s", best)
    logger.info("Saved EXP-4 best threshold to %s", out_path)
    return best


def _run_quality_test_eval(
    model: torch.nn.Module,
    dataloader,
    reports_dir: Path,
    device: torch.device,
    logger,
) -> None:
    if dataloader is None:
        logger.warning("No test dataloader found; skipping EXP-4 test evaluation.")
        return
    model.to(device)
    preds = predict(model, dataloader, device)
    metrics = compute_metrics(preds["prob_good"], preds["y_true"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "exp04_test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("EXP-4 test metrics: %s", metrics)

    pred_df = pd.DataFrame(
        {
            "image_id": preds["image_id"],
            "prob_good": preds["prob_good"],
            "y_true": preds["y_true"],
            "day": preds["day"],
            "split": "test",
        }
    )
    preds_path = reports_dir / "exp04_test_predictions.csv"
    pred_df.to_csv(preds_path, index=False)
    logger.info("Saved EXP-4 test metrics to %s", metrics_path)
    logger.info("Saved EXP-4 test predictions to %s", preds_path)


def build_model(cfg) -> MultiTaskEmbryoNet:
    model_cfg = cfg.model
    encoder_cfg = model_cfg.encoder
    encoder = ConvNeXtMini(
        in_channels=encoder_cfg.in_channels,
        dims=encoder_cfg.dims,
        feature_dim=encoder_cfg.feature_dim,
        weights_path=encoder_cfg.weights_path,
    )
    return MultiTaskEmbryoNet(
        encoder=encoder,
        feature_dim=encoder_cfg.feature_dim,
        quality_mode=model_cfg.heads.quality_mode,
        quality_conditioning=getattr(model_cfg.heads, "quality_conditioning", "morph+stage"),
    )


def get_prev_checkpoint(phase: str, checkpoints_dir: Path, cfg):
    overrides = getattr(cfg.outputs, "checkpoint_paths", {}) or {}
    if phase in overrides:
        return Path(overrides[phase])

    if phase == "stage":
        return checkpoints_dir / "phase1_morph.ckpt"
    if phase == "joint":
        return checkpoints_dir / "phase2_stage.ckpt"
    if phase == "quality":
        joint_ckpt = checkpoints_dir / "phase3_joint.ckpt"
        if joint_ckpt.exists():
            return joint_ckpt
        return checkpoints_dir / "phase2_stage.ckpt"
    return None


def main():
    args = parse_args()
    phase = args.phase
    cfg = load_experiment_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.device is not None:
        cfg.device = args.device

    set_global_seed(cfg.seed, deterministic=True)

    checkpoints_dir = ensure_outputs_dir(cfg.outputs.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ensure_outputs_dir(cfg.outputs.logs_dir)
    logger = configure_logging(logs_dir / "train.log")
    logger.info("Seed=%s", cfg.seed)

    if phase == "morph":
        prev_conditioning = getattr(cfg.model.heads, "quality_conditioning", "morph+stage")
        if prev_conditioning != "none":
            logger.info("Phase morph: forcing quality_conditioning %s -> none.", prev_conditioning)
        cfg.model.heads.quality_conditioning = "none"

    model = build_model(cfg)
    logger.info("Quality conditioning mode: %s", model.quality_conditioning)
    phase_cfg = cfg.training
    loss_weights = resolve_config_dict(phase_cfg.loss_weights)
    if phase == "morph":
        loss_weights = dict(loss_weights or {})
        loss_weights.setdefault("morph", 1.0)
        loss_weights["stage"] = 0.0
        loss_weights["quality"] = 0.0
        logger.info("Morph phase loss weights: %s", loss_weights)
    freeze_cfg = resolve_config_dict(phase_cfg.freeze)
    morph_cfg = getattr(phase_cfg, "morph", None)
    morph_use_class_weights = bool(getattr(morph_cfg, "use_class_weights", False)) if morph_cfg is not None else False
    morph_class_weight_mode = str(getattr(morph_cfg, "class_weight_mode", "inverse_freq")) if morph_cfg is not None else "inverse_freq"
    morph_balance_icm_te = bool(getattr(morph_cfg, "balance_icm_te", False)) if morph_cfg is not None else False
    morph_labeled_mix_ratio = float(getattr(morph_cfg, "labeled_mix_ratio", 0.5)) if morph_cfg is not None else 0.5
    loss_cfg = getattr(phase_cfg, "loss", None)
    loss_use_class_weights = bool(getattr(loss_cfg, "use_class_weights", False)) if loss_cfg is not None else False
    morph_use_class_weights = morph_use_class_weights or loss_use_class_weights

    dataset_types = []
    blast_cfg = OmegaConf.load(cfg.data.blastocyst_config)
    human_cfg = OmegaConf.load(cfg.data.humanembryo2_config)
    quality_cfg = OmegaConf.load(cfg.data.quality_config)
    for dataset_cfg in [blast_cfg, human_cfg, quality_cfg]:
        dataset_types.append(str(dataset_cfg.get("dataset_type", "")))
    assert_no_hungvuong_training(dataset_types)

    data_cfg = cfg.data
    splits_base_dir = data_cfg.splits_base_dir
    splits = {}
    if phase in {"morph", "joint"}:
        blast_split_files = blast_cfg.get("split_files")
        blast_split_entry = (
            {k: str(v) for k, v in blast_split_files.items()}
            if blast_split_files
            else _split_dir(splits_base_dir, data_cfg.blastocyst_config)
        )
        blast_split_entry = _ensure_blastocyst_splits(blast_cfg, blast_split_entry, cfg.seed, logger)
        splits["blastocyst"] = blast_split_entry
        _log_blastocyst_split_stats(blast_split_entry, logger)
        split_cfg = blast_cfg.get("split") or {}
        logger.info(
            "Blastocyst split config: stratified=%s min_val_per_class=%s",
            bool(split_cfg.get("stratified", False)),
            int(split_cfg.get("min_val_per_class", 3)),
        )
    if phase in {"stage", "joint"}:
        splits["humanembryo2"] = _split_dir(splits_base_dir, data_cfg.humanembryo2_config)
    if phase == "quality":
        quality_split_files = quality_cfg.get("split_files")
        splits["quality"] = (
            {k: str(v) for k, v in quality_split_files.items()}
            if quality_split_files
            else _split_dir(splits_base_dir, data_cfg.quality_config)
        )

    quality_pos_weight = None
    if phase == "quality":
        split_paths = _ensure_quality_splits(quality_cfg, splits["quality"], cfg.seed, logger)
        splits["quality"] = {name: str(path) for name, path in split_paths.items()}
        for split_name in ("train", "val", "test"):
            split_path = _resolve_split_path(splits["quality"], split_name)
            if not split_path.exists():
                if split_name == "test":
                    continue
                raise FileNotFoundError(f"Missing quality split: {split_path}")
            counts = _quality_counts(split_path)
            logger.info("Quality %s counts: %s", split_name, counts)
            total = counts.get(0, 0) + counts.get(1, 0)
            if total > 0:
                logger.info(
                    "Quality %s ratio: good=%.3f poor=%.3f",
                    split_name,
                    counts.get(1, 0) / total,
                    counts.get(0, 0) / total,
                )
            if split_name == "train":
                n_pos = counts.get(1, 0)
                n_neg = counts.get(0, 0)
                if n_pos > 0 and n_neg > 0:
                    quality_pos_weight = float(n_neg / n_pos)
                    logger.info("Quality pos_weight set to %.4f (neg=%s pos=%s).", quality_pos_weight, n_neg, n_pos)
                else:
                    logger.warning("Quality train split lacks positives or negatives; pos_weight disabled.")
            elif counts.get(1, 0) == 0:
                logger.warning("Quality %s split has zero positive samples.", split_name)

    lightning_module = MultiTaskLightningModule(
        model=model,
        phase=phase,
        lr=phase_cfg.lr,
        weight_decay=phase_cfg.weight_decay,
        loss_weights=loss_weights,
        freeze_config=freeze_cfg,
        morph_loss_reduction=phase_cfg.morph_loss_reduction,
        quality_pos_weight=quality_pos_weight,
        use_class_weights=morph_use_class_weights,
        class_weight_mode=morph_class_weight_mode,
        live_epoch_line=args.live_epoch_line,
    )

    prev_ckpt = get_prev_checkpoint(phase, checkpoints_dir, cfg)
    if phase in {"stage", "joint", "quality"} and phase_cfg.require_prev_ckpt and not args.allow_missing_ckpt:
        if prev_ckpt is None or not prev_ckpt.exists():
            raise ValueError(
                f"Missing previous checkpoint for phase={phase}. "
                "Run the earlier phase first or pass --allow_missing_ckpt to continue without loading."
            )
    if prev_ckpt is not None and prev_ckpt.exists():
        logger.info("Loading checkpoint weights from %s", prev_ckpt)
        try:
            lightning_module = MultiTaskLightningModule.load_from_checkpoint(
                checkpoint_path=str(prev_ckpt),
                model=model,
                phase=phase,
                lr=phase_cfg.lr,
                weight_decay=phase_cfg.weight_decay,
                loss_weights=loss_weights,
                freeze_config=freeze_cfg,
                morph_loss_reduction=phase_cfg.morph_loss_reduction,
                quality_pos_weight=quality_pos_weight,
                use_class_weights=morph_use_class_weights,
                class_weight_mode=morph_class_weight_mode,
                strict=True,
            )
        except RuntimeError as exc:
            logger.warning("Checkpoint load failed (%s); using safe load with shape filtering.", exc)
            _safe_load_checkpoint(lightning_module, prev_ckpt, logger)
    elif prev_ckpt is not None and not prev_ckpt.exists():
        logger.warning("Previous checkpoint not found at %s; continuing with random initialization.", prev_ckpt)

    transforms_cfg = cfg.transforms
    train_transform_level = getattr(transforms_cfg, phase)
    _log_transforms(transforms_cfg, train_transform_level, logger)
    root_dirs = {
        "blastocyst": blast_cfg.get("root_dir"),
        "humanembryo2": human_cfg.get("root_dir"),
        "quality": quality_cfg.get("root_dir"),
    }

    datamodule = IVFDataModule(
        phase=phase,
        splits=splits,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_transform_level=train_transform_level,
        include_meta_day=data_cfg.include_meta_day_default,
        root_dirs=root_dirs,
        image_size=transforms_cfg.image_size,
        normalize=transforms_cfg.normalize,
        mean=list(transforms_cfg.mean) if transforms_cfg.mean is not None else None,
        std=list(transforms_cfg.std) if transforms_cfg.std is not None else None,
        joint_sampling=phase_cfg.joint_sampling,
        quality_sampling=phase_cfg.quality_sampling,
        morph_labeled_oversample_ratio=float(getattr(phase_cfg, "morph_labeled_oversample_ratio", 0.5)),
        morph_balance_icm_te=morph_balance_icm_te,
        morph_labeled_mix_ratio=morph_labeled_mix_ratio,
    )

    datamodule.setup()
    num_train_batches = None
    log_every_n_steps = 50
    try:
        train_loader = datamodule.train_dataloader()
        num_train_batches = len(train_loader)
        if isinstance(num_train_batches, int) and num_train_batches > 0:
            log_every_n_steps = min(50, max(1, num_train_batches // 2))
    except TypeError:
        pass

    if phase == "morph" and datamodule.train_dataset is not None:
        counts = _summarize_morph_counts(datamodule.train_dataset)
        icm_counts = {cls: int(counts["icm"][i]) for i, cls in enumerate(ICM_CLASSES)}
        te_counts = {cls: int(counts["te"][i]) for i, cls in enumerate(TE_CLASSES)}
        icm_num_classes = 2 if counts["icm"][2] == 0 else 3
        te_num_classes = 2 if counts["te"][2] == 0 else 3
        icm_weights = None
        te_weights = None
        if morph_use_class_weights and morph_class_weight_mode == "inverse_freq":
            icm_weights = _compute_inverse_freq_weights(counts["icm"], icm_num_classes)
            te_weights = _compute_inverse_freq_weights(counts["te"], te_num_classes)
        logger.info(
            "Train setup: num_train_batches=%s log_every_n_steps=%s icm_counts=%s te_counts=%s icm_weights=%s te_weights=%s balance_icm_te=%s labeled_mix_ratio=%.2f",
            num_train_batches,
            log_every_n_steps,
            icm_counts,
            te_counts,
            icm_weights.tolist() if icm_weights is not None else None,
            te_weights.tolist() if te_weights is not None else None,
            morph_balance_icm_te,
            morph_labeled_mix_ratio,
        )

    max_epochs = phase_cfg.epochs.get(phase, 1)
    if args.dry_run:
        logger.info("Dry run complete for phase=%s", phase)
        return

    loggers = []
    if cfg.logging.tensorboard:
        loggers.append(TensorBoardLogger(save_dir=logs_dir / "tensorboard", name=phase))
    if cfg.logging.csv:
        loggers.append(CSVLogger(save_dir=logs_dir / "csv", name=phase))

    accelerator = "cpu"
    devices = 1
    if str(cfg.device).startswith("cuda"):
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1
        else:
            logger.warning("CUDA requested but not available; falling back to CPU.")

    max_steps = -1 if args.max_steps is None else args.max_steps
    enable_progress_bar = False
    callbacks = []
    if not args.disable_progress_bar:
        callbacks.append(StepProgressLogger(single_line=True, update_every_n_steps=50))
    best_ckpt_path = None
    if phase == "morph":
        best_ckpt_path = checkpoints_dir / "phase1_morph_best.ckpt"
        callbacks.append(
            BestMetricCheckpoint(
                ckpt_path=best_ckpt_path,
                primary_metric="val/loss",
                fallback_metric="val/exp_acc",
                primary_mode="min",
                fallback_mode="max",
            )
        )
    if phase == "quality":
        best_ckpt_path = checkpoints_dir / "phase4_quality.ckpt"
        callbacks.append(
            BestMetricCheckpoint(
                ckpt_path=best_ckpt_path,
                primary_metric="val/quality_auprc",
                fallback_metric="val/loss",
                primary_mode="max",
                fallback_mode="min",
            )
        )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        enable_checkpointing=False,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        logger=loggers if loggers else False,
        deterministic=True,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(logs_dir),
        callbacks=callbacks,
    )

    trainer.fit(lightning_module, datamodule=datamodule)

    ckpt_name = {
        "morph": "phase1_morph.ckpt",
        "stage": "phase2_stage.ckpt",
        "joint": "phase3_joint.ckpt",
        "quality": "phase4_quality.ckpt",
    }[phase]
    ckpt_path = checkpoints_dir / ckpt_name
    if phase == "quality":
        if best_ckpt_path and best_ckpt_path.exists():
            ckpt_path = best_ckpt_path
        else:
            trainer.save_checkpoint(ckpt_path)
            logger.warning("Best checkpoint not found; saved LAST checkpoint to %s", ckpt_path)

        reports_dir = ensure_outputs_dir(cfg.outputs.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        eval_device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
        eval_module = MultiTaskLightningModule.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            model=model,
            phase=phase,
            lr=phase_cfg.lr,
            weight_decay=phase_cfg.weight_decay,
            loss_weights=loss_weights,
            freeze_config=freeze_cfg,
            morph_loss_reduction=phase_cfg.morph_loss_reduction,
            quality_pos_weight=quality_pos_weight,
            use_class_weights=morph_use_class_weights,
            class_weight_mode=morph_class_weight_mode,
        )
        _tune_quality_threshold(eval_module.model, datamodule.val_dataloader(), eval_device, reports_dir, logger)
        _run_quality_test_eval(eval_module.model, datamodule.test_dataloader(), reports_dir, eval_device, logger)

    if phase != "quality":
        if phase == "morph" and best_ckpt_path and best_ckpt_path.exists():
            if best_ckpt_path != ckpt_path:
                shutil.copy2(best_ckpt_path, ckpt_path)
            logger.info("Saved BEST checkpoint: %s", ckpt_path)
        else:
            trainer.save_checkpoint(ckpt_path)
            logger.info("Saved LAST checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
