"""
Utilities for ABL-BIN in-domain binary classifier on Hung Vuong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from ivf.data.datasets import BaseImageDataset, make_full_target_dict
from ivf.data.splits import save_splits, split_by_group
from ivf.eval import _normalize_day
from ivf.eval_label_sources import resolve_quality_label


def load_split_csvs(splits_dir: str) -> Dict[str, pd.DataFrame]:
    base = Path(splits_dir)
    train_path = base / "train.csv"
    val_path = base / "val.csv"
    test_path = base / "test.csv"
    missing = [str(path) for path in (train_path, val_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing HV split files: {missing}")
    return {
        "train": pd.read_csv(train_path),
        "val": pd.read_csv(val_path),
        "test": pd.read_csv(test_path),
    }


def load_hv_metadata(dataset_cfg: Dict, hv_root: str) -> pd.DataFrame:
    csv_path = Path(str(dataset_cfg.get("csv_path", "")))
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing HV metadata CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        image_col = dataset_cfg.get("image_col", "image_path")
        if image_col not in df.columns:
            raise ValueError("HV metadata missing image_path column.")
        df = df.copy()
        df["image_path"] = df[image_col].astype(str)
    df = df.copy()
    df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/")
    df["hv_root"] = str(hv_root).replace("\\", "/")
    return df


def split_hv_by_folder(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _bucket(path_value: str) -> Optional[str]:
        parts = str(path_value).replace("\\", "/").split("/")
        return parts[0].lower() if parts else None

    buckets = df["image_path"].apply(_bucket)
    train_df = df[buckets == "train"].copy()
    test_df = df[buckets == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("HV metadata must include both train/ and test/ image_path prefixes.")
    return train_df, test_df


def resolve_group_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def load_or_create_trainval_split(
    train_df: pd.DataFrame,
    group_col: Optional[str],
    val_ratio: float,
    seed: int,
    split_path: Path,
    logger,
    id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split_path.exists():
        mapping = pd.read_csv(split_path)
        if "image_path" not in mapping.columns or "split" not in mapping.columns:
            raise ValueError(f"Invalid split file format: {split_path}")
        mapping["image_path"] = mapping["image_path"].astype(str)
        train_base = train_df.drop(columns=["split"], errors="ignore")
        merged = train_base.merge(mapping[["image_path", "split"]], on="image_path", how="left")
        if merged["split"].isna().any():
            missing = int(merged["split"].isna().sum())
            raise ValueError(f"Split file missing {missing} train samples; regenerate splits.")
        train_split = merged[merged["split"] == "train"].drop(columns=["split"])
        val_split = merged[merged["split"] == "val"].drop(columns=["split"])
        logger.info("Loaded existing train/val split from %s", split_path)
        return train_split, val_split

    splits = split_by_group(
        train_df,
        group_col=group_col,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
    )
    train_split = splits["train"].copy()
    val_split = splits["val"].copy()
    output = pd.concat(
        [
            train_split.assign(split="train"),
            val_split.assign(split="val"),
        ],
        ignore_index=True,
    )
    cols = ["image_path", "split"]
    if id_col and id_col in output.columns:
        cols.insert(0, id_col)
    if group_col and group_col in output.columns and group_col not in cols:
        cols.append(group_col)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    output[cols].to_csv(split_path, index=False)
    logger.info("Saved train/val split to %s", split_path)
    return train_split, val_split


def _resolve_group_col(dfs: Dict[str, pd.DataFrame], candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if all(col in df.columns for df in dfs.values()):
            has_values = any(df[col].notna().any() for df in dfs.values())
            if has_values:
                return col
    return None


def assert_no_group_overlap(dfs: Dict[str, pd.DataFrame], group_col: Optional[str], logger) -> None:
    if not group_col:
        logger.warning("Group column not found; overlap check skipped.")
        return
    groups = {}
    for name, df in dfs.items():
        groups[name] = set(df[group_col].dropna().astype(str))
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[left].intersection(groups[right])
        if overlap:
            raise ValueError(
                f"Group leakage detected between {left}/{right} for {group_col}: {sorted(list(overlap))[:3]}"
            )
    logger.info("Group overlap check passed for %s.", group_col)


def _label_from_row(row, image_col: str, dataset_cfg: Dict) -> Tuple[Optional[int], Optional[str]]:
    quality_col = dataset_cfg.get("quality_col")
    grade_col = dataset_cfg.get("grade_col") or dataset_cfg.get("label_col")
    allow_grade = bool(dataset_cfg.get("allow_grade_labels", False))
    return resolve_quality_label(
        row,
        image_col=image_col,
        quality_col=quality_col,
        grade_col=grade_col,
        allow_grade=allow_grade,
    )


def build_records(
    df: pd.DataFrame,
    dataset_cfg: Dict,
    split_name: str,
    logger,
    group_col: Optional[str] = None,
) -> List[Dict]:
    image_col = dataset_cfg.get("image_col", "image_path")
    id_col = dataset_cfg.get("id_col")
    day_col = dataset_cfg.get("day_col")
    stage_col = dataset_cfg.get("label_col")
    dataset_type = str(dataset_cfg.get("dataset_type", "hungvuong"))

    if image_col not in df.columns:
        raise ValueError(f"Split '{split_name}' missing image_col={image_col}")

    stats = {"total": len(df), "kept": 0, "dropped": 0, "good": 0, "not_good": 0}
    records: List[Dict] = []
    for _, row in df.iterrows():
        label_value, source = _label_from_row(row, image_col, dataset_cfg)
        if label_value is None:
            stats["dropped"] += 1
            continue
        if label_value not in {0, 1}:
            raise ValueError(f"Invalid label in {split_name}: {label_value}")
        stats["kept"] += 1
        if label_value == 1:
            stats["good"] += 1
        else:
            stats["not_good"] += 1

        image_id = row.get(id_col) if id_col and id_col in df.columns else row.get(image_col)
        targets = make_full_target_dict(quality=int(label_value))
        meta = {
            "id": image_id,
            "dataset": dataset_type,
            "quality": "good" if label_value == 1 else "not_good",
            "label_source": source,
            "image_path": row.get(image_col),
        }
        if group_col and group_col in df.columns:
            meta["group_id"] = row.get(group_col)
        if day_col and day_col in df.columns:
            meta["day"] = row.get(day_col)
        if stage_col and stage_col in df.columns:
            meta["stage"] = row.get(stage_col)

        records.append({"image_path": row.get(image_col), "targets": targets, "meta": meta})

    logger.info(
        "HV %s labels: total=%s kept=%s dropped=%s good=%s not_good=%s",
        split_name,
        stats["total"],
        stats["kept"],
        stats["dropped"],
        stats["good"],
        stats["not_good"],
    )
    if stats["kept"] == 0:
        raise ValueError(f"No labeled samples in {split_name} split after filtering.")
    if stats["dropped"] > 0:
        logger.warning("HV %s dropped %s unlabeled samples.", split_name, stats["dropped"])
    return records


def log_split_sizes(dfs: Dict[str, pd.DataFrame], logger) -> None:
    logger.info(
        "HV split sizes: train=%s val=%s test=%s",
        len(dfs["train"]),
        len(dfs["val"]),
        len(dfs["test"]),
    )


def compute_pos_weight(labels: List[int], logger, max_weight: float = 20.0) -> Optional[float]:
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = sum(1 for y in labels if y == 0)
    if n_pos == 0 or n_neg == 0:
        logger.warning("Train labels missing positives or negatives; pos_weight disabled.")
        return None
    pos_weight = float(n_neg / n_pos)
    if max_weight and pos_weight > max_weight:
        logger.warning("pos_weight clipped from %.4f to %.4f", pos_weight, max_weight)
        pos_weight = float(max_weight)
    logger.info("Using pos_weight=%.4f (neg=%s pos=%s).", pos_weight, n_neg, n_pos)
    return pos_weight


def _binary_counts(labels: List[int]) -> Tuple[int, int]:
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = sum(1 for y in labels if y == 0)
    return n_pos, n_neg


def _safe_prob_metrics(probs: List[float], labels: List[int]) -> Dict[str, Optional[float]]:
    n_pos, n_neg = _binary_counts(labels)
    if len(labels) == 0:
        return {"auroc": None, "auprc": None, "skipped_reason": "no_samples"}
    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "auprc": None, "skipped_reason": "single_class"}
    prob_tensor = torch.tensor(probs, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.int64)
    auroc = float(BinaryAUROC()(prob_tensor, label_tensor))
    auprc = float(BinaryAveragePrecision()(prob_tensor, label_tensor))
    return {"auroc": auroc, "auprc": auprc}


def _binary_metrics_from_threshold(probs: List[float], labels: List[int], threshold: float) -> Dict[str, Optional[float]]:
    if not labels:
        return {"f1": None, "acc": None, "precision": None, "recall": None}
    y = np.array(labels, dtype=int)
    p = (np.array(probs, dtype=float) >= threshold).astype(int)
    acc = float((p == y).mean()) if len(y) else None
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    denom = (2 * tp + fp + fn)
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    return {"f1": float(f1), "acc": acc, "precision": float(precision), "recall": float(recall)}


def tune_threshold(probs: List[float], labels: List[int]) -> Dict[str, Optional[float]]:
    if not labels:
        return {"threshold": 0.5, "f1": None, "reason": "no_labels"}
    best = {"threshold": 0.5, "f1": -1.0}
    for step in range(0, 101):
        thresh = step / 100.0
        metrics = _binary_metrics_from_threshold(probs, labels, thresh)
        f1 = metrics.get("f1")
        if f1 is None:
            continue
        if f1 > best["f1"]:
            best = {"threshold": float(thresh), "f1": float(f1)}
    reason = None
    if best["f1"] < 0:
        best["f1"] = None
        reason = "invalid_f1"
    payload = dict(best)
    if reason:
        payload["reason"] = reason
    return payload


def tune_threshold_with_mode(
    probs: List[float],
    labels: List[int],
    days: Optional[List[Optional[int]]],
    mode: str,
    logger,
) -> Dict[str, Optional[float]]:
    if mode not in {"overall", "day5"}:
        raise ValueError(f"Unsupported tune mode: {mode}")
    if mode == "day5" and days is not None:
        day5_probs = [p for p, d in zip(probs, days) if d == 5]
        day5_labels = [y for y, d in zip(labels, days) if d == 5]
        if day5_labels:
            payload = tune_threshold(day5_probs, day5_labels)
            payload["mode"] = "day5"
            payload["subset_n"] = len(day5_labels)
            return payload
        logger.warning("No day5 labels available for threshold tuning; falling back to overall.")
    payload = tune_threshold(probs, labels)
    payload["mode"] = "overall"
    payload["subset_n"] = len(labels)
    return payload


def metrics_block(
    probs: List[float],
    labels: List[int],
    days: List[Optional[int]],
    threshold: float,
    logger,
) -> Dict[str, Dict[str, Optional[float]]]:
    metrics = {}
    for key, day_filter in (
        ("overall", None),
        ("day3", 3),
        ("day5", 5),
    ):
        if day_filter is None:
            subset_probs = probs
            subset_labels = labels
        else:
            subset_probs = [p for p, d in zip(probs, days) if d == day_filter]
            subset_labels = [y for y, d in zip(labels, days) if d == day_filter]
        n_pos, n_neg = _binary_counts(subset_labels)
        stats = {"n": len(subset_labels), "n_pos": n_pos, "n_neg": n_neg}
        prob_metrics = _safe_prob_metrics(subset_probs, subset_labels)
        if prob_metrics.get("skipped_reason"):
            logger.warning("Metrics %s skipped: %s", key, prob_metrics["skipped_reason"])
        bin_metrics = _binary_metrics_from_threshold(subset_probs, subset_labels, threshold)
        metrics[key] = {**stats, **prob_metrics, **bin_metrics}
    return metrics


def collect_predictions(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    split_name: str,
) -> Dict[str, list]:
    model.eval()
    probs = []
    labels = []
    days = []
    stages = []
    image_paths = []
    group_ids = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            logits = model(images)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            targets = batch["targets"]["quality"]
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu().numpy()
            meta = batch.get("meta", [])
            if not isinstance(meta, list):
                meta = [meta] * len(prob)

            for idx, target in enumerate(targets):
                label = int(target)
                if label not in {0, 1}:
                    raise ValueError(f"Invalid label value: {label}")
                prob_val = float(prob[idx])
                if np.isnan(prob_val):
                    raise ValueError("NaN prediction encountered.")
                probs.append(prob_val)
                labels.append(label)
                day_val = meta[idx].get("day") if idx < len(meta) else None
                days.append(_normalize_day(day_val))
                stages.append(meta[idx].get("stage") if idx < len(meta) else None)
                image_paths.append(meta[idx].get("image_path") if idx < len(meta) else None)
                group_ids.append(meta[idx].get("group_id") if idx < len(meta) else None)

    return {
        "split": split_name,
        "prob_good": probs,
        "y_true": labels,
        "day": days,
        "stage": stages,
        "image_path": image_paths,
        "group_id": group_ids,
    }


def write_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def label_distribution(df: pd.DataFrame, dataset_cfg: Dict) -> Dict[str, Dict[str, int]]:
    image_col = dataset_cfg.get("image_col", "image_path")
    quality_col = dataset_cfg.get("quality_col")
    grade_col = dataset_cfg.get("grade_col") or dataset_cfg.get("label_col")
    allow_grade = bool(dataset_cfg.get("allow_grade_labels", False))
    day_col = dataset_cfg.get("day_col")
    buckets = {"overall": [], "day3": [], "day5": []}
    for _, row in df.iterrows():
        label, _ = resolve_quality_label(
            row,
            image_col=image_col,
            quality_col=quality_col,
            grade_col=grade_col,
            allow_grade=allow_grade,
        )
        if label is None:
            continue
        buckets["overall"].append((label, row.get(day_col) if day_col in df.columns else None))
    result = {}
    for key in ("overall", "day3", "day5"):
        if key == "overall":
            labels = [label for label, _ in buckets["overall"]]
        elif key == "day3":
            labels = [label for label, day in buckets["overall"] if _normalize_day(day) == 3]
        else:
            labels = [label for label, day in buckets["overall"] if _normalize_day(day) == 5]
        n_pos, n_neg = _binary_counts(labels)
        result[key] = {"n": len(labels), "n_pos": n_pos, "n_neg": n_neg}
    return result


def log_label_distribution(df: pd.DataFrame, dataset_cfg: Dict, split_name: str, logger) -> None:
    dist = label_distribution(df, dataset_cfg)
    logger.info(
        "HV %s distribution overall n=%s pos=%s neg=%s",
        split_name,
        dist["overall"]["n"],
        dist["overall"]["n_pos"],
        dist["overall"]["n_neg"],
    )
    logger.info(
        "HV %s distribution day3 n=%s pos=%s neg=%s",
        split_name,
        dist["day3"]["n"],
        dist["day3"]["n_pos"],
        dist["day3"]["n_neg"],
    )
    logger.info(
        "HV %s distribution day5 n=%s pos=%s neg=%s",
        split_name,
        dist["day5"]["n"],
        dist["day5"]["n_pos"],
        dist["day5"]["n_neg"],
    )
