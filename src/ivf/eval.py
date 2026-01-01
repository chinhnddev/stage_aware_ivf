"""
Evaluation utilities for external testing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score

from ivf.data.datasets import BaseImageDataset, make_full_target_dict
from ivf.data.label_schema import QUALITY_TO_ID, QualityLabel, map_gardner_to_quality
from ivf.data.transforms import assert_no_augmentation, get_eval_transforms
from ivf.utils.guardrails import assert_no_day_feature, assert_no_segmentation_inputs


def _normalize_quality(value) -> Optional[QualityLabel]:
    if value is None:
        return None
    text = str(value).strip().lower()
    for label in QualityLabel:
        if label.value == text:
            return label
    return None


def _normalize_day(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        day = int(value)
        return day
    text = str(value).lower()
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def build_hungvuong_quality_dataset(config: Dict) -> BaseImageDataset:
    df = pd.read_csv(config["csv_path"])
    grade_col = config.get("grade_col") or config.get("label_col")
    quality_col = config.get("quality_col")
    day_col = config.get("day_col")
    image_col = config["image_col"]
    id_col = config["id_col"]
    root_dir = Path(config["root_dir"])

    records = []
    for _, row in df.iterrows():
        quality_label = None
        if quality_col and quality_col in df.columns:
            quality_label = _normalize_quality(row.get(quality_col))
        if quality_label is None and grade_col and grade_col in df.columns:
            quality_label = map_gardner_to_quality(row.get(grade_col))

        if quality_label is None:
            continue

        image_id = row.get(id_col)
        if pd.isna(image_id):
            image_id = row.get(image_col)

        targets = make_full_target_dict(quality=QUALITY_TO_ID[quality_label])
        meta = {
            "id": image_id,
            "dataset": "hungvuong",
            "quality": quality_label.value,
        }
        if grade_col and grade_col in df.columns:
            meta["grade"] = row.get(grade_col)
        if day_col and day_col in df.columns:
            meta["day"] = row.get(day_col)

        records.append(
            {
                "image_path": str(root_dir / str(row.get(image_col))),
                "targets": targets,
                "meta": meta,
            }
        )

    eval_tf = get_eval_transforms()
    assert_no_augmentation(eval_tf)
    return BaseImageDataset(records, transform=eval_tf, include_meta_day=True)


def predict(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, List]:
    model.eval()
    prob_good: List[float] = []
    y_true: List[int] = []
    days: List[Optional[int]] = []
    image_ids: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            assert_no_day_feature(batch)
            assert_no_segmentation_inputs(batch)
            images = batch["image"].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs["quality"], dim=-1)[:, 1].detach().cpu()
            targets = batch["targets"]["quality"]
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu()
            else:
                targets = torch.tensor(targets)

            meta = batch.get("meta", {})
            if isinstance(meta, list):
                ids = [m.get("id") for m in meta]
                day_vals = [m.get("day") for m in meta]
            elif isinstance(meta, dict):
                ids = meta.get("id")
                day_vals = meta.get("day")
            else:
                ids = None
                day_vals = None

            if ids is None:
                ids = [None] * len(probs)
            if day_vals is None:
                day_vals = [None] * len(probs)

            mask = targets >= 0
            for i, keep in enumerate(mask.tolist()):
                if not keep:
                    continue
                prob_good.append(float(probs[i]))
                y_true.append(int(targets[i]))
                image_ids.append(str(ids[i]))
                days.append(_normalize_day(day_vals[i]))

    return {
        "prob_good": prob_good,
        "y_true": y_true,
        "day": days,
        "image_id": image_ids,
    }


def compute_metrics(prob_good: Iterable[float], y_true: Iterable[int]) -> Dict[str, Optional[float]]:
    prob_tensor = torch.tensor(list(prob_good), dtype=torch.float32)
    y_tensor = torch.tensor(list(y_true), dtype=torch.int64)
    if prob_tensor.numel() == 0:
        return {"auroc": None, "auprc": None, "f1": None}

    metrics = {
        "auroc": BinaryAUROC(),
        "auprc": BinaryAveragePrecision(),
        "f1": BinaryF1Score(),
    }
    results = {}
    for name, metric in metrics.items():
        try:
            results[name] = float(metric(prob_tensor, y_tensor))
        except ValueError:
            results[name] = None
    return results


def slice_by_day(preds: Dict[str, List], day_value: int) -> Tuple[List[float], List[int]]:
    prob_good = []
    y_true = []
    for prob, label, day in zip(preds["prob_good"], preds["y_true"], preds["day"]):
        if day == day_value:
            prob_good.append(prob)
            y_true.append(label)
    return prob_good, y_true
