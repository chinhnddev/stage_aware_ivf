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
from ivf.data.label_schema import QUALITY_TO_ID, QualityLabel
from ivf.data.transforms import assert_no_augmentation, get_eval_transforms
from ivf.eval_label_sources import resolve_quality_label, update_label_source_counts
from ivf.utils.guardrails import assert_no_day_feature, assert_no_segmentation_inputs


def _normalize_day(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            if isinstance(value, float) and value != value:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None
    text = str(value).lower()
    if text in {"nan", "none", ""}:
        return None
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def build_hungvuong_quality_dataset(
    config: Dict,
    image_size: int = 256,
    normalize: bool = False,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> BaseImageDataset:
    df = pd.read_csv(config["csv_path"])
    return build_quality_dataset_from_df(
        df,
        config=config,
        image_size=image_size,
        normalize=normalize,
        mean=mean,
        std=std,
    )


def build_quality_dataset_from_df(
    df: pd.DataFrame,
    config: Dict,
    image_size: int = 256,
    normalize: bool = False,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> BaseImageDataset:
    quality_col = config.get("quality_col")
    day_col = config.get("day_col")
    image_col = config["image_col"]
    id_col = config["id_col"]
    root_dir = Path(config["root_dir"])
    allow_grade = bool(config.get("allow_grade_labels", False))
    grade_col = config.get("grade_col") or config.get("label_col")

    records = []
    source_counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        label_value, source = resolve_quality_label(
            row,
            image_col=image_col,
            quality_col=quality_col,
            grade_col=grade_col,
            allow_grade=allow_grade,
        )
        update_label_source_counts(source_counts, source)
        if label_value is None:
            continue

        image_id = row.get(id_col)
        if pd.isna(image_id):
            image_id = row.get(image_col)

        targets = make_full_target_dict(quality=label_value)
        meta = {
            "id": image_id,
            "dataset": "hungvuong",
            "quality": QualityLabel.GOOD.value if label_value == 1 else QualityLabel.POOR.value,
            "label_source": source,
        }
        if day_col and day_col in df.columns:
            meta["day"] = row.get(day_col)

        records.append(
            {
                "image_path": row.get(image_col),
                "targets": targets,
                "meta": meta,
            }
        )

    eval_tf = get_eval_transforms(
        image_size=image_size,
        normalize=normalize,
        mean=mean,
        std=std,
    )
    assert_no_augmentation(eval_tf)
    if source_counts:
        source_summary = ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items()))
        print(f"External label sources: {source_summary}")
    return BaseImageDataset(records, transform=eval_tf, include_meta_day=True, root_dir=str(root_dir))


def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    return_morph: bool = False,
    return_stage: bool = False,
) -> Dict[str, List]:
    model.eval()
    prob_good: List[float] = []
    y_true: List[int] = []
    days: List[Optional[int]] = []
    image_ids: List[str] = []
    exp_preds: List[int] = []
    icm_preds: List[int] = []
    te_preds: List[int] = []
    stage_preds: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            assert_no_day_feature(batch)
            assert_no_segmentation_inputs(batch)
            images = batch["image"].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs["quality"], dim=-1)[:, 1].detach().cpu()
            if return_morph:
                exp_pred = torch.argmax(outputs["morph"]["exp"], dim=-1).detach().cpu()
                icm_pred = torch.argmax(outputs["morph"]["icm"], dim=-1).detach().cpu()
                te_pred = torch.argmax(outputs["morph"]["te"], dim=-1).detach().cpu()
            if return_stage:
                stage_pred = torch.argmax(outputs["stage"], dim=-1).detach().cpu()
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
                if return_morph:
                    exp_preds.append(int(exp_pred[i]))
                    icm_preds.append(int(icm_pred[i]))
                    te_preds.append(int(te_pred[i]))
                if return_stage:
                    stage_preds.append(int(stage_pred[i]))

    result = {
        "prob_good": prob_good,
        "y_true": y_true,
        "day": days,
        "image_id": image_ids,
    }
    if return_morph:
        result["morph_pred"] = {"exp": exp_preds, "icm": icm_preds, "te": te_preds}
    if return_stage:
        result["stage_pred"] = stage_preds
    return result


def compute_metrics(prob_good: Iterable[float], y_true: Iterable[int]) -> Dict[str, Optional[float]]:
    prob_list = list(prob_good)
    y_list = list(y_true)
    prob_tensor = torch.tensor(prob_list, dtype=torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.int64)
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
