"""
Cross-domain evaluation for joint morphology models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from omegaconf import OmegaConf

from ivf.config import load_experiment_config
from ivf.data.datasets import BaseImageDataset, collate_batch, make_full_target_dict
from ivf.data.label_schema import (
    ICM_CLASSES,
    TE_CLASSES,
    ICM_TO_ID,
    TE_TO_ID,
    normalize_gardner_exp,
    normalize_gardner_grade,
    parse_gardner_components,
)
from ivf.data.transforms import assert_no_augmentation, get_eval_transforms
from ivf.models.factory import build_model_from_config
from ivf.morphology import morph_score
from ivf.utils.paths import ensure_outputs_dir


def _pick_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _is_zero_based(col_name: Optional[str]) -> bool:
    if not col_name:
        return False
    lower = col_name.lower()
    return lower.endswith("_silver") or lower.endswith("_gold")


def _coerce_int(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"ND", "NA", "N/A"}:
        return None
    try:
        return int(float(text))
    except (ValueError, TypeError):
        return None


def _normalize_zero_based_exp(value, exp_max: int) -> Optional[int]:
    num = _coerce_int(value)
    if num is None:
        return None
    if num < 0 or num > exp_max - 1:
        return None
    return num + 1


def _normalize_zero_based_grade(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"A", "B", "C"}:
        return text
    num = _coerce_int(value)
    if num is None:
        return None
    if num == 0:
        return "A"
    if num == 1:
        return "B"
    if num == 2:
        return "C"
    return None


def _resolve_columns(df: pd.DataFrame, args, data_cfg: Optional[dict]):
    image_col = args.image_col or _pick_col(df, ["image_path", "image", "Image", "filename", "file"])
    id_col = args.id_col or _pick_col(df, ["id", "image_id", "embryo_id", "embryo", "name"])
    grade_col = args.grade_col or _pick_col(df, ["grade", "gardner", "gardner_grade", "label"])
    exp_col = args.exp_col or _pick_col(df, ["exp", "EXP", "EXP_gold", "EXP_silver", "expansion"])
    icm_col = args.icm_col or _pick_col(df, ["icm", "ICM", "ICM_gold", "ICM_silver"])
    te_col = args.te_col or _pick_col(df, ["te", "TE", "TE_gold", "TE_silver"])
    day_col = args.day_col or _pick_col(df, ["day", "day_post_insemination"])
    if data_cfg:
        image_col = image_col or data_cfg.get("image_col")
        id_col = id_col or data_cfg.get("id_col")
        grade_col = grade_col or data_cfg.get("label_col") or data_cfg.get("grade_col")
        day_col = day_col or data_cfg.get("day_col")
    if image_col is None:
        raise ValueError("Missing image column; pass --image_col to specify it.")
    return image_col, id_col, grade_col, exp_col, icm_col, te_col, day_col


def _build_records(
    df: pd.DataFrame,
    exp_max: int,
    image_col: str,
    id_col: Optional[str],
    grade_col: Optional[str],
    exp_col: Optional[str],
    icm_col: Optional[str],
    te_col: Optional[str],
    day_col: Optional[str],
):
    records = []
    exp_zero_based = _is_zero_based(exp_col)
    icm_zero_based = _is_zero_based(icm_col)
    te_zero_based = _is_zero_based(te_col)

    for _, row in df.iterrows():
        image_path = row.get(image_col)
        if pd.isna(image_path):
            continue
        image_id = row.get(id_col) if id_col else None
        if pd.isna(image_id):
            image_id = row.get(image_col)
        grade = row.get(grade_col) if grade_col else None
        components = parse_gardner_components(grade, exp_max=exp_max) if grade is not None else None
        exp_raw = row.get(exp_col) if exp_col else None
        icm_raw = row.get(icm_col) if icm_col else None
        te_raw = row.get(te_col) if te_col else None

        if exp_zero_based:
            exp = _normalize_zero_based_exp(exp_raw, exp_max=exp_max)
        else:
            exp = normalize_gardner_exp(exp_raw, exp_max=exp_max)
        if exp is None and components is not None:
            exp = components[0]
        if icm_zero_based:
            icm = _normalize_zero_based_grade(icm_raw)
        else:
            icm = normalize_gardner_grade(icm_raw)
        if te_zero_based:
            te = _normalize_zero_based_grade(te_raw)
        else:
            te = normalize_gardner_grade(te_raw)
        if icm is None and components is not None:
            icm = components[1]
        if te is None and components is not None:
            te = components[2]

        exp_id = exp - 1 if exp is not None else None
        exp_mask = 1 if exp_id is not None else 0
        icm_id = ICM_TO_ID.get(icm) if icm is not None else None
        te_id = TE_TO_ID.get(te) if te is not None else None
        icm_mask = 1 if icm_id is not None else 0
        te_mask = 1 if te_id is not None else 0

        targets = make_full_target_dict(
            exp=exp_id,
            icm=icm_id,
            te=te_id,
            exp_mask=exp_mask,
            icm_mask=icm_mask,
            te_mask=te_mask,
        )
        meta = {
            "id": image_id,
            "grade": grade,
            "exp": exp,
            "icm": icm,
            "te": te,
        }
        if day_col and day_col in df.columns:
            meta["day"] = row.get(day_col)
        records.append({"image_path": image_path, "targets": targets, "meta": meta})
    return records


def _task_metrics(y_true, y_pred, labels):
    if not y_true:
        return {"accuracy": None, "precision_weighted": None, "recall_weighted": None, "f1_weighted": None, "cohen_kappa": None}
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    try:
        kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    except ValueError:
        kappa = None
    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "cohen_kappa": None if kappa is None else float(kappa),
    }


def _save_confusion_matrix(y_true, y_pred, labels, label_names, out_path: Path):
    if not y_true:
        cm = np.zeros((len(labels), len(labels)), dtype=int)
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=label_names, columns=label_names)
    df.to_csv(out_path, index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-domain evaluation for morphology models.")
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument("--checkpoint", required=True, help="Morphology checkpoint path.")
    parser.add_argument("--data_csv", required=True, help="CSV file containing morphology data.")
    parser.add_argument("--output_dir", default=None, help="Output directory under outputs/.")
    parser.add_argument("--root_dir", default=None, help="Optional root dir for relative image paths.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]")
    parser.add_argument("--image_col", default=None)
    parser.add_argument("--id_col", default=None)
    parser.add_argument("--grade_col", default=None)
    parser.add_argument("--exp_col", default=None)
    parser.add_argument("--icm_col", default=None)
    parser.add_argument("--te_col", default=None)
    parser.add_argument("--day_col", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)

    morph_cfg = getattr(cfg.training, "morph", None)
    exp_max = int(getattr(morph_cfg, "exp_max", 5)) if morph_cfg is not None else 5

    data_cfg = None
    if cfg.data and cfg.data.blastocyst_config:
        data_cfg = OmegaConf.load(cfg.data.blastocyst_config)

    df = pd.read_csv(args.data_csv)
    image_col, id_col, grade_col, exp_col, icm_col, te_col, day_col = _resolve_columns(df, args, data_cfg)
    records = _build_records(
        df,
        exp_max=exp_max,
        image_col=image_col,
        id_col=id_col,
        grade_col=grade_col,
        exp_col=exp_col,
        icm_col=icm_col,
        te_col=te_col,
        day_col=day_col,
    )

    eval_tf = get_eval_transforms(
        image_size=cfg.transforms.image_size,
        normalize=cfg.transforms.normalize,
        mean=list(cfg.transforms.mean) if cfg.transforms.mean is not None else None,
        std=list(cfg.transforms.std) if cfg.transforms.std is not None else None,
        crop_size=getattr(cfg.transforms, "crop_size", None),
    )
    assert_no_augmentation(eval_tf)

    root_dir = args.root_dir or (data_cfg.get("root_dir") if data_cfg else None)
    dataset = BaseImageDataset(records, transform=eval_tf, include_meta_day=True, root_dir=root_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    device = args.device or cfg.device
    device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    model = build_model_from_config(cfg, phase="morph")
    module_state = torch.load(args.checkpoint, map_location="cpu")
    state_dict = module_state.get("state_dict", module_state)
    model_state = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()

    exp_true, exp_pred = [], []
    icm_true, icm_pred = [], []
    te_true, te_pred = [], []
    preds_rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            exp_logits = outputs["morph"]["exp"]
            icm_logits = outputs["morph"]["icm"]
            te_logits = outputs["morph"]["te"]
            exp_probs = torch.softmax(exp_logits, dim=-1).cpu().numpy()
            icm_probs = torch.softmax(icm_logits, dim=-1).cpu().numpy()
            te_probs = torch.softmax(te_logits, dim=-1).cpu().numpy()
            exp_argmax = exp_probs.argmax(axis=1)
            icm_argmax = icm_probs.argmax(axis=1)
            te_argmax = te_probs.argmax(axis=1)

            targets = batch["targets"]
            meta = batch.get("meta", [])
            for i in range(images.shape[0]):
                t_exp = int(targets["exp"][i]) if targets["exp"][i] >= 0 else None
                t_icm = int(targets["icm"][i]) if targets["icm"][i] >= 0 else None
                t_te = int(targets["te"][i]) if targets["te"][i] >= 0 else None
                exp_mask = int(targets["exp_mask"][i]) == 1
                icm_mask = int(targets["icm_mask"][i]) == 1
                te_mask = int(targets["te_mask"][i]) == 1

                pred_exp = int(exp_argmax[i]) + 1
                pred_icm = ICM_CLASSES[int(icm_argmax[i])]
                pred_te = TE_CLASSES[int(te_argmax[i])]

                pred_exp_idx = int(exp_argmax[i])
                if exp_mask and t_exp is not None and t_exp < exp_max and pred_exp_idx < exp_max:
                    exp_true.append(t_exp)
                    exp_pred.append(pred_exp_idx)
                if icm_mask and t_icm is not None:
                    icm_true.append(t_icm)
                    icm_pred.append(int(icm_argmax[i]))
                if te_mask and t_te is not None:
                    te_true.append(t_te)
                    te_pred.append(int(te_argmax[i]))

                meta_item = meta[i] if isinstance(meta, list) else {}
                image_id = meta_item.get("id", str(i))
                valid_icm = icm_mask
                valid_te = te_mask
                q_score = morph_score(
                    exp_probs[i],
                    icm_probs[i],
                    te_probs[i],
                    valid_icm=valid_icm,
                    valid_te=valid_te,
                    config=cfg.morph_score,
                )

                row = {
                    "image_id": image_id,
                    "exp_pred": pred_exp,
                    "icm_pred": pred_icm,
                    "te_pred": pred_te,
                    "q_score": q_score,
                    "valid_icm": int(bool(valid_icm)),
                    "valid_te": int(bool(valid_te)),
                }
                for idx in range(exp_probs.shape[1]):
                    row[f"exp_prob_{idx + 1}"] = float(exp_probs[i][idx])
                for idx, label in enumerate(ICM_CLASSES[: icm_probs.shape[1]]):
                    row[f"icm_prob_{label}"] = float(icm_probs[i][idx])
                for idx, label in enumerate(TE_CLASSES[: te_probs.shape[1]]):
                    row[f"te_prob_{label}"] = float(te_probs[i][idx])
                preds_rows.append(row)

    if args.output_dir:
        out_dir = ensure_outputs_dir(args.output_dir)
    else:
        out_dir = ensure_outputs_dir(cfg.outputs.reports_dir) / "morph_crossdomain"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "exp": _task_metrics(exp_true, exp_pred, labels=list(range(exp_max))),
        "icm": _task_metrics(icm_true, icm_pred, labels=list(range(len(ICM_CLASSES)))),
        "te": _task_metrics(te_true, te_pred, labels=list(range(len(TE_CLASSES)))),
    }
    for head in ("exp", "icm", "te"):
        with (out_dir / f"{head}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics[head], f, indent=2)

    _save_confusion_matrix(
        exp_true,
        exp_pred,
        labels=list(range(exp_max)),
        label_names=[str(i) for i in range(1, exp_max + 1)],
        out_path=out_dir / "exp_confusion.csv",
    )
    _save_confusion_matrix(
        icm_true,
        icm_pred,
        labels=list(range(len(ICM_CLASSES))),
        label_names=ICM_CLASSES,
        out_path=out_dir / "icm_confusion.csv",
    )
    _save_confusion_matrix(
        te_true,
        te_pred,
        labels=list(range(len(TE_CLASSES))),
        label_names=TE_CLASSES,
        out_path=out_dir / "te_confusion.csv",
    )

    pred_df = pd.DataFrame(preds_rows)
    pred_df.to_csv(out_dir / "morph_predictions.csv", index=False)

    q_scores = pred_df["q_score"].astype(float).to_numpy() if not pred_df.empty else np.array([])
    q_stats = {}
    if q_scores.size:
        q_stats = {
            "mean": float(np.mean(q_scores)),
            "std": float(np.std(q_scores)),
            "p05": float(np.percentile(q_scores, 5)),
            "p25": float(np.percentile(q_scores, 25)),
            "p50": float(np.percentile(q_scores, 50)),
            "p75": float(np.percentile(q_scores, 75)),
            "p95": float(np.percentile(q_scores, 95)),
        }
    with (out_dir / "morph_score_stats.json").open("w", encoding="utf-8") as f:
        json.dump(q_stats, f, indent=2)

    print(f"Saved morphology reports to {out_dir}")


if __name__ == "__main__":
    main()
