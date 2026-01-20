"""
Paper-protocol morphology evaluation on Blastocyst-Dataset testset_filenames.csv.

Protocol (paper reproduction, matching reference repo scripts):
- EXP 5-class: 0..4 (0=EXP1 ... 4=EXP5)
- ICM/TE 4-class: 0=A, 1=B, 2=C, 3=ND
- ICM/TE metrics are computed ONLY on samples where BOTH exp_pred and exp_gt are >=3
  (i.e., exp label not in {0,1}), matching calculate_model_metrics.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from ivf.config.loader import load_experiment_config
from ivf.data.datasets import BaseImageDataset
from ivf.data.transforms import get_eval_transforms
from ivf.models.factory import build_model_from_config
from ivf.utils.logging import get_logger


PAPER_CLASSES_4 = ["A", "B", "C", "ND"]
EXP_LABELS = ["0", "1", "2", "3", "4", "5"]  # 5 = not assessable (paper script)


def _coerce_int(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _load_testset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] >= 4:
        first_row = [str(v).strip().lower() for v in df.iloc[0].tolist()]
        if any("exp" in v or "icm" in v or "te" in v for v in first_row):
            df = pd.read_csv(path)
    if df.shape[1] < 4:
        raise ValueError(f"testset must have 4 columns: filename, exp, icm, te (got {df.shape[1]}).")
    df = df.iloc[:, :4].copy()
    df.columns = ["filename", "raw_exp", "raw_icm", "raw_te"]
    df["filename"] = df["filename"].astype(str).str.strip()
    for col in ("raw_exp", "raw_icm", "raw_te"):
        df[col] = df[col].apply(_coerce_int)
    return df


def _resolve_blastocyst_paths(exp_cfg) -> Tuple[Path, Optional[Path]]:
    blast_cfg_path = Path(exp_cfg.data.blastocyst_config)
    blast_cfg = json.loads(json.dumps(__import__("yaml").safe_load(blast_cfg_path.read_text(encoding="utf-8"))))
    csv_path = Path(blast_cfg["csv_path"])
    root_dir = Path(blast_cfg.get("root_dir")) if blast_cfg.get("root_dir") else None
    return csv_path, root_dir


def _build_test_records(metadata_csv: Path, testset_df: pd.DataFrame) -> list:
    meta = pd.read_csv(metadata_csv)
    if "image_path" not in meta.columns:
        raise ValueError(f"metadata csv missing image_path column: {metadata_csv}")
    meta = meta.copy()
    meta["filename"] = meta["image_path"].astype(str).apply(lambda p: Path(str(p).replace('\\', '/')).name)
    meta = meta.drop_duplicates(subset=["filename"])
    meta_map = meta.set_index("filename")

    records = []
    missing = 0
    invalid = 0
    for _, row in testset_df.iterrows():
        filename = row["filename"]
        if filename not in meta_map.index:
            missing += 1
            continue
        raw_exp = row["raw_exp"]
        raw_icm = row["raw_icm"]
        raw_te = row["raw_te"]
        if raw_exp is None or raw_icm is None or raw_te is None:
            invalid += 1
            continue
        # Match paper script normalization (invalid -> "not assessable"):
        # - EXP: invalid -> 5
        # - ICM/TE: invalid -> 3 (ND)
        if raw_exp < 0 or raw_exp > 4:
            raw_exp = 5
        if raw_icm < 0 or raw_icm > 2:
            raw_icm = 3
        if raw_te < 0 or raw_te > 2:
            raw_te = 3
        if raw_exp == 5:
            invalid += 1
            continue
        exp_label = raw_exp
        targets = {
            "exp": exp_label,
            "icm": raw_icm,
            "te": raw_te,
            "exp_mask": 1,
            "icm_mask": 1,
            "te_mask": 1,
        }
        meta_row = meta_map.loc[filename]
        records.append(
            {
                "image_path": meta_row["image_path"],
                "targets": targets,
                "meta": {
                    "filename": filename,
                    "raw_exp": raw_exp,
                    "raw_icm": raw_icm,
                    "raw_te": raw_te,
                },
            }
        )

    logger = get_logger("ivf")
    logger.info("Paper test records: total=%s kept=%s missing=%s invalid=%s", len(testset_df), len(records), missing, invalid)
    if len(records) == 0:
        raise ValueError("No usable test records found.")
    return records


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, labels: Optional[list] = None) -> Dict:
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    kappa = float(cohen_kappa_score(y_true, y_pred))
    out = {
        "accuracy": acc,
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
        "cohen_kappa": kappa,
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return out, cm


def _extract_model_state_dict(checkpoint: dict) -> dict:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format: missing state_dict.")
    model_state = {}
    for key, value in state.items():
        if key.startswith("model."):
            model_state[key[len("model.") :]] = value
    return model_state if model_state else state


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--testset", required=True, help="paper testset_filenames.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    logger = get_logger("ivf")
    cfg = load_experiment_config(args.config)
    protocol = str(getattr(cfg.training.morph, "protocol", "paper")).lower()
    if protocol != "paper":
        raise ValueError(f"eval_morph_paper requires training.morph.protocol=paper (got {protocol}).")
    model_name = str(getattr(cfg.model, "name", "")).lower()
    if model_name not in {"morph_paper", "paper_morph", "paper"}:
        raise ValueError(f"eval_morph_paper requires model.name=morph_paper (got {model_name}).")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv, root_dir = _resolve_blastocyst_paths(cfg)
    testset_df = _load_testset(Path(args.testset))
    records = _build_test_records(metadata_csv, testset_df)

    transforms_cfg = cfg.transforms
    eval_tf = get_eval_transforms(
        image_size=int(getattr(transforms_cfg, "image_size", 224)),
        crop_size=int(getattr(transforms_cfg, "crop_size", 224)),
        normalize=bool(getattr(transforms_cfg, "normalize", True)),
        mean=list(getattr(transforms_cfg, "mean", [0.5, 0.5, 0.5])),
        std=list(getattr(transforms_cfg, "std", [0.5, 0.5, 0.5])),
    )
    dataset = BaseImageDataset(records, transform=eval_tf, include_meta_day=False, root_dir=str(root_dir) if root_dir else None)

    device = args.device or getattr(cfg, "device", "cpu")
    torch_device = torch.device(device)

    model = build_model_from_config(cfg, phase="morph")
    state = torch.load(str(ckpt_path), map_location="cpu")
    model_state = _extract_model_state_dict(state)
    incompat = model.load_state_dict(model_state, strict=False)
    if getattr(incompat, "missing_keys", None):
        logger.warning("Missing keys when loading checkpoint: %s", incompat.missing_keys[:10])
    if getattr(incompat, "unexpected_keys", None):
        logger.warning("Unexpected keys when loading checkpoint: %s", incompat.unexpected_keys[:10])
    model.to(torch_device)
    model.eval()

    batch_size = int(args.batch_size or getattr(cfg, "batch_size", 16))
    num_workers = int(args.num_workers or getattr(cfg, "num_workers", 4))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_exp_true, y_exp_pred = [], []
    y_icm_true_all, y_icm_pred_all = [], []
    y_te_true_all, y_te_pred_all = [], []

    for batch in loader:
        images = batch["image"].to(torch_device)
        targets = batch["targets"]
        exp_true = targets["exp"].cpu().numpy().astype(int)
        icm_true = targets["icm"].cpu().numpy().astype(int)
        te_true = targets["te"].cpu().numpy().astype(int)

        outputs = model(images)
        exp_logits = outputs["exp"]
        icm_logits = outputs["icm"]
        te_logits = outputs["te"]
        exp_pred = exp_logits.argmax(dim=-1).cpu().numpy().astype(int)
        icm_pred = icm_logits.argmax(dim=-1).cpu().numpy().astype(int)
        te_pred = te_logits.argmax(dim=-1).cpu().numpy().astype(int)

        y_exp_true.append(exp_true)
        y_exp_pred.append(exp_pred)
        y_icm_true_all.append(icm_true)
        y_icm_pred_all.append(icm_pred)
        y_te_true_all.append(te_true)
        y_te_pred_all.append(te_pred)

    y_exp_true = np.concatenate(y_exp_true)
    y_exp_pred = np.concatenate(y_exp_pred)
    y_icm_true_all = np.concatenate(y_icm_true_all)
    y_icm_pred_all = np.concatenate(y_icm_pred_all)
    y_te_true_all = np.concatenate(y_te_true_all)
    y_te_pred_all = np.concatenate(y_te_pred_all)

    # Match reference repo evaluation: only consider ICM/TE when exp_pred and exp_gt are >=3 (i.e., not in {0,1})
    gate = (y_exp_true >= 2) & (y_exp_pred >= 2)
    y_icm_true = y_icm_true_all[gate]
    y_icm_pred = y_icm_pred_all[gate]
    y_te_true = y_te_true_all[gate]
    y_te_pred = y_te_pred_all[gate]

    exp_metrics, exp_cm = _compute_metrics(y_exp_true, y_exp_pred, labels=[0, 1, 2, 3, 4, 5])
    icm_metrics, icm_cm = _compute_metrics(y_icm_true, y_icm_pred, labels=[0, 1, 2, 3])
    te_metrics, te_cm = _compute_metrics(y_te_true, y_te_pred, labels=[0, 1, 2, 3])

    (output_dir / "metrics_exp.json").write_text(json.dumps(exp_metrics, indent=2), encoding="utf-8")
    (output_dir / "metrics_icm.json").write_text(json.dumps(icm_metrics, indent=2), encoding="utf-8")
    (output_dir / "metrics_te.json").write_text(json.dumps(te_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(exp_cm, index=EXP_LABELS, columns=EXP_LABELS).to_csv(output_dir / "confusion_exp.csv")
    pd.DataFrame(icm_cm, index=PAPER_CLASSES_4, columns=PAPER_CLASSES_4).to_csv(output_dir / "confusion_icm.csv")
    pd.DataFrame(te_cm, index=PAPER_CLASSES_4, columns=PAPER_CLASSES_4).to_csv(output_dir / "confusion_te.csv")

    logger.info("Saved paper morphology reports to %s", output_dir)
    logger.info("EXP metrics: %s", exp_metrics)
    logger.info("ICM metrics: %s", icm_metrics)
    logger.info("TE metrics: %s", te_metrics)
    logger.info("ICM/TE eval gate: kept=%s / total=%s", int(gate.sum()), int(gate.shape[0]))


if __name__ == "__main__":
    main()
