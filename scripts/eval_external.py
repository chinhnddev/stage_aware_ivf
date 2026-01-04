"""
External evaluation on Hung Vuong dataset.

Usage:
    python scripts/eval_external.py --config configs/experiment/base.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ivf.config import load_experiment_config
from ivf.data.datasets import collate_batch
from ivf.data.label_schema import (
    EXPANSION_CLASSES,
    normalize_gardner_exp,
    normalize_gardner_grade,
    parse_gardner_components,
    q_proxy_from_components,
)
from ivf.eval import _normalize_day, build_quality_dataset_from_df, compute_metrics, predict, slice_by_day
from ivf.models.encoder import ConvNeXtMini
from ivf.models.multitask import MultiTaskEmbryoNet
from ivf.utils.guardrails import assert_no_hungvuong_training
from ivf.utils.logging import configure_logging
from ivf.utils.paths import ensure_outputs_dir
from ivf.utils.seed import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on external Hung Vuong dataset.")
    parser.add_argument("--config", default="configs/experiment/base.yaml", help="Experiment config path.")
    parser.add_argument("--checkpoint", default=None, help="Optional path to phase4 checkpoint.")
    parser.add_argument("--q_checkpoint", default=None, help="Optional path to phase4_q checkpoint.")
    parser.add_argument("--output-dir", default=None, help="Output directory for reports.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument("--dry_run", action="store_true", help="Validate pipeline without evaluation.")
    parser.add_argument(
        "--tune_day5_threshold",
        action="store_true",
        help="Tune threshold on external day5 subset (analysis-only).",
    )
    parser.add_argument(
        "--hv_split",
        choices=["predefined", "random"],
        default="predefined",
        help="Use predefined HV splits if available, otherwise random split.",
    )
    parser.add_argument("--hv_val_ratio", type=float, default=0.2, help="Validation ratio for HV split.")
    parser.add_argument(
        "--calibrate_mode",
        choices=["none", "threshold", "temperature"],
        default="threshold",
        help="Calibration mode using HV-val only.",
    )
    parser.add_argument(
        "--calibrate_stage",
        choices=["all", "day5"],
        default="day5",
        help="Calibration subset for HV-val.",
    )
    parser.add_argument(
        "--analysis_oracle_day5_threshold",
        action="store_true",
        help="Compute oracle day5 threshold on HV-test (analysis-only).",
    )
    parser.add_argument(
        "--analysis_morph_rule",
        action="store_true",
        help="Analysis-only: derive quality from predicted morphology.",
    )
    parser.add_argument(
        "--eval_q",
        action="store_true",
        help="Also evaluate q head and write external_q_metrics.json.",
    )
    return parser.parse_args()


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


def _load_threshold(reports_dir: Path, logger) -> float | None:
    threshold_path = reports_dir / "exp04_best_threshold.json"
    if not threshold_path.exists():
        return None
    try:
        with threshold_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        threshold = payload.get("threshold")
        if threshold is None:
            return None
        threshold = float(threshold)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.warning("Failed to read threshold from %s", threshold_path)
        return None
    logger.info("Using tuned threshold %.4f from %s", threshold, threshold_path)
    return threshold


def _f1_at_threshold(prob_good, y_true, threshold: float) -> float | None:
    if len(prob_good) == 0:
        return None
    tp = fp = fn = 0
    for prob, label in zip(prob_good, y_true):
        pred = 1 if prob >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def _tune_threshold(prob_good, y_true) -> tuple[float | None, float | None]:
    if len(prob_good) == 0:
        return None, None
    best_thresh = None
    best_f1 = -1.0
    for thresh in [i / 100 for i in range(5, 96)]:
        f1 = _f1_at_threshold(prob_good, y_true, thresh)
        if f1 is None:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
    return best_thresh, (None if best_f1 < 0 else float(best_f1))


def _apply_temperature(prob_good, temperature: float) -> list[float]:
    calibrated = []
    for prob in prob_good:
        p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
        logit = float(np.log(p / (1 - p)))
        logit /= temperature
        calibrated.append(float(1 / (1 + np.exp(-logit))))
    return calibrated


def _tune_temperature(prob_good, y_true) -> float | None:
    if len(prob_good) == 0:
        return None
    probs = np.array(prob_good, dtype=float)
    labels = np.array(y_true, dtype=int)
    best_t = None
    best_nll = float("inf")
    for t in np.linspace(0.5, 3.0, 26):
        calibrated = _apply_temperature(probs, t)
        cal = np.clip(np.array(calibrated), 1e-6, 1 - 1e-6)
        nll = -np.mean(labels * np.log(cal) + (1 - labels) * np.log(1 - cal))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def _split_external_df(df: pd.DataFrame, id_col: str, val_ratio: float, seed: int):
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("hv_val_ratio must be between 0 and 1.")
    if id_col in df.columns:
        groups = df[id_col].astype(str).fillna("")
        group_ids = []
        for idx, value in groups.items():
            text = str(value).strip()
            group_ids.append(text if text else f"missing_{idx}")
        df = df.copy()
        df["_group_id"] = group_ids
        unique_groups = list(dict.fromkeys(group_ids))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_groups)
        n_val = max(1, int(round(len(unique_groups) * val_ratio)))
        val_groups = set(unique_groups[:n_val])
        val_df = df[df["_group_id"].isin(val_groups)].copy()
        test_df = df[~df["_group_id"].isin(val_groups)].copy()
        val_df.drop(columns=["_group_id"], inplace=True)
        test_df.drop(columns=["_group_id"], inplace=True)
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = max(1, int(round(len(df) * val_ratio)))
        val_df = df.iloc[:n_val].copy()
        test_df = df.iloc[n_val:].copy()
    return val_df, test_df


def _load_predefined_splits(hung_cfg, df: pd.DataFrame):
    split_files = hung_cfg.get("split_files")
    if split_files:
        val_path = Path(split_files.get("val", ""))
        test_path = Path(split_files.get("test", ""))
        val_df = pd.read_csv(val_path) if val_path.exists() else pd.DataFrame()
        test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
        return val_df, test_df
    if "split" in df.columns:
        val_df = df[df["split"].astype(str).str.lower().isin({"val", "valid", "validation"})].copy()
        test_df = df[df["split"].astype(str).str.lower().isin({"test", "holdout"})].copy()
        return val_df, test_df
    return None, None


def _select_calibration_subset(prob_good, y_true, days, mode: str):
    if mode == "all":
        return prob_good, y_true
    filtered_prob = []
    filtered_true = []
    for prob, label, day in zip(prob_good, y_true, days):
        if day == 5:
            filtered_prob.append(prob)
            filtered_true.append(label)
    return filtered_prob, filtered_true


def _metrics_block(prob_good, y_true, days, threshold: float | None = None):
    overall = compute_metrics(prob_good, y_true)
    day3_probs, day3_true = slice_by_day({"prob_good": prob_good, "y_true": y_true, "day": days}, 3)
    day5_probs, day5_true = slice_by_day({"prob_good": prob_good, "y_true": y_true, "day": days}, 5)
    day3 = compute_metrics(day3_probs, day3_true)
    day5 = compute_metrics(day5_probs, day5_true)
    if threshold is not None:
        overall["f1"] = _f1_at_threshold(prob_good, y_true, threshold)
        day3["f1"] = _f1_at_threshold(day3_probs, day3_true, threshold)
        day5["f1"] = _f1_at_threshold(day5_probs, day5_true, threshold)
    return overall, day3, day5


def _predict_q(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    include_unlabeled: bool = True,
) -> Dict[str, list]:
    model.eval()
    q_scores = []
    y_true = []
    days = []
    image_ids = []
    domains = []
    exp_vals = []
    icm_vals = []
    te_vals = []
    grade_vals = []
    gardner_vals = []
    label_sources = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            outputs = model(images)
            if "q" not in outputs:
                raise RuntimeError("Model forward missing q output.")
            q_pred = outputs["q"].detach().cpu()
            targets = batch["targets"].get("quality")
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu()
            else:
                targets = torch.tensor(targets)

            meta = batch.get("meta", {})
            if isinstance(meta, list):
                ids = [m.get("id") for m in meta]
                day_vals = [m.get("day") for m in meta]
                domain_vals = [m.get("dataset") for m in meta]
                exp_meta = [m.get("exp") for m in meta]
                icm_meta = [m.get("icm") for m in meta]
                te_meta = [m.get("te") for m in meta]
                grade_meta = [m.get("grade") for m in meta]
                gardner_meta = [m.get("gardner") for m in meta]
                label_meta = [m.get("label_source") for m in meta]
            elif isinstance(meta, dict):
                ids = meta.get("id")
                day_vals = meta.get("day")
                domain_vals = meta.get("dataset")
                exp_meta = meta.get("exp")
                icm_meta = meta.get("icm")
                te_meta = meta.get("te")
                grade_meta = meta.get("grade")
                gardner_meta = meta.get("gardner")
                label_meta = meta.get("label_source")
            else:
                ids = None
                day_vals = None
                domain_vals = None
                exp_meta = None
                icm_meta = None
                te_meta = None
                grade_meta = None
                gardner_meta = None
                label_meta = None

            if ids is None:
                ids = [None] * len(q_pred)
            if day_vals is None:
                day_vals = [None] * len(q_pred)
            if domain_vals is None:
                domain_vals = [None] * len(q_pred)
            if exp_meta is None:
                exp_meta = [None] * len(q_pred)
            if icm_meta is None:
                icm_meta = [None] * len(q_pred)
            if te_meta is None:
                te_meta = [None] * len(q_pred)
            if grade_meta is None:
                grade_meta = [None] * len(q_pred)
            if gardner_meta is None:
                gardner_meta = [None] * len(q_pred)
            if label_meta is None:
                label_meta = [None] * len(q_pred)
            if not isinstance(day_vals, list):
                day_vals = [day_vals] * len(q_pred)
            if not isinstance(domain_vals, list):
                domain_vals = [domain_vals] * len(q_pred)
            if not isinstance(exp_meta, list):
                exp_meta = [exp_meta] * len(q_pred)
            if not isinstance(icm_meta, list):
                icm_meta = [icm_meta] * len(q_pred)
            if not isinstance(te_meta, list):
                te_meta = [te_meta] * len(q_pred)
            if not isinstance(grade_meta, list):
                grade_meta = [grade_meta] * len(q_pred)
            if not isinstance(gardner_meta, list):
                gardner_meta = [gardner_meta] * len(q_pred)
            if not isinstance(label_meta, list):
                label_meta = [label_meta] * len(q_pred)

            for i in range(len(q_pred)):
                label_val = None
                if i < len(targets) and targets[i] >= 0:
                    label_val = int(targets[i])
                if not include_unlabeled and label_val is None:
                    continue
                q_scores.append(float(q_pred[i]))
                y_true.append(label_val)
                image_ids.append(str(ids[i]))
                days.append(_normalize_day(day_vals[i]))
                domains.append(domain_vals[i])
                exp_vals.append(exp_meta[i])
                icm_vals.append(icm_meta[i])
                te_vals.append(te_meta[i])
                grade_vals.append(grade_meta[i])
                gardner_vals.append(gardner_meta[i])
                label_sources.append(label_meta[i])

    return {
        "q_score": q_scores,
        "y_true": y_true,
        "day": days,
        "image_id": image_ids,
        "domain": domains,
        "exp_raw": exp_vals,
        "icm_raw": icm_vals,
        "te_raw": te_vals,
        "grade_raw": grade_vals,
        "gardner_raw": gardner_vals,
        "label_source": label_sources,
    }


def _spearman_corr(x_vals, y_vals) -> Optional[float]:
    if len(x_vals) < 2:
        return None
    x_series = pd.Series(x_vals)
    y_series = pd.Series(y_vals)
    x_rank = x_series.rank(method="average").to_numpy(dtype=float)
    y_rank = y_series.rank(method="average").to_numpy(dtype=float)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _regression_metrics(scores, targets) -> Dict[str, Optional[float]]:
    if not scores:
        return {"mae": None, "rmse": None, "spearman": None}
    scores_arr = np.array(scores, dtype=float)
    targets_arr = np.array(targets, dtype=float)
    mae = float(np.mean(np.abs(scores_arr - targets_arr)))
    rmse = float(np.sqrt(np.mean((scores_arr - targets_arr) ** 2)))
    spearman = _spearman_corr(scores_arr, targets_arr)
    return {"mae": mae, "rmse": rmse, "spearman": spearman}


def _compute_q_proxy_values(preds: Dict[str, list], use_exp_cols: bool, weights: dict) -> Dict[str, list]:
    q_proxy = []
    q_source = []
    exp_gt = []
    icm_gt = []
    te_gt = []
    for exp_raw, icm_raw, te_raw, grade_raw, gardner_raw in zip(
        preds.get("exp_raw", []),
        preds.get("icm_raw", []),
        preds.get("te_raw", []),
        preds.get("grade_raw", []),
        preds.get("gardner_raw", []),
    ):
        if use_exp_cols:
            exp_val = normalize_gardner_exp(exp_raw)
            icm_val = normalize_gardner_grade(icm_raw)
            te_val = normalize_gardner_grade(te_raw)
            q_val = q_proxy_from_components(exp_val, icm_val, te_val, weights=weights)
            source = "exp_icm_te"
        else:
            raw = gardner_raw if gardner_raw is not None else grade_raw
            components = parse_gardner_components(raw)
            exp_val = icm_val = te_val = None
            if components is not None:
                exp_val, icm_val, te_val = components
            q_val = q_proxy_from_components(exp_val, icm_val, te_val, weights=weights)
            source = "gardner_parse"
        exp_gt.append(exp_val)
        icm_gt.append(icm_val)
        te_gt.append(te_val)
        q_proxy.append(q_val)
        q_source.append(source)
    return {
        "q_proxy": q_proxy,
        "q_proxy_source": q_source,
        "exp_gt": exp_gt,
        "icm_gt": icm_gt,
        "te_gt": te_gt,
    }


def _filter_binary_preds(preds: Dict[str, list]) -> Dict[str, list]:
    scores = []
    labels = []
    days = []
    domains = []
    for score, label, day, domain in zip(
        preds.get("q_score", []),
        preds.get("y_true", []),
        preds.get("day", []),
        preds.get("domain", []),
    ):
        if label is None:
            continue
        scores.append(score)
        labels.append(int(label))
        days.append(day)
        domains.append(domain)
    return {"q_score": scores, "y_true": labels, "day": days, "domain": domains}


def _load_q_thresholds(reports_dir: Path, logger) -> dict:
    path = reports_dir / "q_thresholds.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.warning("Failed to read q thresholds from %s", path)
        return {}
    return payload if isinstance(payload, dict) else {}


def load_checkpoint(model: MultiTaskEmbryoNet, checkpoint_path: Path, logger=None) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        if logger:
            logger.warning("Missing keys when loading checkpoint: %s", missing)
        else:
            print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        if logger:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)
        else:
            print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    if args.tune_day5_threshold:
        args.analysis_oracle_day5_threshold = True
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.device is not None:
        cfg.device = args.device

    set_global_seed(cfg.seed, deterministic=True)
    logs_dir = ensure_outputs_dir(cfg.outputs.logs_dir)
    logger = configure_logging(logs_dir / "train.log")
    logger.info("Seed=%s", cfg.seed)
    if args.tune_day5_threshold:
        logger.info("Using --tune_day5_threshold as analysis_oracle_day5_threshold.")

    data_cfg = cfg.data
    hung_cfg_path = data_cfg.hungvuong_config
    if not hung_cfg_path:
        raise ValueError("Missing hungvuong_config in experiment config.")
    hung_cfg = OmegaConf.load(hung_cfg_path)

    dataset_types = []
    for path in [data_cfg.blastocyst_config, data_cfg.humanembryo2_config, data_cfg.quality_config]:
        dataset_cfg = OmegaConf.load(path)
        dataset_types.append(str(dataset_cfg.get("dataset_type", "")))
    assert_no_hungvuong_training(dataset_types)

    model = build_model(cfg)
    checkpoint_path = args.checkpoint or Path(cfg.outputs.checkpoints_dir) / "phase4_quality.ckpt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.dry_run:
        logger.info("Dry run: checkpoint found at %s", checkpoint_path)
        return
    logger.info("Loading checkpoint weights from %s", checkpoint_path)
    load_checkpoint(model, checkpoint_path, logger=logger)
    logger.info("Checkpoint weights loaded.")

    df = pd.read_csv(hung_cfg["csv_path"])
    val_df = test_df = None
    if args.hv_split == "predefined":
        val_df, test_df = _load_predefined_splits(hung_cfg, df)
        if val_df is None or test_df is None or val_df.empty or test_df.empty:
            logger.warning("No predefined HV splits found; falling back to random split.")
            val_df, test_df = _split_external_df(df, hung_cfg["id_col"], args.hv_val_ratio, cfg.seed)
    else:
        val_df, test_df = _split_external_df(df, hung_cfg["id_col"], args.hv_val_ratio, cfg.seed)

    if val_df is None or test_df is None:
        raise ValueError("Failed to create HV val/test splits.")
    logger.info("HV split sizes: val=%s test=%s", len(val_df), len(test_df))

    val_dataset = build_quality_dataset_from_df(
        val_df,
        hung_cfg,
        image_size=cfg.transforms.image_size,
        normalize=cfg.transforms.normalize,
        mean=list(cfg.transforms.mean) if cfg.transforms.mean is not None else None,
        std=list(cfg.transforms.std) if cfg.transforms.std is not None else None,
    )
    test_dataset = build_quality_dataset_from_df(
        test_df,
        hung_cfg,
        image_size=cfg.transforms.image_size,
        normalize=cfg.transforms.normalize,
        mean=list(cfg.transforms.mean) if cfg.transforms.mean is not None else None,
        std=list(cfg.transforms.std) if cfg.transforms.std is not None else None,
    )
    if len(test_dataset) == 0:
        raise ValueError("External test dataset is empty after label filtering.")
    if len(val_dataset) == 0:
        logger.warning("External val dataset is empty after label filtering; calibration disabled.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
    )

    device_str = str(args.device or cfg.device or "cpu")
    device = torch.device(device_str)
    if "cuda" in device_str and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; using CPU.")
        device = torch.device("cpu")
    model.to(device)

    if args.dry_run:
        logger.info("Dry run: checkpoint loaded, validating dataloader and forward pass.")
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch["image"].to(device))
                if "quality" not in outputs:
                    raise RuntimeError("Model forward missing quality logits.")
                prob_good = torch.softmax(outputs["quality"], dim=-1)[:, 1]
                if prob_good.numel() == 0:
                    raise RuntimeError("Dry run produced empty prob_good output.")
                break
        logger.info("Dry run complete.")
        return

    preds_val = predict(model, val_loader, device, return_morph=False)
    preds_test = predict(model, test_loader, device, return_morph=args.analysis_morph_rule)

    output_dir = ensure_outputs_dir(args.output_dir or cfg.outputs.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_threshold = _load_threshold(output_dir, logger)
    if source_threshold is None:
        source_threshold = 0.5
        logger.warning("Source threshold missing; using 0.5 for zero-shot F1.")

    zero_overall, zero_day3, zero_day5 = _metrics_block(
        preds_test["prob_good"],
        preds_test["y_true"],
        preds_test["day"],
        threshold=source_threshold,
    )

    metrics = {
        "zero_shot": {
            "threshold_source": source_threshold,
            "overall": zero_overall,
            "day3": zero_day3,
            "day5": zero_day5,
        }
    }

    calibrated_block = None
    if args.calibrate_mode != "none" and len(preds_val["prob_good"]) > 0:
        cal_probs, cal_true = _select_calibration_subset(
            preds_val["prob_good"],
            preds_val["y_true"],
            preds_val["day"],
            args.calibrate_stage,
        )
        if len(cal_probs) == 0:
            logger.warning("Calibration subset is empty; skipping calibration.")
        elif args.calibrate_mode == "threshold":
            tuned_thresh, tuned_f1 = _tune_threshold(cal_probs, cal_true)
            if tuned_thresh is not None:
                cal_overall, cal_day3, cal_day5 = _metrics_block(
                    preds_test["prob_good"],
                    preds_test["y_true"],
                    preds_test["day"],
                    threshold=tuned_thresh,
                )
                calibrated_block = {
                    "method": "threshold",
                    "threshold": tuned_thresh,
                    "stage": args.calibrate_stage,
                    "f1_val": tuned_f1,
                    "overall": cal_overall,
                    "day3": cal_day3,
                    "day5": cal_day5,
                }
        elif args.calibrate_mode == "temperature":
            tuned_temp = _tune_temperature(cal_probs, cal_true)
            if tuned_temp is not None:
                calibrated_probs = _apply_temperature(preds_test["prob_good"], tuned_temp)
                cal_overall, cal_day3, cal_day5 = _metrics_block(
                    calibrated_probs,
                    preds_test["y_true"],
                    preds_test["day"],
                    threshold=0.5,
                )
                calibrated_block = {
                    "method": "temperature",
                    "temperature": tuned_temp,
                    "threshold": 0.5,
                    "stage": args.calibrate_stage,
                    "overall": cal_overall,
                    "day3": cal_day3,
                    "day5": cal_day5,
                }

    if calibrated_block:
        metrics["calibrated"] = calibrated_block

    if args.analysis_oracle_day5_threshold:
        day5_probs, day5_true = slice_by_day(preds_test, 5)
        tuned_thresh, tuned_f1 = _tune_threshold(day5_probs, day5_true)
        if tuned_thresh is not None:
            oracle_overall, oracle_day3, oracle_day5 = _metrics_block(
                preds_test["prob_good"],
                preds_test["y_true"],
                preds_test["day"],
                threshold=tuned_thresh,
            )
            metrics["analysis_oracle"] = {
                "threshold": tuned_thresh,
                "f1_day5": tuned_f1,
                "overall": oracle_overall,
                "day3": oracle_day3,
                "day5": oracle_day5,
            }

    if args.analysis_morph_rule and "morph_pred" in preds_test:
        exp_vals = [EXPANSION_CLASSES[idx] for idx in preds_test["morph_pred"]["exp"]]
        icm_vals = preds_test["morph_pred"]["icm"]
        te_vals = preds_test["morph_pred"]["te"]
        rule_probs = []
        for exp, icm, te in zip(exp_vals, icm_vals, te_vals):
            good = int(exp >= 3 and icm in {0, 1} and te in {0, 1})
            rule_probs.append(float(good))
        rule_overall, rule_day3, rule_day5 = _metrics_block(
            rule_probs,
            preds_test["y_true"],
            preds_test["day"],
            threshold=0.5,
        )
        metrics["analysis_morph_rule"] = {
            "overall": rule_overall,
            "day3": rule_day3,
            "day5": rule_day5,
        }
    metrics_path = output_dir / "external_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preds_path = output_dir / "external_predictions.csv"
    with open(preds_path, "w", encoding="utf-8") as f:
        f.write("image_id,prob_good,y_true,day,split\n")
        for image_id, prob, y, day in zip(
            preds_test["image_id"],
            preds_test["prob_good"],
            preds_test["y_true"],
            preds_test["day"],
        ):
            day_val = "" if day is None else day
            f.write(f"{image_id},{prob:.6f},{y},{day_val},test\n")

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved predictions to %s", preds_path)
    logger.info("Zero-shot: %s", metrics.get("zero_shot"))
    if "calibrated" in metrics:
        logger.info("Calibrated: %s", metrics.get("calibrated"))
    if "analysis_oracle" in metrics:
        logger.info("Analysis oracle: %s", metrics.get("analysis_oracle"))
    if "analysis_morph_rule" in metrics:
        logger.info("Analysis morph rule: %s", metrics.get("analysis_morph_rule"))

    if args.eval_q:
        q_ckpt_path = args.q_checkpoint or (Path(cfg.outputs.checkpoints_dir) / "phase4_q.ckpt")
        q_ckpt_path = Path(q_ckpt_path)
        if not q_ckpt_path.exists():
            logger.warning("Q checkpoint not found: %s; skipping q eval.", q_ckpt_path)
            return
        q_model = build_model(cfg)
        logger.info("Loading Q checkpoint weights from %s", q_ckpt_path)
        load_checkpoint(q_model, q_ckpt_path, logger=logger)
        q_model.to(device)

        q_extra_cols = ["exp", "icm", "te", "grade", "gardner"]
        grade_col = hung_cfg.get("grade_col") or hung_cfg.get("label_col")
        if grade_col and grade_col not in q_extra_cols:
            q_extra_cols.append(grade_col)

        q_val_dataset = build_quality_dataset_from_df(
            val_df,
            hung_cfg,
            image_size=cfg.transforms.image_size,
            normalize=cfg.transforms.normalize,
            mean=list(cfg.transforms.mean) if cfg.transforms.mean is not None else None,
            std=list(cfg.transforms.std) if cfg.transforms.std is not None else None,
            keep_unlabeled=True,
            extra_meta_cols=q_extra_cols,
        )
        q_test_dataset = build_quality_dataset_from_df(
            test_df,
            hung_cfg,
            image_size=cfg.transforms.image_size,
            normalize=cfg.transforms.normalize,
            mean=list(cfg.transforms.mean) if cfg.transforms.mean is not None else None,
            std=list(cfg.transforms.std) if cfg.transforms.std is not None else None,
            keep_unlabeled=True,
            extra_meta_cols=q_extra_cols,
        )
        q_val_loader = DataLoader(
            q_val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_batch,
        )
        q_test_loader = DataLoader(
            q_test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_batch,
        )

        q_preds_val = _predict_q(q_model, q_val_loader, device, include_unlabeled=True)
        q_preds_test = _predict_q(q_model, q_test_loader, device, include_unlabeled=True)

        q_cfg = getattr(cfg.training, "q", None)
        q_weights = getattr(q_cfg, "q_weights", {}) if q_cfg is not None else {}
        use_exp_cols_val = all(col in val_df.columns for col in ("exp", "icm", "te"))
        use_exp_cols_test = all(col in test_df.columns for col in ("exp", "icm", "te"))
        proxy_val = _compute_q_proxy_values(q_preds_val, use_exp_cols_val, q_weights)
        proxy_test = _compute_q_proxy_values(q_preds_test, use_exp_cols_test, q_weights)

        def _proxy_metrics(preds, proxy):
            scores = []
            targets = []
            for score, target in zip(preds.get("q_score", []), proxy.get("q_proxy", [])):
                if target is None:
                    continue
                scores.append(score)
                targets.append(target)
            return _regression_metrics(scores, targets), scores, targets

        proxy_val_metrics, _, proxy_val_targets = _proxy_metrics(q_preds_val, proxy_val)
        proxy_test_metrics, _, proxy_test_targets = _proxy_metrics(q_preds_test, proxy_test)

        q_metrics = {
            "proxy_regression": {
                "val": proxy_val_metrics,
                "test": proxy_test_metrics,
            }
        }

        labeled_test = _filter_binary_preds(q_preds_test)
        if labeled_test["y_true"]:
            q_overall = compute_metrics(labeled_test["q_score"], labeled_test["y_true"])
            q_day3_probs, q_day3_true = slice_by_day(
                {"prob_good": labeled_test["q_score"], "y_true": labeled_test["y_true"], "day": labeled_test["day"]},
                3,
            )
            q_day5_probs, q_day5_true = slice_by_day(
                {"prob_good": labeled_test["q_score"], "y_true": labeled_test["y_true"], "day": labeled_test["day"]},
                5,
            )
            q_day3 = compute_metrics(q_day3_probs, q_day3_true)
            q_day5 = compute_metrics(q_day5_probs, q_day5_true)
            q_metrics["binary"] = {
                "overall": q_overall,
                "day3": q_day3,
                "day5": q_day5,
            }

            thresholds = _load_q_thresholds(output_dir, logger)
            global_thresh = None
            if isinstance(thresholds.get("global"), dict):
                global_thresh = thresholds["global"].get("threshold")

            def _pick_threshold(domain, day):
                by_domain = thresholds.get("by_domain", {})
                by_stage = thresholds.get("by_stage", {})
                if domain and isinstance(by_domain, dict) and domain in by_domain:
                    return by_domain[domain].get("threshold")
                if day is not None and isinstance(by_stage, dict) and str(day) in by_stage:
                    return by_stage[str(day)].get("threshold")
                return global_thresh

            def _threshold_metrics(scores, labels, days, domains):
                if not scores:
                    return {"f1": None, "acc": None}
                preds = []
                for score, day_val, domain in zip(scores, days, domains):
                    thresh = _pick_threshold(domain, day_val)
                    if thresh is None:
                        thresh = 0.5
                    preds.append(1 if score >= thresh else 0)
                y = np.array(labels)
                p = np.array(preds)
                acc = float((p == y).mean()) if len(y) else None
                tp = int(((p == 1) & (y == 1)).sum())
                fp = int(((p == 1) & (y == 0)).sum())
                fn = int(((p == 0) & (y == 1)).sum())
                denom = (2 * tp + fp + fn)
                f1 = (2 * tp / denom) if denom > 0 else 0.0
                return {"f1": f1, "acc": acc}

            q_thresh_overall = _threshold_metrics(
                labeled_test["q_score"],
                labeled_test["y_true"],
                labeled_test["day"],
                labeled_test["domain"],
            )
            day3_domains = [d for d, day in zip(labeled_test["domain"], labeled_test["day"]) if day == 3]
            day5_domains = [d for d, day in zip(labeled_test["domain"], labeled_test["day"]) if day == 5]
            q_thresh_day3 = _threshold_metrics(q_day3_probs, q_day3_true, [3] * len(q_day3_probs), day3_domains)
            q_thresh_day5 = _threshold_metrics(q_day5_probs, q_day5_true, [5] * len(q_day5_probs), day5_domains)

            q_metrics["thresholded"] = {
                "overall": q_thresh_overall,
                "day3": q_thresh_day3,
                "day5": q_thresh_day5,
            }
        elif proxy_test_targets:
            if proxy_val_targets:
                derived_threshold = float(np.median(proxy_val_targets))
                threshold_source = "val_median"
            else:
                derived_threshold = float(np.median(proxy_test_targets))
                threshold_source = "test_median"

            derived_scores = []
            derived_labels = []
            derived_days = []
            for score, target, day in zip(q_preds_test["q_score"], proxy_test["q_proxy"], q_preds_test["day"]):
                if target is None:
                    continue
                derived_scores.append(score)
                derived_labels.append(1 if target >= derived_threshold else 0)
                derived_days.append(day)

            proxy_overall = compute_metrics(derived_scores, derived_labels)
            proxy_day3_probs, proxy_day3_true = slice_by_day(
                {"prob_good": derived_scores, "y_true": derived_labels, "day": derived_days},
                3,
            )
            proxy_day5_probs, proxy_day5_true = slice_by_day(
                {"prob_good": derived_scores, "y_true": derived_labels, "day": derived_days},
                5,
            )
            q_metrics["proxy_binary"] = {
                "threshold": derived_threshold,
                "threshold_source": threshold_source,
                "overall": proxy_overall,
                "day3": compute_metrics(proxy_day3_probs, proxy_day3_true),
                "day5": compute_metrics(proxy_day5_probs, proxy_day5_true),
            }
        q_metrics_path = output_dir / "external_q_metrics.json"
        with open(q_metrics_path, "w", encoding="utf-8") as f:
            json.dump(q_metrics, f, indent=2)

        q_preds_path = output_dir / "external_q_predictions.csv"
        with open(q_preds_path, "w", encoding="utf-8") as f:
            f.write("image_id,domain,stage,q_score,label_binary,exp_gt,icm_gt,te_gt,gardner_raw,q_proxy_external,q_proxy_source\n")
            for idx, (image_id, score, label, day_val, domain) in enumerate(
                zip(
                    q_preds_test["image_id"],
                    q_preds_test["q_score"],
                    q_preds_test["y_true"],
                    q_preds_test["day"],
                    q_preds_test["domain"],
                )
            ):
                stage = "" if day_val is None else day_val
                dom = "" if domain is None else domain
                exp_gt = proxy_test["exp_gt"][idx] if idx < len(proxy_test["exp_gt"]) else None
                icm_gt = proxy_test["icm_gt"][idx] if idx < len(proxy_test["icm_gt"]) else None
                te_gt = proxy_test["te_gt"][idx] if idx < len(proxy_test["te_gt"]) else None
                gardner_raw = q_preds_test["gardner_raw"][idx] if idx < len(q_preds_test["gardner_raw"]) else None
                if gardner_raw is None:
                    gardner_raw = q_preds_test["grade_raw"][idx] if idx < len(q_preds_test["grade_raw"]) else None
                q_proxy_val = proxy_test["q_proxy"][idx] if idx < len(proxy_test["q_proxy"]) else None
                q_source = proxy_test["q_proxy_source"][idx] if idx < len(proxy_test["q_proxy_source"]) else None
                label_str = "" if label is None else str(label)
                exp_str = "" if exp_gt is None else str(exp_gt)
                icm_str = "" if icm_gt is None else str(icm_gt)
                te_str = "" if te_gt is None else str(te_gt)
                gardner_str = "" if gardner_raw is None else str(gardner_raw)
                proxy_str = "" if q_proxy_val is None else f"{q_proxy_val:.6f}"
                source_str = "" if q_source is None else str(q_source)
                f.write(
                    f"{image_id},{dom},{stage},{score:.6f},{label_str},{exp_str},{icm_str},{te_str},{gardner_str},{proxy_str},{source_str}\n"
                )

        logger.info("Saved q metrics to %s", q_metrics_path)
        logger.info("Saved q predictions to %s", q_preds_path)


if __name__ == "__main__":
    main()
