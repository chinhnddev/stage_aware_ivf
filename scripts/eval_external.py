"""
External evaluation on Hung Vuong dataset.

Usage:
    python scripts/eval_external.py --config configs/experiment/base.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ivf.config import load_experiment_config
from ivf.data.datasets import collate_batch
from ivf.data.label_schema import EXPANSION_CLASSES
from ivf.eval import build_quality_dataset_from_df, compute_metrics, predict, slice_by_day
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


if __name__ == "__main__":
    main()
