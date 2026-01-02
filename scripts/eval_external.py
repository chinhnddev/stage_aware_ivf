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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ivf.config import load_experiment_config
from ivf.data.datasets import collate_batch
from ivf.eval import build_hungvuong_quality_dataset, compute_metrics, predict, slice_by_day
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
    if not prob_good:
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
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.device is not None:
        cfg.device = args.device

    set_global_seed(cfg.seed, deterministic=True)
    logs_dir = ensure_outputs_dir(cfg.outputs.logs_dir)
    logger = configure_logging(logs_dir / "train.log")

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

    dataset = build_hungvuong_quality_dataset(
        hung_cfg,
        image_size=cfg.transforms.image_size,
        normalize=cfg.transforms.normalize,
        mean=list(cfg.transforms.mean) if cfg.transforms.mean is not None else None,
        std=list(cfg.transforms.std) if cfg.transforms.std is not None else None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
    )

    device = torch.device("cpu")
    if str(cfg.device).startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available; using CPU.")
    model.to(device)

    preds = predict(model, dataloader, device)

    output_dir = ensure_outputs_dir(args.output_dir or cfg.outputs.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = _load_threshold(output_dir, logger)

    overall = compute_metrics(preds["prob_good"], preds["y_true"])
    day3_probs, day3_true = slice_by_day(preds, 3)
    day5_probs, day5_true = slice_by_day(preds, 5)
    day3 = compute_metrics(day3_probs, day3_true)
    day5 = compute_metrics(day5_probs, day5_true)
    if threshold is not None:
        overall["f1"] = _f1_at_threshold(preds["prob_good"], preds["y_true"], threshold)
        day3["f1"] = _f1_at_threshold(day3_probs, day3_true, threshold)
        day5["f1"] = _f1_at_threshold(day5_probs, day5_true, threshold)

    metrics = {
        "overall": overall,
        "day3": day3,
        "day5": day5,
    }
    if threshold is not None:
        metrics["threshold"] = threshold
    metrics_path = output_dir / "external_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preds_path = output_dir / "external_predictions.csv"
    with open(preds_path, "w", encoding="utf-8") as f:
        f.write("image_id,prob_good,y_true,day,split\n")
        for image_id, prob, y, day in zip(preds["image_id"], preds["prob_good"], preds["y_true"], preds["day"]):
            day_val = "" if day is None else day
            f.write(f"{image_id},{prob:.6f},{y},{day_val},external\n")

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved predictions to %s", preds_path)
    logger.info("Overall: %s", overall)
    logger.info("Day3: %s", day3)
    logger.info("Day5: %s", day5)


if __name__ == "__main__":
    main()
