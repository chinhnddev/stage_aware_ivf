"""
Train a specific phase of the IVF multitask pipeline.

Usage:
    python scripts/train_phase.py --phase morph --config configs/experiment/base.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
import random
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from omegaconf import OmegaConf

from ivf.config import load_experiment_config, resolve_config_dict
from ivf.data.datamodule import IVFDataModule
from ivf.models.encoder import ConvNeXtMini
from ivf.models.multitask import MultiTaskEmbryoNet
from ivf.eval import compute_metrics, predict
from ivf.train.callbacks import BestMetricCheckpoint, StepProgressLogger
from ivf.train.lightning_module import MultiTaskLightningModule
from ivf.utils.guardrails import assert_no_hungvuong_training
from ivf.utils.logging import configure_logging
from ivf.utils.paths import ensure_outputs_dir
from ivf.utils.seed import set_global_seed


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
    if not missing and all(path.exists() for path in split_paths.values() if path is not None):
        group_col = split_cfg.get("group_col")
        if group_col and group_col in pd.read_csv(split_paths["train"]).columns:
            train_df = pd.read_csv(split_paths["train"])
            val_df = pd.read_csv(split_paths["val"])
            test_df = pd.read_csv(split_paths["test"]) if split_paths["test"].exists() else None
            splits = [train_df, val_df] + ([test_df] if test_df is not None else [])
            _assert_no_group_overlap([s for s in splits if s is not None], group_col)
        return split_paths
    if missing:
        logger.warning("Quality split files missing (%s); regenerating splits.", ", ".join(missing))

    df = pd.read_csv(quality_cfg.csv_path)
    df = _derive_quality_labels(df, logger)
    if df.empty:
        raise ValueError("No samples remain after deriving quality labels.")

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
    )


def get_prev_checkpoint(phase: str, checkpoints_dir: Path, cfg):
    overrides = getattr(cfg.outputs, "checkpoint_paths", {}) or {}
    if phase in overrides:
        return Path(overrides[phase])

    mapping = {
        "stage": "phase1_morph.ckpt",
        "joint": "phase2_stage.ckpt",
        "quality": "phase3_joint.ckpt",
    }
    if phase in mapping:
        return checkpoints_dir / mapping[phase]
    return None


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

    checkpoints_dir = ensure_outputs_dir(cfg.outputs.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ensure_outputs_dir(cfg.outputs.logs_dir)
    logger = configure_logging(logs_dir / "train.log")

    model = build_model(cfg)
    phase = args.phase
    phase_cfg = cfg.training
    loss_weights = resolve_config_dict(phase_cfg.loss_weights)
    freeze_cfg = resolve_config_dict(phase_cfg.freeze)

    dataset_types = []
    blast_cfg = OmegaConf.load(cfg.data.blastocyst_config)
    human_cfg = OmegaConf.load(cfg.data.humanembryo2_config)
    quality_cfg = OmegaConf.load(cfg.data.quality_config)
    for dataset_cfg in [blast_cfg, human_cfg, quality_cfg]:
        dataset_types.append(str(dataset_cfg.get("dataset_type", "")))
    assert_no_hungvuong_training(dataset_types)

    data_cfg = cfg.data
    splits_base_dir = data_cfg.splits_base_dir
    quality_split_files = quality_cfg.get("split_files")
    splits = {
        "blastocyst": _split_dir(splits_base_dir, data_cfg.blastocyst_config),
        "humanembryo2": _split_dir(splits_base_dir, data_cfg.humanembryo2_config),
        "quality": {k: str(v) for k, v in quality_split_files.items()} if quality_split_files else _split_dir(splits_base_dir, data_cfg.quality_config),
    }

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
        )
    elif prev_ckpt is not None and not prev_ckpt.exists():
        logger.warning("Previous checkpoint not found at %s; continuing with random initialization.", prev_ckpt)

    transforms_cfg = cfg.transforms
    train_transform_level = getattr(transforms_cfg, phase)
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
    )

    max_epochs = phase_cfg.epochs.get(phase, 1)
    if args.dry_run:
        datamodule.setup()
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
    if args.enable_progress_bar:
        enable_progress_bar = True
    elif args.disable_progress_bar:
        enable_progress_bar = False
    else:
        enable_progress_bar = False
    callbacks = [StepProgressLogger()]
    best_ckpt_path = None
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
        log_every_n_steps=50,
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
            logger.warning("Best checkpoint not found; saved last checkpoint to %s", ckpt_path)

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
        )
        _tune_quality_threshold(eval_module.model, datamodule.val_dataloader(), eval_device, reports_dir, logger)
        _run_quality_test_eval(eval_module.model, datamodule.test_dataloader(), reports_dir, eval_device, logger)

    if phase != "quality":
        trainer.save_checkpoint(ckpt_path)
        logger.info("Saved checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
