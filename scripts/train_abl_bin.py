"""
Train ABL-BIN in-domain binary classifier on Hung Vuong (HV).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy, BinaryF1Score
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from omegaconf import OmegaConf

from ivf.config import load_experiment_config
from ivf.data.datasets import BaseImageDataset, collate_batch
from ivf.data.transforms import assert_no_augmentation, get_eval_transforms
from ivf.models.baseline_binary import BaselineBinaryClassifier
from ivf.train.callbacks import BestMetricCheckpoint, StepProgressLogger
from ivf.utils.guardrails import assert_no_day_feature, assert_no_segmentation_inputs
from ivf.utils.logging import configure_logging
from ivf.utils.paths import ensure_outputs_dir
from ivf.utils.seed import set_global_seed

from abl_bin_utils import (
    _resolve_group_col,
    assert_no_group_overlap,
    build_records,
    collect_predictions,
    compute_pos_weight,
    load_hv_metadata,
    load_or_create_trainval_split,
    log_label_distribution,
    log_split_sizes,
    resolve_group_col,
    split_hv_by_folder,
    metrics_block,
    tune_threshold_with_mode,
    write_json,
)


class AblBinLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        pos_weight: Optional[float],
        backbone: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.backbone = backbone
        if pos_weight is not None:
            self.register_buffer("pos_weight_tensor", torch.tensor(float(pos_weight)))
        else:
            self.pos_weight_tensor = None

        self.val_metrics = nn.ModuleDict(
            {
                "auroc": BinaryAUROC(),
                "auprc": BinaryAveragePrecision(),
                "f1": BinaryF1Score(),
                "acc": BinaryAccuracy(),
            }
        )
        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _shared_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        assert_no_day_feature(batch)
        assert_no_segmentation_inputs(batch)
        images = batch["image"]
        targets = batch["targets"]["quality"]
        logits = self.model(images)
        mask = targets >= 0
        if mask.any():
            pos_weight = self.pos_weight_tensor if self.pos_weight_tensor is not None else None
            loss = F.binary_cross_entropy_with_logits(logits[mask], targets[mask].float(), pos_weight=pos_weight)
            probs = torch.sigmoid(logits[mask])
            targets = targets[mask]
        else:
            loss = torch.tensor(0.0, device=logits.device)
            probs = None
            targets = None
        return {"loss": loss, "probs": probs, "targets": targets}

    def training_step(self, batch: Dict, batch_idx: int):
        outputs = self._shared_step(batch)
        batch_size = batch["image"].shape[0]
        self.log("train/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return outputs["loss"]

    def validation_step(self, batch: Dict, batch_idx: int):
        outputs = self._shared_step(batch)
        batch_size = batch["image"].shape[0]
        self.log("val/loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if outputs["probs"] is not None and outputs["targets"] is not None:
            probs = outputs["probs"]
            targets = outputs["targets"].long()
            for key, metric in self.val_metrics.items():
                metric.update(probs, targets)
                self.log(f"val/{key}", metric, on_epoch=True, prog_bar=False, batch_size=batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ABL-BIN on Hung Vuong.")
    parser.add_argument("--config", default="configs/experiment/base.yaml", help="Experiment config path.")
    parser.add_argument("--backbone", choices=["resnet50", "convnext_mini"], default="convnext_mini")
    parser.add_argument("--hv_root", required=True, help="HV dataset root containing train/ and test/ folders.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-run.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument("--tune_on", choices=["overall", "day5"], default="overall", help="Threshold tuning mode.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio from HV/train.")
    parser.add_argument("--save_dir", default="outputs/abl_bin", help="Output directory for reports.")
    return parser.parse_args()


def _load_splits(args, hung_cfg, logger, seed: int) -> Dict[str, pd.DataFrame]:
    df = load_hv_metadata(hung_cfg, args.hv_root)
    hv_train_full, hv_test = split_hv_by_folder(df)
    group_candidates = ["patient_id", "embryo_id", "cycle_id"]
    group_col = resolve_group_col(hv_train_full, group_candidates)
    id_col = hung_cfg.get("id_col")
    split_path = Path("outputs/splits") / f"hv_trainval_seed{seed}.csv"
    hv_train, hv_val = load_or_create_trainval_split(
        hv_train_full,
        group_col=group_col,
        val_ratio=args.val_ratio,
        seed=seed,
        split_path=split_path,
        logger=logger,
        id_col=id_col,
    )
    logger.info("USING HV/train for training+val split; USING HV/test for final reporting")
    log_split_sizes({"train": hv_train, "val": hv_val, "test": hv_test}, logger)
    return {"train": hv_train, "val": hv_val, "test": hv_test}


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_train_transforms(image_size: int, normalize: bool, mean, std):
    ops = [
        T.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
    ]
    tail = []
    if normalize:
        mean_vals = list(mean) if mean is not None else [0.5, 0.5, 0.5]
        std_vals = list(std) if std is not None else [0.5, 0.5, 0.5]
        tail.append(T.ToTensor())
        tail.append(T.Normalize(mean=mean_vals, std=std_vals))
    else:
        tail.append(T.ToTensor())
    return T.Compose(ops + tail)


def _parse_seeds(args, default_seed: int) -> list[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.seed is not None:
        return [int(args.seed)]
    return [int(default_seed)]


def _run_once(seed: int, args, cfg, hung_cfg, split_dfs, group_col, logger, save_dir: Path) -> Dict[str, float]:
    set_global_seed(seed, deterministic=True)
    logger.info("Seed=%s", seed)

    log_label_distribution(split_dfs["train"], hung_cfg, "train", logger)
    log_label_distribution(split_dfs["val"], hung_cfg, "val", logger)
    log_label_distribution(split_dfs["test"], hung_cfg, "test", logger)

    train_records = build_records(split_dfs["train"], hung_cfg, "train", logger, group_col=group_col)
    val_records = build_records(split_dfs["val"], hung_cfg, "val", logger, group_col=group_col)
    test_records = build_records(split_dfs["test"], hung_cfg, "test", logger, group_col=group_col)

    transforms_cfg = cfg.transforms
    train_tf = _build_train_transforms(
        transforms_cfg.image_size,
        transforms_cfg.normalize,
        transforms_cfg.mean,
        transforms_cfg.std,
    )
    eval_tf = get_eval_transforms(
        image_size=transforms_cfg.image_size,
        normalize=transforms_cfg.normalize,
        mean=list(transforms_cfg.mean) if transforms_cfg.mean is not None else None,
        std=list(transforms_cfg.std) if transforms_cfg.std is not None else None,
    )
    assert_no_augmentation(eval_tf)

    root_dir = args.hv_root
    train_dataset = BaseImageDataset(train_records, transform=train_tf, include_meta_day=True, root_dir=root_dir)
    val_dataset = BaseImageDataset(val_records, transform=eval_tf, include_meta_day=True, root_dir=root_dir)
    test_dataset = BaseImageDataset(test_records, transform=eval_tf, include_meta_day=True, root_dir=root_dir)

    batch_size = args.batch_size or cfg.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        worker_init_fn=_seed_worker,
    )

    encoder_cfg = cfg.model.encoder
    model = BaselineBinaryClassifier(
        backbone=args.backbone,
        in_channels=encoder_cfg.in_channels,
        dims=encoder_cfg.dims,
        feature_dim=encoder_cfg.feature_dim,
        weights_path=encoder_cfg.weights_path,
        head="linear",
        mlp_hidden=encoder_cfg.feature_dim,
        dropout=0.0,
    )
    train_labels = [sample["targets"]["quality"] for sample in train_records]
    pos_weight = compute_pos_weight(train_labels, logger)
    lightning_module = AblBinLightningModule(
        model=model,
        lr=args.lr or cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        pos_weight=pos_weight,
        backbone=args.backbone,
    )

    loggers = []
    logs_dir = ensure_outputs_dir(cfg.outputs.logs_dir)
    if cfg.logging.tensorboard:
        loggers.append(TensorBoardLogger(save_dir=logs_dir / "tensorboard", name=f"abl_bin_seed{seed}"))
    if cfg.logging.csv:
        loggers.append(CSVLogger(save_dir=logs_dir / "csv", name=f"abl_bin_seed{seed}"))

    accelerator = "cpu"
    devices = 1
    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    elif str(cfg.device).startswith("cuda"):
        logger.warning("CUDA requested but not available; falling back to CPU.")

    max_epochs = args.max_epochs if args.max_epochs is not None else 10
    callbacks = [
        StepProgressLogger(single_line=True, update_every_n_steps=50),
        EarlyStopping(monitor="val/auprc", mode="max", patience=5),
    ]

    ckpt_path = ensure_outputs_dir(cfg.outputs.checkpoints_dir) / f"abl_bin_{args.backbone}_seed{seed}.ckpt"
    callbacks.append(
        BestMetricCheckpoint(
            ckpt_path=ckpt_path,
            primary_metric="val/auprc",
            fallback_metric="val/loss",
            primary_mode="max",
            fallback_mode="min",
        )
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=min(50, max(1, len(train_loader) // 2)),
        logger=loggers if loggers else False,
        deterministic=True,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(logs_dir),
        callbacks=callbacks,
    )

    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if not ckpt_path.exists():
        trainer.save_checkpoint(str(ckpt_path))
        logger.warning("Best checkpoint missing; saved LAST checkpoint to %s", ckpt_path)
    else:
        logger.info("Saved BEST checkpoint: %s", ckpt_path)

    eval_device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
    eval_model = BaselineBinaryClassifier(
        backbone=args.backbone,
        in_channels=encoder_cfg.in_channels,
        dims=encoder_cfg.dims,
        feature_dim=encoder_cfg.feature_dim,
        weights_path=encoder_cfg.weights_path,
        head="linear",
        mlp_hidden=encoder_cfg.feature_dim,
        dropout=0.0,
    )
    eval_module = AblBinLightningModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model=eval_model,
        lr=args.lr or cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        pos_weight=pos_weight,
        backbone=args.backbone,
    )

    val_preds = collect_predictions(eval_module.model, val_loader, eval_device, "val")
    threshold_info = tune_threshold_with_mode(
        val_preds["prob_good"],
        val_preds["y_true"],
        val_preds["day"],
        args.tune_on,
        logger,
    )

    write_json(threshold_info, save_dir / "best_threshold.json")

    threshold_value = float(threshold_info.get("threshold", 0.5))
    logger.info("REPORTING ON TEST SPLIT")
    test_preds = collect_predictions(eval_module.model, test_loader, eval_device, "test")
    metrics = metrics_block(
        test_preds["prob_good"],
        test_preds["y_true"],
        test_preds["day"],
        threshold_value,
        logger,
    )
    payload = {
        "seed": seed,
        "backbone": args.backbone,
        "threshold": threshold_info,
        "imbalance": {"method": "pos_weight", "pos_weight": pos_weight, "max_clip": 20.0},
        "metrics": metrics,
    }
    write_json(payload, save_dir / "metrics_test.json")

    pred_labels = [1 if p >= threshold_value else 0 for p in test_preds["prob_good"]]
    pred_df = pd.DataFrame(
        {
            "image_path": test_preds["image_path"],
            "y_true": test_preds["y_true"],
            "p_good": test_preds["prob_good"],
            "y_pred": pred_labels,
            "day": test_preds["day"],
            "stage": test_preds["stage"],
            "group_id": test_preds["group_id"],
            "split": "test",
        }
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(save_dir / "predictions_test.csv", index=False)
    logger.info("Saved ABL-BIN reports to %s", save_dir)

    return {
        "overall_auprc": metrics["overall"].get("auprc"),
        "day5_auprc": metrics["day5"].get("auprc"),
        "day5_f1": metrics["day5"].get("f1"),
    }


def _summarize_metrics(values: list[Optional[float]]) -> Dict[str, Optional[float]]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None}
    arr = np.array(clean, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    hung_cfg = OmegaConf.load(cfg.data.hungvuong_config)
    logs_dir = ensure_outputs_dir(cfg.outputs.logs_dir)
    logger = configure_logging(logs_dir / "abl_bin.log")

    seeds = _parse_seeds(args, cfg.seed)
    multi_seed = len(seeds) > 1
    results = []
    for seed in seeds:
        split_dfs = _load_splits(args, hung_cfg, logger, seed)
        group_candidates = ["patient_id", "embryo_id", "cycle_id"]
        group_col = _resolve_group_col(split_dfs, group_candidates)
        assert_no_group_overlap(split_dfs, group_col, logger)
        save_dir = Path(args.save_dir)
        if multi_seed:
            save_dir = save_dir / f"seed_{seed}"
        results.append(_run_once(seed, args, cfg, hung_cfg, split_dfs, group_col, logger, save_dir))

    if len(seeds) > 1:
        summary = {
            "overall_auprc": _summarize_metrics([r.get("overall_auprc") for r in results]),
            "day5_auprc": _summarize_metrics([r.get("day5_auprc") for r in results]),
            "day5_f1": _summarize_metrics([r.get("day5_f1") for r in results]),
            "seeds": seeds,
        }
        summary_path = Path(args.save_dir) / "summary_mean_std.json"
        write_json(summary, summary_path)
        logger.info("Saved multi-seed summary to %s", summary_path)


if __name__ == "__main__":
    main()
