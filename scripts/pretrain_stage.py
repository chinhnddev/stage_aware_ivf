"""
Supervised stage pretraining using HumanEmbryo2 metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from omegaconf import OmegaConf

from ivf.data.datasets import BaseImageDataset, collate_batch, make_full_target_dict
from ivf.data.transforms import get_eval_transforms, get_train_transforms
from ivf.models.encoder import ConvNeXtMini
from ivf.models.heads import StageHead
from ivf.utils.logging import configure_logging
from ivf.utils.paths import ensure_outputs_dir


class StagePretrainNet(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.stage = StageHead(feature_dim)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.stage(features)


def _normalize_stage(value: Optional[str]) -> Optional[str]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip().lower()
    return text if text else None


def _split_df(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if "split" not in df.columns:
        raise ValueError("Missing split column in stage metadata.")
    split = df["split"].astype(str).str.lower()
    return {
        "train": df[split == "train"].copy(),
        "val": df[split == "val"].copy(),
        "test": df[split == "test"].copy(),
    }


def _build_mapping(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = []
    for value in train_df["stage"].tolist():
        norm = _normalize_stage(value)
        if norm is not None:
            labels.append(norm)
    unique = sorted(set(labels))
    if not unique:
        raise ValueError("No valid stage labels found in train split.")
    to_index = {label: idx for idx, label in enumerate(unique)}
    to_label = {idx: label for label, idx in to_index.items()}
    return to_index, to_label


def _build_records(
    df: pd.DataFrame,
    stage_to_idx: Dict[str, int],
    split_name: str,
    logger,
) -> list:
    records = []
    dropped_missing = 0
    dropped_unseen = 0
    for _, row in df.iterrows():
        stage = _normalize_stage(row.get("stage"))
        if stage is None:
            dropped_missing += 1
            continue
        if stage not in stage_to_idx:
            dropped_unseen += 1
            continue
        record = {
            "image_path": row.get("image_path"),
            "targets": make_full_target_dict(stage=stage_to_idx[stage]),
            "meta": {
                "id": row.get("embryo_id"),
                "stage": stage,
                "split": split_name,
            },
        }
        records.append(record)
    if dropped_missing or dropped_unseen:
        logger.warning(
            "Stage %s drop counts: missing=%s unseen=%s",
            split_name,
            dropped_missing,
            dropped_unseen,
        )
    logger.info("Stage %s records: %s", split_name, len(records))
    return records


def _build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def _lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain stage classifier using HumanEmbryo2 metadata.")
    parser.add_argument("--config", required=True, help="Stage pretraining config path.")
    parser.add_argument("--csv_path", default="data/metadata/humanembryo2.csv")
    parser.add_argument("--root_dir", default=None, help="Root dir for image paths.")
    parser.add_argument("--output_dir", default="outputs/pretrain_stage", help="Output directory under outputs/.")
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    output_dir = ensure_outputs_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "pretrain_stage.log")

    df = pd.read_csv(args.csv_path)
    splits = _split_df(df)
    stage_to_idx, idx_to_stage = _build_mapping(splits["train"])
    mapping_path = output_dir / "stage_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump({"to_index": stage_to_idx, "to_label": idx_to_stage}, f, indent=2)
    logger.info("Saved stage mapping to %s", mapping_path)

    train_records = _build_records(splits["train"], stage_to_idx, "train", logger)
    val_records = _build_records(splits["val"], stage_to_idx, "val", logger)
    test_records = _build_records(splits["test"], stage_to_idx, "test", logger)

    image_size = int(cfg.training.image_size)
    train_tf = get_train_transforms(
        "light",
        image_size=image_size,
        normalize=bool(cfg.training.normalize),
        mean=list(cfg.training.mean),
        std=list(cfg.training.std),
        crop_size=image_size,
        crop_scale=(0.8, 1.0),
        crop_ratio=(0.9, 1.1),
        rotation_degrees=float(cfg.training.rotation_degrees),
        enable_vertical_flip=False,
        translate_max=0.0,
    )
    eval_tf = get_eval_transforms(
        image_size=image_size,
        normalize=bool(cfg.training.normalize),
        mean=list(cfg.training.mean),
        std=list(cfg.training.std),
        crop_size=image_size,
    )
    logger.info("Train transforms: %s", train_tf)
    logger.info("Eval transforms: %s", eval_tf)

    root_dir = args.root_dir or cfg.data.root_dir
    train_ds = BaseImageDataset(train_records, transform=train_tf, include_meta_day=False, root_dir=root_dir)
    val_ds = BaseImageDataset(val_records, transform=eval_tf, include_meta_day=False, root_dir=root_dir)
    test_ds = BaseImageDataset(test_records, transform=eval_tf, include_meta_day=False, root_dir=root_dir)

    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)

    model_cfg = cfg.model
    encoder = ConvNeXtMini(
        in_channels=3,
        dims=list(model_cfg.dims),
        feature_dim=int(model_cfg.feature_dim),
        width_mult=float(getattr(model_cfg, "width_mult", 1.0)),
        depth_mult=float(getattr(model_cfg, "depth_mult", 1.0)),
    )
    model = StagePretrainNet(encoder, feature_dim=encoder.feature_dim, num_classes=len(stage_to_idx))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: total=%s", total_params)

    device = args.device or cfg.device
    device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    total_steps = int(cfg.training.epochs) * len(train_loader)
    warmup_steps = int(cfg.training.warmup_epochs) * len(train_loader)
    scheduler = _build_scheduler(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    f1_metric = MulticlassF1Score(num_classes=len(stage_to_idx), average="macro").to(device)

    best_f1 = -1.0
    history = []
    for epoch in range(int(cfg.training.epochs)):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["targets"]["stage"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))

        model.eval()
        f1_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["targets"]["stage"].to(device, non_blocking=True)
                logits = model(images)
                preds = torch.argmax(logits, dim=-1)
                f1_metric.update(preds, targets)
        val_f1 = float(f1_metric.compute().item()) if len(val_ds) > 0 else 0.0

        history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_macro_f1": val_f1})
        logger.info("Epoch %s/%s train_loss=%.4f val_macro_f1=%.4f", epoch + 1, cfg.training.epochs, avg_loss, val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = output_dir / "stage_best.ckpt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "stage_mapping": stage_to_idx,
                    "epoch": epoch + 1,
                    "best_val_macro_f1": best_f1,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
            logger.info("Saved best checkpoint to %s", ckpt_path)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"best_val_macro_f1": best_f1, "history": history}, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    if len(test_ds) > 0:
        model.eval()
        f1_metric.reset()
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["targets"]["stage"].to(device, non_blocking=True)
                logits = model(images)
                preds = torch.argmax(logits, dim=-1)
                f1_metric.update(preds, targets)
        test_f1 = float(f1_metric.compute().item())
        logger.info("Test macro-F1: %.4f", test_f1)


if __name__ == "__main__":
    main()
