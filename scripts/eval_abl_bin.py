"""
Evaluate ABL-BIN on Hung Vuong (HV) using predefined splits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from omegaconf import OmegaConf

from ivf.config import load_experiment_config
from ivf.data.datasets import BaseImageDataset, collate_batch
from ivf.data.transforms import assert_no_augmentation, get_eval_transforms
from ivf.models.baseline_binary import BaselineBinaryClassifier
from ivf.utils.logging import configure_logging
from ivf.utils.paths import ensure_outputs_dir
from ivf.utils.seed import set_global_seed

from abl_bin_utils import (
    _resolve_group_col,
    assert_no_group_overlap,
    build_records,
    collect_predictions,
    load_hv_metadata,
    load_or_create_trainval_split,
    log_label_distribution,
    log_split_sizes,
    metrics_block,
    resolve_group_col,
    split_hv_by_folder,
    tune_threshold_with_mode,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ABL-BIN on Hung Vuong.")
    parser.add_argument("--config", default="configs/experiment/base.yaml", help="Experiment config path.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--hv_root", required=True, help="HV dataset root containing train/ and test/ folders.")
    parser.add_argument("--seed", type=int, default=None, help="Seed to locate/reuse train/val split.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument("--save_dir", default="outputs/abl_bin", help="Report output directory.")
    parser.add_argument("--tune_on", choices=["overall", "day5"], default="overall", help="Threshold tuning mode.")
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
        val_ratio=0.2,
        seed=seed,
        split_path=split_path,
        logger=logger,
        id_col=id_col,
    )
    logger.info("USING HV/train for training+val split; USING HV/test for final reporting")
    split_dfs = {"train": hv_train, "val": hv_val, "test": hv_test}
    log_split_sizes(split_dfs, logger)
    group_col = _resolve_group_col(split_dfs, group_candidates)
    assert_no_group_overlap(split_dfs, group_col, logger)
    return split_dfs


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    seed = args.seed if args.seed is not None else cfg.seed
    set_global_seed(seed, deterministic=True)

    logs_dir = ensure_outputs_dir(cfg.outputs.logs_dir)
    logger = configure_logging(logs_dir / "abl_bin_eval.log")

    hung_cfg = OmegaConf.load(cfg.data.hungvuong_config)
    split_dfs = _load_splits(args, hung_cfg, logger, seed)

    log_label_distribution(split_dfs["train"], hung_cfg, "train", logger)
    log_label_distribution(split_dfs["val"], hung_cfg, "val", logger)
    log_label_distribution(split_dfs["test"], hung_cfg, "test", logger)
    group_candidates = ["patient_id", "embryo_id", "cycle_id"]
    group_col = _resolve_group_col(split_dfs, group_candidates)

    val_records = build_records(split_dfs["val"], hung_cfg, "val", logger, group_col=group_col)
    test_records = build_records(split_dfs["test"], hung_cfg, "test", logger, group_col=group_col)

    transforms_cfg = cfg.transforms
    eval_tf = get_eval_transforms(
        image_size=transforms_cfg.image_size,
        normalize=transforms_cfg.normalize,
        mean=list(transforms_cfg.mean) if transforms_cfg.mean is not None else None,
        std=list(transforms_cfg.std) if transforms_cfg.std is not None else None,
    )
    assert_no_augmentation(eval_tf)

    root_dir = args.hv_root
    val_dataset = BaseImageDataset(val_records, transform=eval_tf, include_meta_day=True, root_dir=root_dir)
    test_dataset = BaseImageDataset(test_records, transform=eval_tf, include_meta_day=True, root_dir=root_dir)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        worker_init_fn=_seed_worker,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        worker_init_fn=_seed_worker,
    )

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    backbone = hparams.get("backbone")
    if backbone is None:
        raise ValueError("Checkpoint missing backbone hyperparameter.")

    encoder_cfg = cfg.model.encoder
    model = BaselineBinaryClassifier(
        backbone=backbone,
        in_channels=encoder_cfg.in_channels,
        dims=encoder_cfg.dims,
        feature_dim=encoder_cfg.feature_dim,
        weights_path=encoder_cfg.weights_path,
        head="linear",
        mlp_hidden=encoder_cfg.feature_dim,
        dropout=0.0,
    )
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
    model.to(device)

    reports_dir = ensure_outputs_dir(args.save_dir)
    threshold_path = reports_dir / "best_threshold.json"
    threshold_info = None
    if threshold_path.exists():
        with threshold_path.open("r", encoding="utf-8") as f:
            threshold_info = json.load(f)
        if not isinstance(threshold_info, dict) or "threshold" not in threshold_info:
            logger.warning("Invalid threshold file at %s; re-tuning.", threshold_path)
            threshold_info = None
        else:
            logger.info("Loaded threshold from %s", threshold_path)

    if threshold_info is None:
        val_preds = collect_predictions(model, val_loader, device, "val")
        threshold_info = tune_threshold_with_mode(
            val_preds["prob_good"],
            val_preds["y_true"],
            val_preds["day"],
            args.tune_on,
            logger,
        )
    threshold_value = float(threshold_info.get("threshold", 0.5))

    logger.info("REPORTING ON TEST SPLIT")
    test_preds = collect_predictions(model, test_loader, device, "test")
    metrics = metrics_block(
        test_preds["prob_good"],
        test_preds["y_true"],
        test_preds["day"],
        threshold_value,
        logger,
    )

    write_json(threshold_info, reports_dir / "best_threshold.json")
    pos_weight = hparams.get("pos_weight")
    imbalance = {"method": "pos_weight" if pos_weight else "none", "pos_weight": pos_weight, "max_clip": 20.0}
    write_json(
        {"threshold": threshold_info, "imbalance": imbalance, "metrics": metrics},
        reports_dir / "metrics_test.json",
    )

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
    reports_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(reports_dir / "predictions_test.csv", index=False)
    logger.info("Saved ABL-BIN evaluation reports to %s", reports_dir)


if __name__ == "__main__":
    main()
