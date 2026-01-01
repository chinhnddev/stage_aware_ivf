"""
Train a specific phase of the IVF multitask pipeline.

Usage:
    python scripts/train_phase.py --phase morph --config configs/experiment/base.yaml
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from omegaconf import OmegaConf

from ivf.config import load_experiment_config, resolve_config_dict
from ivf.data.datamodule import IVFDataModule
from ivf.models.encoder import ConvNeXtMini
from ivf.models.multitask import MultiTaskEmbryoNet
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
    parser.add_argument("--dry_run", action="store_true", help="Validate pipeline without training.")
    return parser.parse_args()


def _split_dir(splits_base_dir: str, config_path: str) -> Path:
    return Path(splits_base_dir) / Path(config_path).stem


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

    lightning_module = MultiTaskLightningModule(
        model=model,
        phase=phase,
        lr=phase_cfg.lr,
        weight_decay=phase_cfg.weight_decay,
        loss_weights=loss_weights,
        freeze_config=freeze_cfg,
    )

    prev_ckpt = get_prev_checkpoint(phase, checkpoints_dir, cfg)
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
        )

    data_cfg = cfg.data
    splits_base_dir = data_cfg.splits_base_dir
    splits = {
        "blastocyst": _split_dir(splits_base_dir, data_cfg.blastocyst_config),
        "humanembryo2": _split_dir(splits_base_dir, data_cfg.humanembryo2_config),
        "quality": _split_dir(splits_base_dir, data_cfg.quality_config),
    }

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
    if args.disable_progress_bar:
        enable_progress_bar = False
    elif args.enable_progress_bar:
        enable_progress_bar = True
    else:
        enable_progress_bar = not sys.platform.startswith("win")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        enable_checkpointing=False,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=10,
        logger=loggers if loggers else False,
        deterministic=True,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(logs_dir),
    )

    trainer.fit(lightning_module, datamodule=datamodule)

    ckpt_name = {
        "morph": "phase1_morph.ckpt",
        "stage": "phase2_stage.ckpt",
        "joint": "phase3_joint.ckpt",
        "quality": "phase4_quality.ckpt",
    }[phase]
    ckpt_path = checkpoints_dir / ckpt_name
    trainer.save_checkpoint(ckpt_path)
    logger.info("Saved checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
