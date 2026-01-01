"""
End-to-end smoke test using synthetic data.

Usage:
    python scripts/smoke_e2e.py --config configs/experiment/smoke.yaml --fast
"""

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ivf.config import load_experiment_config
from ivf.data.adapters.synthetic import create_synthetic_splits, generate_synthetic_metadata
from ivf.utils.paths import ensure_outputs_dir
from ivf.utils.seed import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic end-to-end smoke test.")
    parser.add_argument("--config", default="configs/experiment/smoke.yaml", help="Experiment config path.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument("--fast", action="store_true", help="Use minimal dataset size.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    set_global_seed(cfg.seed, deterministic=True)

    synthetic_cfg = OmegaConf.load(cfg.data.blastocyst_config)
    images_dir = Path(synthetic_cfg["root_dir"])
    csv_path = Path(synthetic_cfg["csv_path"])
    splits_dir = Path(cfg.data.splits_base_dir) / Path(cfg.data.blastocyst_config).stem

    num_samples = 6 if args.fast else 24
    image_size = (64, 64)

    ensure_outputs_dir(str(images_dir))
    ensure_outputs_dir(str(csv_path))
    ensure_outputs_dir(str(splits_dir))

    generate_synthetic_metadata(images_dir, csv_path, num_samples, image_size=image_size, seed=cfg.seed)
    create_synthetic_splits(
        csv_path,
        images_dir,
        splits_dir,
        group_col=synthetic_cfg.get("split", {}).get("group_col", "patient_id"),
        val_ratio=synthetic_cfg.get("split", {}).get("val_ratio", 0.2),
        test_ratio=synthetic_cfg.get("split", {}).get("test_ratio", 0.2),
        seed=cfg.seed,
    )

    phases = ["morph", "stage", "joint", "quality"]
    for phase in phases:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_phase.py"),
            "--phase",
            phase,
            "--config",
            args.config,
            "--device",
            cfg.device,
            "--num_workers",
            str(cfg.num_workers),
            "--seed",
            str(cfg.seed),
        ]
        subprocess.run(cmd, check=True)

    eval_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_external.py"),
        "--config",
        args.config,
        "--device",
        cfg.device,
        "--num_workers",
        str(cfg.num_workers),
        "--seed",
        str(cfg.seed),
    ]
    subprocess.run(eval_cmd, check=True)

    checkpoints_dir = ensure_outputs_dir(cfg.outputs.checkpoints_dir)
    checkpoints = [
        checkpoints_dir / "phase1_morph.ckpt",
        checkpoints_dir / "phase2_stage.ckpt",
        checkpoints_dir / "phase3_joint.ckpt",
        checkpoints_dir / "phase4_quality.ckpt",
    ]
    for path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    reports_dir = ensure_outputs_dir(cfg.outputs.reports_dir)
    metrics_path = reports_dir / "external_metrics.json"
    preds_path = reports_dir / "external_predictions.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Missing outputs/reports/external_metrics.json")
    if not preds_path.exists():
        raise FileNotFoundError("Missing outputs/reports/external_predictions.csv")

    print("Smoke test complete: outputs verified.")


if __name__ == "__main__":
    main()
