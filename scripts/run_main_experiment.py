"""
Run the full IVF experiment pipeline and external evaluation.

Usage:
    python scripts/run_main_experiment.py --config configs/experiment/base.yaml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ivf.config import load_experiment_config
from ivf.utils.logging import configure_logging
from ivf.utils.manifest import write_run_manifest
from ivf.utils.paths import ensure_outputs_dir
from ivf.utils.seed import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline.")
    parser.add_argument("--config", default="configs/experiment/base.yaml", help="Experiment config path.")
    parser.add_argument("--skip_train_if_exists", type=str, default="true", help="Skip training if checkpoint exists.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument("--dry_run", action="store_true", help="Validate pipeline without training.")
    return parser.parse_args()


def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def run_cmd(args: list) -> None:
    print("Running:", " ".join(args))
    subprocess.run(args, check=True)


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
    logger.info("Seed=%s", cfg.seed)
    skip_train = str_to_bool(args.skip_train_if_exists)

    checkpoints_dir = ensure_outputs_dir(cfg.outputs.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    phase_ckpts = {
        "morph": checkpoints_dir / "phase1_morph.ckpt",
        "stage": checkpoints_dir / "phase2_stage.ckpt",
        "joint": checkpoints_dir / "phase3_joint.ckpt",
        "quality": checkpoints_dir / "phase4_quality.ckpt",
    }

    for phase in ["morph", "stage", "joint", "quality"]:
        ckpt = phase_ckpts[phase]
        if skip_train and ckpt.exists():
            print(f"Skipping {phase}: checkpoint exists at {ckpt}")
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_phase.py"),
            "--phase",
            phase,
            "--config",
            args.config,
        ]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.device is not None:
            cmd += ["--device", args.device]
        if args.num_workers is not None:
            cmd += ["--num_workers", str(args.num_workers)]
        if args.dry_run:
            cmd += ["--dry_run"]
        run_cmd(cmd)

    eval_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_external.py"),
        "--config",
        args.config,
    ]
    if args.seed is not None:
        eval_cmd += ["--seed", str(args.seed)]
    if args.device is not None:
        eval_cmd += ["--device", args.device]
    if args.num_workers is not None:
        eval_cmd += ["--num_workers", str(args.num_workers)]
    if args.dry_run:
        eval_cmd += ["--dry_run"]
    run_cmd(eval_cmd)

    export_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_results.py"),
        "--config",
        args.config,
    ]
    if args.seed is not None:
        export_cmd += ["--seed", str(args.seed)]
    if args.device is not None:
        export_cmd += ["--device", args.device]
    if args.num_workers is not None:
        export_cmd += ["--num_workers", str(args.num_workers)]
    if args.dry_run:
        export_cmd += ["--dry_run"]
    run_cmd(export_cmd)

    reports_dir = ensure_outputs_dir(cfg.outputs.reports_dir)
    if not args.dry_run:
        report_path = reports_dir / "external_metrics.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print("External metrics summary:")
            if "zero_shot" in report:
                for mode in ["zero_shot", "calibrated"]:
                    block = report.get(mode)
                    if not block:
                        continue
                    for key in ["overall", "day3", "day5"]:
                        metrics = block.get(key, {})
                        print(f"{mode}/{key}: AUROC={metrics.get('auroc')} AUPRC={metrics.get('auprc')} F1={metrics.get('f1')}")
            else:
                for key in ["overall", "day3", "day5"]:
                    metrics = report.get(key, {})
                    print(f"{key}: AUROC={metrics.get('auroc')} AUPRC={metrics.get('auprc')} F1={metrics.get('f1')}")
        else:
            print("External metrics report not found.")

        outputs_root = ensure_outputs_dir("outputs")
        write_run_manifest(cfg, phase_ckpts, outputs_dir=outputs_root)
        logger.info("Run manifest written to %s", outputs_root / "run_manifest.yaml")


if __name__ == "__main__":
    main()
