"""
Export paper-ready results tables from reports.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ivf.config import load_experiment_config
from ivf.utils.logging import configure_logging
from ivf.utils.paths import ensure_outputs_dir
from ivf.utils.seed import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Export paper-ready results tables.")
    parser.add_argument("--config", default="configs/experiment/base.yaml", help="Experiment config path (unused).")
    parser.add_argument("--seed", type=int, default=None, help="Unused; for interface consistency.")
    parser.add_argument("--device", default=None, help="Unused; for interface consistency.")
    parser.add_argument("--num_workers", type=int, default=None, help="Unused; for interface consistency.")
    parser.add_argument("--dry_run", action="store_true", help="Validate pipeline without writing outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    logs_dir = ensure_outputs_dir(cfg.outputs.logs_dir)
    reports_dir = ensure_outputs_dir(cfg.outputs.reports_dir)
    logger = configure_logging(logs_dir / "train.log")
    set_global_seed(cfg.seed if args.seed is None else args.seed, deterministic=True)

    metrics_path = reports_dir / "external_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Missing outputs/reports/external_metrics.json")

    if args.dry_run:
        logger.info("Dry run: found external_metrics.json")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    if "zero_shot" in metrics:
        for mode_key in ["zero_shot", "calibrated", "analysis_oracle", "analysis_morph_rule"]:
            block = metrics.get(mode_key)
            if not block:
                continue
            for split in ["overall", "day3", "day5"]:
                row = block.get(split, {})
                rows.append(
                    {
                        "mode": mode_key,
                        "split": split,
                        "auroc": row.get("auroc"),
                        "auprc": row.get("auprc"),
                        "f1": row.get("f1"),
                    }
                )
    else:
        for split in ["overall", "day3", "day5"]:
            row = metrics.get(split, {})
            rows.append(
                {
                    "mode": "legacy",
                    "split": split,
                    "auroc": row.get("auroc"),
                    "auprc": row.get("auprc"),
                    "f1": row.get("f1"),
                }
            )

    output_csv = reports_dir / "summary_table.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "split", "auroc", "auprc", "f1"])
        writer.writeheader()
        writer.writerows(rows)

    latex_lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Mode & Split & AUROC & AUPRC & F1 \\",
        r"\midrule",
    ]
    for row in rows:
        latex_lines.append(
            f"{row['mode']} & {row['split']} & {row['auroc']} & {row['auprc']} & {row['f1']} \\\\"
        )
    latex_lines += [r"\bottomrule", r"\end{tabular}"]
    output_tex = reports_dir / "summary_table.tex"
    output_tex.write_text("\n".join(latex_lines), encoding="utf-8")

    logger.info("Saved summary table to %s", output_csv)
    logger.info("Saved LaTeX table to %s", output_tex)


if __name__ == "__main__":
    main()
