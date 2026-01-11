"""
Prepare deterministic HV train/val/test splits for ABL-BIN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ivf.data.splits import save_splits, split_by_group
from ivf.eval_label_sources import resolve_quality_label
from ivf.utils.seed import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare HV splits for ABL-BIN.")
    parser.add_argument("--config", default="configs/data/hungvuong.yaml", help="Dataset config path.")
    parser.add_argument("--output_dir", default="data/processed/splits/hungvuong", help="Output splits directory.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio from train split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split.")
    parser.add_argument("--group_col", default=None, help="Override group column (default uses config id_col).")
    parser.add_argument("--dry_run", action="store_true", help="Validate and print stats without writing splits.")
    return parser.parse_args()


def _normalize_split(value) -> str:
    return str(value).strip().lower()


def _label_stats(df: pd.DataFrame, cfg: dict) -> dict:
    image_col = cfg.get("image_col", "image_path")
    quality_col = cfg.get("quality_col")
    grade_col = cfg.get("grade_col") or cfg.get("label_col")
    allow_grade = bool(cfg.get("allow_grade_labels", False))
    stats = {"good": 0, "poor": 0, "missing": 0}
    for _, row in df.iterrows():
        label, _ = resolve_quality_label(
            row,
            image_col=image_col,
            quality_col=quality_col,
            grade_col=grade_col,
            allow_grade=allow_grade,
        )
        if label is None:
            stats["missing"] += 1
        elif label == 1:
            stats["good"] += 1
        else:
            stats["poor"] += 1
    return stats


def _assert_group_overlap(train_df, val_df, test_df, group_col: str) -> None:
    def _groups(df):
        return set(df[group_col].dropna().astype(str))
    groups = {"train": _groups(train_df), "val": _groups(val_df), "test": _groups(test_df)}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[left].intersection(groups[right])
        if overlap:
            raise ValueError(f"Group leakage between {left}/{right} for {group_col}: {sorted(list(overlap))[:3]}")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed, deterministic=True)

    cfg = OmegaConf.load(args.config)
    csv_path = Path(cfg["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing HV metadata CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError("HV metadata missing 'split' column (expected train/test).")

    split_series = df["split"].apply(_normalize_split)
    train_df = df[split_series == "train"].copy()
    val_df = df[split_series.isin({"val", "valid", "validation"})].copy()
    test_df = df[split_series.isin({"test", "holdout"})].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("HV metadata must include non-empty train and test splits.")

    group_col = args.group_col or cfg.get("id_col")
    if group_col and group_col not in df.columns:
        group_col = None

    if val_df.empty:
        splits = split_by_group(
            train_df,
            group_col=group_col,
            val_ratio=args.val_ratio,
            test_ratio=0.0,
            seed=args.seed,
        )
        train_df = splits["train"].copy()
        val_df = splits["val"].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    print(f"Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print("Train label stats:", _label_stats(train_df, cfg))
    print("Val label stats:", _label_stats(val_df, cfg))
    print("Test label stats:", _label_stats(test_df, cfg))

    if group_col:
        _assert_group_overlap(train_df, val_df, test_df, group_col)
        print(f"Group overlap check passed for {group_col}.")
    else:
        print("Group column not found; overlap check skipped.")

    if args.dry_run:
        print("Dry run: splits not written.")
        return

    output_dir = Path(args.output_dir)
    save_splits({"train": train_df, "val": val_df, "test": test_df}, output_dir=output_dir)
    print(f"Saved HV splits to {output_dir}")


if __name__ == "__main__":
    main()
