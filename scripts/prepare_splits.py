"""
Prepare data splits with leakage prevention.

Usage:
    python scripts/prepare_splits.py --config configs/data/blastocyst.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ivf.data.adapters import (
    humanembryo2_records_to_dataframe,
    hungvuong_records_to_dataframe,
    load_blastocyst_records,
    load_humanembryo2_records,
    load_hungvuong_records,
    records_to_dataframe,
)
from ivf.data.label_schema import map_gardner_to_quality
from ivf.data.splits import save_splits, split_by_group, summarize_distribution
from ivf.utils.seed import set_global_seed

MISSINGNESS_THRESHOLD = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data splits.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", default="data/processed/splits", help="Base output directory for splits.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split generation.")
    parser.add_argument("--device", default=None, help="Unused; for interface consistency.")
    parser.add_argument("--num_workers", type=int, default=None, help="Unused; for interface consistency.")
    parser.add_argument("--dry_run", action="store_true", help="Validate pipeline without writing splits.")
    return parser.parse_args()


def _resolve_group_col(df: pd.DataFrame, cfg: dict) -> str:
    split_cfg = cfg.get("split", {}) or {}
    group_col = split_cfg.get("group_col")
    id_col = cfg.get("id_col")

    if group_col and group_col in df.columns:
        missing_frac = float(df[group_col].isna().mean())
        if missing_frac <= MISSINGNESS_THRESHOLD:
            return group_col
        print(
            f"Warning: group_col '{group_col}' missingness {missing_frac:.1%} too high; "
            f"falling back to id_col '{id_col}'."
        )
    elif group_col:
        print(f"Warning: group_col '{group_col}' missing; falling back to id_col '{id_col}'.")

    fallback_col = id_col if id_col in df.columns else None
    if fallback_col is None and "id" in df.columns:
        fallback_col = "id"
        if id_col and id_col != "id":
            print(f"Warning: id_col '{id_col}' missing; falling back to 'id'.")

    if fallback_col is None:
        raise ValueError("Fallback id_col is missing; cannot perform group-wise split.")

    missing_frac = float(df[fallback_col].isna().mean())
    if missing_frac > MISSINGNESS_THRESHOLD:
        print(f"Warning: id_col '{fallback_col}' missingness {missing_frac:.1%}.")
    return fallback_col


def _normalize_split_paths(df: pd.DataFrame, root_dir: str) -> pd.DataFrame:
    if "image_path" not in df.columns:
        return df
    root = Path(root_dir)

    def _normalize(value):
        if pd.isna(value):
            return value
        path = Path(str(value).replace("\\", "/"))
        if path.is_absolute():
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        return path.as_posix()

    df = df.copy()
    df["image_path"] = df["image_path"].apply(_normalize)
    return df


def make_blastocyst_splits(cfg: dict):
    day_col = cfg.get("day_col") if cfg.get("include_meta_day", True) else None
    raw_df = pd.read_csv(cfg["csv_path"])
    records = load_blastocyst_records(
        Path(cfg["root_dir"]),
        Path(cfg["csv_path"]),
        image_col=cfg["image_col"],
        grade_col=cfg["label_col"],
        id_col=cfg["id_col"],
        day_col=day_col,
    )
    invalid_count = max(len(raw_df) - len(records), 0)
    if invalid_count:
        print(f"Blastocyst invalid Gardner labels: {invalid_count}/{len(raw_df)}")
    df = records_to_dataframe(records)
    df = _normalize_split_paths(df, cfg["root_dir"])
    split_cfg = cfg.get("split", {})
    splits = split_by_group(
        df,
        group_col=_resolve_group_col(df, cfg),
        val_ratio=split_cfg.get("val_ratio", 0.2),
        test_ratio=split_cfg.get("test_ratio", 0.0),
        seed=split_cfg.get("seed", 42),
    )
    summary = {k: summarize_distribution(v, stage_col=None, quality_col="quality", day_col=day_col) for k, v in splits.items()}
    return splits, summary


def make_humanembryo2_splits(cfg: dict):
    day_col = cfg.get("day_col") if cfg.get("include_meta_day", True) else None
    records = load_humanembryo2_records(
        Path(cfg["root_dir"]),
        Path(cfg["csv_path"]),
        image_col=cfg["image_col"],
        stage_col=cfg["label_col"],
        id_col=cfg["id_col"],
        day_col=day_col,
    )
    df = humanembryo2_records_to_dataframe(records)
    df = _normalize_split_paths(df, cfg["root_dir"])
    split_cfg = cfg.get("split", {})
    splits = split_by_group(
        df,
        group_col=_resolve_group_col(df, cfg),
        val_ratio=split_cfg.get("val_ratio", 0.2),
        test_ratio=split_cfg.get("test_ratio", 0.0),
        seed=split_cfg.get("seed", 42),
    )
    summary = {k: summarize_distribution(v, stage_col="stage", quality_col=None, day_col=day_col) for k, v in splits.items()}
    return splits, summary


def make_quality_public_splits(cfg: dict):
    day_col = cfg.get("day_col") if cfg.get("include_meta_day", True) else None
    df = pd.read_csv(cfg["csv_path"])
    df = df.copy()
    df["image_path"] = df[cfg["image_col"]].apply(lambda x: str(Path(cfg["root_dir"]) / str(x)))
    df = _normalize_split_paths(df, cfg["root_dir"])
    df["quality"] = df[cfg["label_col"]].apply(map_gardner_to_quality).apply(lambda x: x.value if x else None)
    unknown_count = int(df["quality"].isna().sum())
    if unknown_count:
        print(f"Quality public unknown labels: {unknown_count}/{len(df)}")
    df = df[df["quality"].notna()]
    if day_col and day_col in df.columns:
        df["day"] = df[day_col]
    if cfg.get("id_col") and cfg["id_col"] in df.columns:
        df["id"] = df[cfg["id_col"]]

    split_cfg = cfg.get("split", {})
    splits = split_by_group(
        df,
        group_col=_resolve_group_col(df, cfg),
        val_ratio=split_cfg.get("val_ratio", 0.1),
        test_ratio=split_cfg.get("test_ratio", 0.1),
        seed=split_cfg.get("seed", 42),
    )
    summary = {k: summarize_distribution(v, stage_col=None, quality_col="quality", day_col=day_col) for k, v in splits.items()}
    return splits, summary


def make_hungvuong_splits(cfg: dict):
    day_col = cfg.get("day_col") if cfg.get("include_meta_day", True) else None
    records = load_hungvuong_records(
        Path(cfg["root_dir"]),
        Path(cfg["csv_path"]),
        image_col=cfg["image_col"],
        id_col=cfg["id_col"],
        stage_col=cfg.get("label_col"),
        grade_col=cfg.get("grade_col") or cfg.get("label_col"),
        quality_col=cfg.get("quality_col"),
        day_col=day_col,
    )
    df = hungvuong_records_to_dataframe(records)
    df = _normalize_split_paths(df, cfg["root_dir"])
    if "quality" in df.columns:
        unknown_count = int(df["quality"].isna().sum())
        if unknown_count:
            print(f"Hung Vuong unknown quality labels: {unknown_count}/{len(df)}")
    # External test only
    splits = {"test": df}
    summary = {"test": summarize_distribution(df, stage_col="stage", quality_col="quality", day_col=day_col)}
    return splits, summary


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    dataset_type = cfg.get("dataset_type")
    output_dir = Path(args.output_dir) / Path(args.config).stem
    set_global_seed(args.seed, deterministic=True)
    if args.dry_run:
        print(f"Dry run: validated config for {dataset_type}")
        return

    if "split" in cfg and cfg["split"] is not None:
        cfg["split"]["seed"] = args.seed

    if dataset_type == "blastocyst":
        splits, summary = make_blastocyst_splits(cfg)
    elif dataset_type == "humanembryo2":
        splits, summary = make_humanembryo2_splits(cfg)
    elif dataset_type == "quality_public":
        splits, summary = make_quality_public_splits(cfg)
    elif dataset_type == "hungvuong":
        splits, summary = make_hungvuong_splits(cfg)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    save_splits(splits, output_dir=Path(output_dir))

    print(f"Saved splits to {output_dir}")
    for split_name, dist in summary.items():
        print(f"[{split_name}] distribution: {dist}")


if __name__ == "__main__":
    main()
