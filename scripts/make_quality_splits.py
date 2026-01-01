"""
Generate EXP-4 quality splits with derived quality labels and group-safe splitting.

Usage:
    python scripts/make_quality_splits.py --in data/metadata/quality_public.csv --out_dir data/processed --seed 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


MISSINGNESS_THRESHOLD = 0.1


def _coerce_int(value) -> Optional[int]:
    if value is None or pd.isna(value):
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


def _valid_fraction(series: pd.Series) -> float:
    values = series.astype(str).str.strip()
    valid = series.notna() & (values != "") & (values.str.lower() != "nan")
    if len(series) == 0:
        return 0.0
    return float(valid.mean())


def _pick_group_col(df: pd.DataFrame, logger: logging.Logger) -> str:
    if "embryo_id" not in df.columns:
        raise ValueError("No embryo_id column available for grouping.")
    frac = _valid_fraction(df["embryo_id"])
    if frac < 1.0 - MISSINGNESS_THRESHOLD:
        raise ValueError(f"embryo_id missingness {1.0 - frac:.1%} too high for grouping.")
    logger.info("Using embryo_id for grouping (valid %.1f%%).", frac * 100)
    return "embryo_id"


def _derive_quality_label(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    exp_vals = df["exp"].apply(_coerce_int) if "exp" in df.columns else pd.Series([None] * len(df))
    icm_vals = df["icm"].apply(_coerce_int) if "icm" in df.columns else pd.Series([None] * len(df))
    te_vals = df["te"].apply(_coerce_int) if "te" in df.columns else pd.Series([None] * len(df))

    missing_exp = exp_vals.isna().sum()
    missing_icm = icm_vals.isna().sum()
    missing_te = te_vals.isna().sum()

    valid_mask = exp_vals.notna() & icm_vals.notna() & te_vals.notna()
    df = df.copy()
    df["quality_label"] = np.nan
    good_mask = (exp_vals >= 3) & icm_vals.isin([1, 2]) & te_vals.isin([1, 2])
    df.loc[valid_mask, "quality_label"] = np.where(good_mask[valid_mask], 1, 0)
    df.loc[valid_mask, "quality"] = np.where(good_mask[valid_mask], "good", "poor")

    dropped = int((~valid_mask).sum())
    logger.info(
        "Derived quality_label. Dropped missing: exp=%s icm=%s te=%s total=%s",
        missing_exp,
        missing_icm,
        missing_te,
        dropped,
    )
    return df[df["quality_label"].notna()].copy()


def _split_groups(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].astype(str)
    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=groups))
    temp_df = df.iloc[temp_idx]
    temp_groups = temp_df[group_col].astype(str)

    val_fraction = val_ratio / (val_ratio + test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_fraction, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_groups))

    train_df = df.iloc[train_idx].copy()
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    return train_df, val_df, test_df


def _assert_no_group_overlap(splits, group_col: str) -> None:
    groups = [set(df[group_col].astype(str)) for df in splits]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlap = groups[i].intersection(groups[j])
            if overlap:
                raise ValueError(f"Group leakage detected between splits: {sorted(list(overlap))[:3]}")


def _log_label_distribution(df: pd.DataFrame, name: str, logger: logging.Logger) -> None:
    counts = df["quality_label"].value_counts(dropna=False).to_dict()
    logger.info("%s label distribution: %s", name, counts)
    if counts.get(1, 0) == 0:
        logger.warning("%s has zero positive (good) samples.", name)


def parse_args():
    parser = argparse.ArgumentParser(description="Create quality_public splits with derived labels.")
    parser.add_argument("--in", dest="input_csv", required=True, help="Path to quality_public.csv")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory for split CSVs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("quality_splits")

    df = pd.read_csv(args.input_csv)
    if not {"exp", "icm", "te"}.issubset(df.columns):
        raise ValueError("Input CSV must contain exp, icm, te columns.")

    df = _derive_quality_label(df, logger)
    if df.empty:
        raise ValueError("No samples remain after quality_label derivation.")

    group_col = _pick_group_col(df, logger)
    train_df, val_df, test_df = _split_groups(
        df,
        group_col=group_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more splits are empty; adjust ratios or check group distribution.")
    _assert_no_group_overlap([train_df, val_df, test_df], group_col=group_col)

    _log_label_distribution(train_df, "train", logger)
    _log_label_distribution(val_df, "val", logger)
    _log_label_distribution(test_df, "test", logger)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "quality_public_train.csv"
    val_path = out_dir / "quality_public_val.csv"
    test_path = out_dir / "quality_public_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Wrote train split to %s (%s rows)", train_path, len(train_df))
    logger.info("Wrote val split to %s (%s rows)", val_path, len(val_df))
    logger.info("Wrote test split to %s (%s rows)", test_path, len(test_df))


if __name__ == "__main__":
    main()
