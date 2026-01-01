"""
Split utilities with leakage prevention via group-wise splits.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def split_by_group(
    df: pd.DataFrame,
    group_col: Optional[str],
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Create splits ensuring samples sharing the same group_id stay together.
    """
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0,1).")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be in [0,1).")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    if group_col and group_col in df.columns:
        groups = df[group_col].fillna("__unknown__").astype(str).unique()
    else:
        groups = np.arange(len(df))
        group_col = None

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    test_count = int(n_groups * test_ratio)
    val_count = int(n_groups * val_ratio)

    test_groups = set(groups[:test_count])
    val_groups = set(groups[test_count : test_count + val_count])
    train_groups = set(groups[test_count + val_count :])

    def _mask(groups_subset):
        if group_col is None:
            return df.index.isin(groups_subset)
        return df[group_col].fillna("__unknown__").astype(str).isin(groups_subset)

    splits = {
        "train": df[~(_mask(val_groups) | _mask(test_groups))].copy(),
        "val": df[_mask(val_groups)].copy(),
    }

    if test_ratio > 0:
        splits["test"] = df[_mask(test_groups)].copy()
    return splits


def save_splits(splits: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        split_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)


def summarize_distribution(df: pd.DataFrame, stage_col: Optional[str], quality_col: Optional[str], day_col: Optional[str]) -> Dict[str, dict]:
    summary = {}
    if stage_col and stage_col in df.columns:
        summary["stage"] = df[stage_col].value_counts(dropna=False).to_dict()
    if quality_col and quality_col in df.columns:
        summary["quality"] = df[quality_col].value_counts(dropna=False).to_dict()
    if day_col and day_col in df.columns:
        summary["day"] = df[day_col].value_counts(dropna=False).to_dict()
    return summary
