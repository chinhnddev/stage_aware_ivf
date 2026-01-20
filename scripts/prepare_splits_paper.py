"""
Prepare blastocyst splits following the paper-style constrained test sampling.

Test set:
  - sample `test_no_c_size` items excluding ICM-C/TE-C (icm==2 or te==2)
  - then force include class C until min counts are met (ICM-C >= min_icm_c, TE-C >= min_te_c)
  - total test size == test_size

Train/val:
  - split remaining data 80/20 with fixed seed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _coerce_int(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _validate_labels(df: pd.DataFrame, exp_col: str, icm_col: str, te_col: str) -> Tuple[pd.DataFrame, dict]:
    exp = df[exp_col].apply(_coerce_int)
    icm = df[icm_col].apply(_coerce_int)
    te = df[te_col].apply(_coerce_int)

    stats = {
        "total": int(len(df)),
        "missing_exp": int(exp.isna().sum()),
        "missing_icm": int(icm.isna().sum()),
        "missing_te": int(te.isna().sum()),
        "invalid_exp": int((~exp.isna() & ~exp.isin(range(5))).sum()),
        "invalid_icm": int((~icm.isna() & ~icm.isin(range(4))).sum()),
        "invalid_te": int((~te.isna() & ~te.isin(range(4))).sum()),
    }
    valid_mask = exp.isin(range(5)) & icm.isin(range(4)) & te.isin(range(4))
    valid_df = df.loc[valid_mask].copy()
    valid_df["__exp"] = exp[valid_mask].astype(int)
    valid_df["__icm"] = icm[valid_mask].astype(int)
    valid_df["__te"] = te[valid_mask].astype(int)
    return valid_df, stats


def _ensure_unique_filenames(df: pd.DataFrame, image_col: str) -> pd.DataFrame:
    df = df.copy()
    df["__file"] = df[image_col].apply(lambda x: Path(str(x)).name)
    dup_mask = df["__file"].duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count:
        print(f"Warning: duplicate filenames detected: {dup_count}. Dropping duplicates.")
        df = df.loc[~dup_mask].copy()
    return df


def _write_split(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    out = df.copy()
    out["split"] = name
    out = out.drop(columns=[col for col in out.columns if col.startswith("__")], errors="ignore")
    out.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare constrained splits for blastocyst dataset.")
    parser.add_argument("--csv_path", default="data/metadata/blastocyst.csv", help="Metadata CSV path.")
    parser.add_argument(
        "--complete_csv",
        default=None,
        help="Optional paper-format complete.csv (no header, columns: filename, exp, icm, te). When set, --csv_path/--*_col are ignored.",
    )
    parser.add_argument("--output_dir", default="data/processed/splits/blastocyst", help="Output directory.")
    parser.add_argument("--image_col", default="image_path", help="Image path column.")
    parser.add_argument("--exp_col", default="exp", help="EXP column (0..4).")
    parser.add_argument("--icm_col", default="icm", help="ICM column (0..3).")
    parser.add_argument("--te_col", default="te", help="TE column (0..3).")
    parser.add_argument("--seed", type=int, default=10123, help="Random seed.")
    parser.add_argument("--test_size", type=int, default=300, help="Total test size.")
    parser.add_argument("--test_no_c_size", type=int, default=268, help="Initial test size excluding class C.")
    parser.add_argument("--min_icm_c", type=int, default=7, help="Minimum ICM-C count in test.")
    parser.add_argument("--min_te_c", type=int, default=25, help="Minimum TE-C count in test.")
    args = parser.parse_args()

    if args.complete_csv:
        complete_path = Path(args.complete_csv)
        if not complete_path.exists():
            raise FileNotFoundError(f"Missing complete.csv: {complete_path}")
        df_raw = pd.read_csv(complete_path, header=None)
        if df_raw.shape[1] < 4:
            raise ValueError(f"complete.csv must have 4 columns: filename, exp, icm, te (got {df_raw.shape[1]}).")
        df_raw = df_raw.iloc[:, :4].copy()
        df_raw.columns = ["filename", "exp", "icm", "te"]
        df_raw["filename"] = df_raw["filename"].astype(str).str.strip()
        df_raw["__file"] = df_raw["filename"]
        df_raw["__exp"] = df_raw["exp"].apply(_coerce_int)
        df_raw["__icm"] = df_raw["icm"].apply(_coerce_int)
        df_raw["__te"] = df_raw["te"].apply(_coerce_int)
        stats = {
            "total": int(len(df_raw)),
            "missing_exp": int(df_raw["__exp"].isna().sum()),
            "missing_icm": int(df_raw["__icm"].isna().sum()),
            "missing_te": int(df_raw["__te"].isna().sum()),
            "invalid_exp": int((~df_raw["__exp"].isna() & ~df_raw["__exp"].isin(range(5))).sum()),
            "invalid_icm": int((~df_raw["__icm"].isna() & ~df_raw["__icm"].isin(range(4))).sum()),
            "invalid_te": int((~df_raw["__te"].isna() & ~df_raw["__te"].isin(range(4))).sum()),
        }
        valid_mask = (
            df_raw["__exp"].isin(range(5))
            & df_raw["__icm"].isin(range(4))
            & df_raw["__te"].isin(range(4))
        )
        df = df_raw.loc[valid_mask].copy()
        df["__exp"] = df["__exp"].astype(int)
        df["__icm"] = df["__icm"].astype(int)
        df["__te"] = df["__te"].astype(int)
        print(f"Loaded {stats['total']} rows from {complete_path}")
    else:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        df_raw = pd.read_csv(csv_path)
        for col in (args.image_col, args.exp_col, args.icm_col, args.te_col):
            if col not in df_raw.columns:
                raise ValueError(f"Missing column '{col}' in {csv_path}")

        df_raw = _ensure_unique_filenames(df_raw, args.image_col)
        df, stats = _validate_labels(df_raw, args.exp_col, args.icm_col, args.te_col)
        print(f"Loaded {stats['total']} rows from {csv_path}")
    print(
        "Label stats: missing_exp=%s missing_icm=%s missing_te=%s invalid_exp=%s invalid_icm=%s invalid_te=%s"
        % (
            stats["missing_exp"],
            stats["missing_icm"],
            stats["missing_te"],
            stats["invalid_exp"],
            stats["invalid_icm"],
            stats["invalid_te"],
        )
    )
    print(f"Valid rows for sampling: {len(df)}")

    rng = np.random.default_rng(args.seed)
    size = len(df)
    if size < args.test_size:
        raise ValueError(f"Not enough samples ({size}) for test_size={args.test_size}")

    test_indices = set()
    exp_count = np.zeros(5, dtype=int)
    icm_count = np.zeros(4, dtype=int)
    te_count = np.zeros(4, dtype=int)

    max_attempts = size * 50
    attempts = 0
    while len(test_indices) < args.test_no_c_size and attempts < max_attempts:
        idx = int(rng.integers(0, size))
        attempts += 1
        if idx in test_indices:
            continue
        icm = int(df["__icm"].iloc[idx])
        te = int(df["__te"].iloc[idx])
        if icm == 2 or te == 2:
            continue
        test_indices.add(idx)
        exp_count[int(df["__exp"].iloc[idx])] += 1
        icm_count[icm] += 1
        te_count[te] += 1

    if len(test_indices) < args.test_no_c_size:
        raise RuntimeError(
            f"Failed to sample {args.test_no_c_size} non-C items; got {len(test_indices)}."
        )

    for idx in rng.permutation(size):
        if len(test_indices) >= args.test_size:
            break
        if idx in test_indices:
            continue
        icm = int(df["__icm"].iloc[idx])
        te = int(df["__te"].iloc[idx])
        add = False
        if icm == 2 and icm_count[2] < args.min_icm_c:
            add = True
        if te == 2 and te_count[2] < args.min_te_c:
            add = True
        if add:
            test_indices.add(idx)
            exp_count[int(df["__exp"].iloc[idx])] += 1
            icm_count[icm] += 1
            te_count[te] += 1
        if len(test_indices) == args.test_size:
            break

    if icm_count[2] < args.min_icm_c or te_count[2] < args.min_te_c:
        raise RuntimeError(
            f"Test constraints not satisfied: ICM-C={icm_count[2]} TE-C={te_count[2]}."
        )

    if len(test_indices) < args.test_size:
        remaining = [idx for idx in range(size) if idx not in test_indices]
        rng.shuffle(remaining)
        for idx in remaining:
            if len(test_indices) >= args.test_size:
                break
            test_indices.add(idx)
            exp_count[int(df["__exp"].iloc[idx])] += 1
            icm_count[int(df["__icm"].iloc[idx])] += 1
            te_count[int(df["__te"].iloc[idx])] += 1

    if len(test_indices) != args.test_size:
        raise RuntimeError(f"Expected {args.test_size} test files, got {len(test_indices)}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = df.iloc[sorted(test_indices)].copy()
    remain_df = df.drop(index=df.index[sorted(test_indices)]).copy()
    train_df, val_df = train_test_split(
        remain_df,
        test_size=0.2,
        shuffle=True,
        random_state=args.seed,
    )

    if args.complete_csv:
        # Paper-style split CSVs (no header; 4 columns: filename, exp, icm, te)
        for name, split_df in (("train", train_df), ("val", val_df)):
            out_path = output_dir / f"{name}.csv"
            with out_path.open("w", encoding="utf-8") as f:
                for _, row in split_df.iterrows():
                    f.write(f"{row['__file']}, {int(row['__exp'])}, {int(row['__icm'])}, {int(row['__te'])}\n")
    else:
        _write_split(train_df, output_dir, "train")
        _write_split(val_df, output_dir, "val")

    testset_path = output_dir / "testset_filenames.csv"
    with testset_path.open("w", encoding="utf-8") as f:
        for _, row in test_df.iterrows():
            f.write(
                f"{row['__file']}, {int(row['__exp'])}, {int(row['__icm'])}, {int(row['__te'])}\n"
            )

    print(f"Saved train/val/test to {output_dir}")
    print(f"Saved testset_filenames.csv to {testset_path}")
    print("Test exp counts:", exp_count)
    print("Test icm counts:", icm_count)
    print("Test te counts:", te_count)


if __name__ == "__main__":
    main()
