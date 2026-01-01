"""
Prepare metadata CSV for the public quality dataset derived from Gardner grades.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from ivf.data.label_schema import map_gardner_to_quality


ICM_TE_MAP: Dict[int, str] = {1: "A", 2: "B", 3: "C"}


def _coerce_int(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"ND", "NA"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _build_gardner(exp_val, icm_val, te_val) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    exp = _coerce_int(exp_val)
    icm = _coerce_int(icm_val)
    te = _coerce_int(te_val)
    if exp is None or icm is None or te is None:
        return None, exp, icm, te
    if exp < 1 or exp > 6:
        return None, exp, icm, te
    icm_grade = ICM_TE_MAP.get(icm)
    te_grade = ICM_TE_MAP.get(te)
    if icm_grade is None or te_grade is None:
        return None, exp, icm, te
    return f"{exp}{icm_grade}{te_grade}", exp, icm, te


def _pick_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _load_split(split_csv: Path, split_name: str, images_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(split_csv, sep=";")
    df = df.dropna(axis=1, how="all")

    image_col = _pick_col(df, ["Image", "image", "filename"])
    exp_col = _pick_col(df, ["EXP_silver", "EXP_gold", "EXP"])
    icm_col = _pick_col(df, ["ICM_silver", "ICM_gold", "ICM"])
    te_col = _pick_col(df, ["TE_silver", "TE_gold", "TE"])

    if image_col is None:
        raise ValueError(f"Missing image column in {split_csv}")

    records = []
    missing_quality = 0
    for _, row in df.iterrows():
        filename = row.get(image_col)
        if pd.isna(filename):
            continue
        image_rel = (images_dir / str(filename)).as_posix()

        grade, exp, icm, te = _build_gardner(
            row.get(exp_col) if exp_col else None,
            row.get(icm_col) if icm_col else None,
            row.get(te_col) if te_col else None,
        )
        quality = map_gardner_to_quality(grade) if grade else None
        if quality is None:
            missing_quality += 1

        stem = Path(str(filename)).stem
        embryo_id = stem.split("_")[0] if "_" in stem else stem

        records.append(
            {
                "image_path": image_rel,
                "grade": grade,
                "gardner": grade,
                "quality": quality.value if quality else None,
                "exp": exp,
                "icm": icm,
                "te": te,
                "day": None,
                "embryo_id": embryo_id,
                "patient_id": None,
                "split": split_name,
                "dataset": "quality_public",
            }
        )

    if missing_quality:
        logger.warning("Split %s: %s/%s rows have unknown quality mapping.", split_name, missing_quality, len(df))
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare metadata CSV for quality public dataset.")
    parser.add_argument("--raw-dir", default="data/blastocyst_Dataset", help="Path to raw blastocyst dataset root.")
    parser.add_argument("--output", default="data/metadata/quality_public.csv", help="Output metadata CSV path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("metadata")

    raw_dir = Path(args.raw_dir)
    images_dir = Path("Images")
    if not (raw_dir / images_dir).exists():
        images_dir = Path("images")
    train_csv = raw_dir / "Gardner_train_silver.csv"
    test_csv = raw_dir / "Gardner_test_gold_onlyGardnerScores.csv"
    if not train_csv.exists() and not test_csv.exists():
        raise FileNotFoundError("Missing Gardner CSVs in raw dataset directory.")

    frames = []
    if train_csv.exists():
        frames.append(_load_split(train_csv, "train", images_dir, logger))
    if test_csv.exists():
        frames.append(_load_split(test_csv, "test", images_dir, logger))

    if not frames:
        raise RuntimeError("No records loaded from blastocyst dataset.")

    df = pd.concat(frames, ignore_index=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s rows to %s", len(df), output_path)
    logger.warning("Day and patient_id are not available in this dataset; columns are left empty.")


if __name__ == "__main__":
    main()
