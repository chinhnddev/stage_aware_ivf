"""
Prepare metadata CSV for the Hung Vuong hospital dataset.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
QUALITY_MAP = {"0": "poor", "1": "good"}


def _infer_day(filename: str) -> int | None:
    match = re.search(r"[dD](\d+)", filename)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare metadata CSV for Hung Vuong dataset.")
    parser.add_argument("--raw-dir", default="data/hv_hospital_day3_day5", help="Path to raw Hung Vuong root.")
    parser.add_argument("--output", default="data/metadata/hungvuong.csv", help="Output metadata CSV path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("metadata")

    root_dir = Path(args.raw_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {root_dir}")

    records = []
    for split_dir in root_dir.iterdir():
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name.lower()
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue
            label_key = label_dir.name
            quality = QUALITY_MAP.get(label_key)
            if quality is None:
                logger.warning("Skipping unknown label directory: %s", label_dir)
                continue
            for image_path in label_dir.rglob("*"):
                if image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                rel_path = image_path.relative_to(root_dir).as_posix()
                embryo_id = image_path.stem
                day = _infer_day(image_path.name)
                records.append(
                    {
                        "image_path": rel_path,
                        "quality": quality,
                        "quality_raw": label_key,
                        "day": day,
                        "stage": None,
                        "grade": None,
                        "gardner": None,
                        "embryo_id": embryo_id,
                        "patient_id": None,
                        "split": split_name,
                        "dataset": "hungvuong",
                    }
                )

    if not records:
        raise RuntimeError("No images found under Hung Vuong root.")

    df = pd.DataFrame(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s rows to %s", len(df), output_path)
    logger.warning("Assumed folder labels: 0=poor, 1=good. Update mapping if dataset uses a different convention.")
    logger.warning("Patient_id, stage, and Gardner grades are not available; columns are left empty.")


if __name__ == "__main__":
    main()
