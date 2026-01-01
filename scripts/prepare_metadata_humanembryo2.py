"""
Prepare metadata CSV for the HumanEmbryo2.0 dataset.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


STAGE_MAP = {
    "2cell": "cleavage",
    "4cell": "cleavage",
    "8cell": "cleavage",
    "morula": "morula",
    "blastocyst": "blastocyst",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _collect_stage_dirs(root_dir: Path, logger: logging.Logger):
    stage_dirs = []
    for item in root_dir.iterdir():
        if not item.is_dir():
            continue
        key = item.name.lower()
        if key in STAGE_MAP:
            stage_dirs.append(item)
        else:
            logger.warning("Skipping unknown stage directory: %s", item)
    return stage_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare metadata CSV for HumanEmbryo2.0 dataset.")
    parser.add_argument("--raw-dir", default="data/HumanEmbryo2.0", help="Path to raw HumanEmbryo2.0 root.")
    parser.add_argument("--output", default="data/metadata/humanembryo2.csv", help="Output metadata CSV path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("metadata")

    root_dir = Path(args.raw_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {root_dir}")

    records = []
    stage_dirs = _collect_stage_dirs(root_dir, logger)
    if not stage_dirs:
        raise RuntimeError("No stage directories found under HumanEmbryo2.0 root.")

    for stage_dir in stage_dirs:
        stage_label = STAGE_MAP[stage_dir.name.lower()]
        split_dirs = [d for d in stage_dir.iterdir() if d.is_dir()]
        if not split_dirs:
            split_dirs = [stage_dir]

        for split_dir in split_dirs:
            split_name = split_dir.name.lower() if split_dir != stage_dir else ""
            for image_path in split_dir.rglob("*"):
                if image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                rel_path = image_path.relative_to(root_dir).as_posix()
                embryo_id = image_path.stem
                records.append(
                    {
                        "image_path": rel_path,
                        "stage": stage_label,
                        "day": None,
                        "grade": None,
                        "gardner": None,
                        "quality": None,
                        "embryo_id": embryo_id,
                        "patient_id": None,
                        "split": split_name,
                        "dataset": "humanembryo2",
                    }
                )

    if not records:
        raise RuntimeError("No images found under HumanEmbryo2.0 root.")

    df = pd.DataFrame(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s rows to %s", len(df), output_path)
    logger.warning("Day, patient_id, and Gardner grades are not available; columns are left empty.")


if __name__ == "__main__":
    main()
