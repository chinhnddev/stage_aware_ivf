"""
Prepare metadata CSV for dataset_test (external evaluation only).

Folder mapping:
  1 -> blastocyst (good)
  2 -> non-blastocyst (poor)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _iter_images(folder: Path):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def _relative_posix(path: Path, root_dir: Path) -> str:
    rel = path.relative_to(root_dir)
    return rel.as_posix()


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare metadata for dataset_test.")
    parser.add_argument("--root_dir", default="data/dataset_test", help="Root directory of dataset_test.")
    parser.add_argument("--output", default="data/metadata/dataset_test.csv", help="Output CSV path.")
    parser.add_argument("--day", type=int, default=5, help="Day value for all samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")

    records = []
    mapping = {
        "1": (1, "good"),
        "2": (0, "poor"),
    }
    counts = {"good": 0, "poor": 0}

    for folder_name, (label, quality) in mapping.items():
        folder = root_dir / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        for path in _iter_images(folder):
            records.append(
                {
                    "image_path": _relative_posix(path, root_dir),
                    "quality_label": label,
                    "quality": quality,
                    "day": args.day,
                    "embryo_id": path.stem,
                    "dataset": "dataset_test",
                }
            )
            counts[quality] += 1

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No images found in dataset_test.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved metadata to {output_path}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
