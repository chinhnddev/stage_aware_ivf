"""
Adapter for the Kaggle Human blastocyst dataset (morphology learning).
"""

from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional

import pandas as pd

from ivf.data.datasets import BaseImageDataset
from ivf.data.label_schema import gardner_to_morphology_targets, map_gardner_to_quality


def load_blastocyst_records(
    root_dir: Path,
    csv_path: Path,
    image_col: str,
    grade_col: str,
    id_col: str,
    day_col: Optional[str] = None,
) -> List[MutableMapping]:
    df = pd.read_csv(csv_path)
    records: List[MutableMapping] = []
    for _, row in df.iterrows():
        grade = row.get(grade_col)
        try:
            targets = gardner_to_morphology_targets(grade)
        except ValueError:
            continue  # skip unknown/invalid grades

        quality = map_gardner_to_quality(grade)
        meta = {
            "id": row.get(id_col),
            "dataset": "blastocyst_kaggle",
            "grade": grade,
        }
        if quality:
            meta["quality"] = quality.value
        if day_col and day_col in df.columns and pd.notna(row.get(day_col)):
            meta["day"] = row.get(day_col)

        record: MutableMapping = {
            "image_path": str(Path(root_dir) / str(row.get(image_col))),
            "targets": targets,
            "meta": meta,
        }
        records.append(record)
    return records


def records_to_dataframe(records: Iterable[MutableMapping]) -> pd.DataFrame:
    rows = []
    for rec in records:
        meta = rec.get("meta", {})
        rows.append(
            {
                "image_path": rec.get("image_path"),
                "id": meta.get("id"),
                "grade": meta.get("grade"),
                "quality": meta.get("quality"),
                "day": meta.get("day"),
                "dataset": meta.get("dataset"),
            }
        )
    return pd.DataFrame(rows)


class BlastocystKaggleDataset(BaseImageDataset):
    """
    Morphology dataset using Gardner grades for EXP/ICM/TE targets.
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        image_col: str,
        grade_col: str,
        id_col: str,
        transform=None,
        include_meta_day: bool = True,
        day_col: Optional[str] = None,
    ) -> None:
        records = load_blastocyst_records(
            Path(root_dir),
            Path(csv_path),
            image_col=image_col,
            grade_col=grade_col,
            id_col=id_col,
            day_col=day_col,
        )
        super().__init__(records, transform=transform, include_meta_day=include_meta_day)
