"""
Adapter for Hung Vuong hospital dataset (external test only).
"""

from pathlib import Path
from typing import List, Mapping, MutableMapping, Optional

import pandas as pd

from ivf.data.datasets import BaseImageDataset
from ivf.data.label_schema import STAGE_TO_ID, StageLabel, map_gardner_to_quality


def _normalize_stage(stage_value) -> Optional[StageLabel]:
    if stage_value is None:
        return None
    text = str(stage_value).strip().lower()
    for stage in StageLabel:
        if stage.value == text:
            return stage
    return None


def load_hungvuong_records(
    root_dir: Path,
    csv_path: Path,
    image_col: str,
    id_col: str,
    stage_col: Optional[str] = None,
    grade_col: Optional[str] = None,
    day_col: Optional[str] = None,
) -> List[MutableMapping]:
    df = pd.read_csv(csv_path)
    records: List[MutableMapping] = []
    for _, row in df.iterrows():
        meta = {
            "id": row.get(id_col),
            "dataset": "hungvuong",
        }
        targets: MutableMapping = {}

        if stage_col and stage_col in df.columns:
            stage = _normalize_stage(row.get(stage_col))
            if stage is not None:
                targets["stage"] = STAGE_TO_ID[stage]
                meta["stage"] = stage.value

        if grade_col and grade_col in df.columns:
            grade = row.get(grade_col)
            quality = map_gardner_to_quality(grade)
            if quality:
                targets["quality"] = quality.value
                meta["quality"] = quality.value
            meta["grade"] = grade

        if day_col and day_col in df.columns and pd.notna(row.get(day_col)):
            meta["day"] = row.get(day_col)

        record: MutableMapping = {
            "image_path": str(Path(root_dir) / str(row.get(image_col))),
            "targets": targets,
            "meta": meta,
        }
        records.append(record)
    return records


def records_to_dataframe(records: List[MutableMapping]) -> pd.DataFrame:
    rows = []
    for rec in records:
        meta = rec.get("meta", {})
        rows.append(
            {
                "image_path": rec.get("image_path"),
                "id": meta.get("id"),
                "stage": meta.get("stage"),
                "quality": meta.get("quality"),
                "grade": meta.get("grade"),
                "day": meta.get("day"),
                "dataset": meta.get("dataset"),
            }
        )
    return pd.DataFrame(rows)


class HungVuongDataset(BaseImageDataset):
    """
    External test dataset. Training/validation splits are forbidden.
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        image_col: str,
        id_col: str,
        transform=None,
        include_meta_day: bool = True,
        stage_col: Optional[str] = None,
        grade_col: Optional[str] = None,
        day_col: Optional[str] = None,
        split: str = "test",
        mode: str = "test",
    ) -> None:
        if split.lower() in {"train", "val", "valid", "validation"} or mode.lower() in {
            "train",
            "val",
            "valid",
            "validation",
        }:
            raise ValueError("Hung Vuong dataset is reserved for external testing only.")

        records = load_hungvuong_records(
            Path(root_dir),
            Path(csv_path),
            image_col=image_col,
            id_col=id_col,
            stage_col=stage_col,
            grade_col=grade_col,
            day_col=day_col,
        )
        super().__init__(records, transform=transform, include_meta_day=include_meta_day)
