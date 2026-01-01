"""
Adapter for the HumanEmbryo2.0 dataset (stage classification).
"""

from pathlib import Path
from typing import List, Mapping, MutableMapping, Optional

import pandas as pd

from ivf.data.datasets import BaseImageDataset
from ivf.data.label_schema import STAGE_TO_ID, StageLabel


def _normalize_stage(stage_value) -> Optional[StageLabel]:
    if stage_value is None:
        return None
    text = str(stage_value).strip().lower()
    for stage in StageLabel:
        if stage.value == text:
            return stage
    return None


def load_humanembryo2_records(
    root_dir: Path,
    csv_path: Path,
    image_col: str,
    stage_col: str,
    id_col: str,
    day_col: Optional[str] = None,
) -> List[MutableMapping]:
    df = pd.read_csv(csv_path)
    records: List[MutableMapping] = []
    for _, row in df.iterrows():
        stage = _normalize_stage(row.get(stage_col))
        if stage is None:
            continue

        meta = {
            "id": row.get(id_col),
            "dataset": "humanembryo2",
            "stage": stage.value,
        }
        if day_col and day_col in df.columns and pd.notna(row.get(day_col)):
            meta["day"] = row.get(day_col)

        record: MutableMapping = {
            "image_path": str(Path(root_dir) / str(row.get(image_col))),
            "targets": {"stage": STAGE_TO_ID[stage]},
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
                "day": meta.get("day"),
                "dataset": meta.get("dataset"),
            }
        )
    return pd.DataFrame(rows)


class HumanEmbryo2Dataset(BaseImageDataset):
    """
    Stage classification dataset.
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        image_col: str,
        stage_col: str,
        id_col: str,
        transform=None,
        include_meta_day: bool = True,
        day_col: Optional[str] = None,
    ) -> None:
        records = load_humanembryo2_records(
            Path(root_dir),
            Path(csv_path),
            image_col=image_col,
            stage_col=stage_col,
            id_col=id_col,
            day_col=day_col,
        )
        super().__init__(records, transform=transform, include_meta_day=include_meta_day)
