"""
Helpers for resolving external quality labels and logging label provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ivf.data.label_schema import QualityLabel, map_gardner_to_quality


def _normalize_quality_value(value: Any) -> Optional[QualityLabel]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in {0, 1}:
        return QualityLabel.GOOD if int(value) == 1 else QualityLabel.POOR
    text = str(value).strip().lower()
    if text in {"0", "1"}:
        return QualityLabel.GOOD if text == "1" else QualityLabel.POOR
    for label in QualityLabel:
        if label.value == text:
            return label
    return None


def _quality_from_folder(path_value: Optional[str]) -> Optional[QualityLabel]:
    if path_value is None:
        return None
    path = Path(str(path_value).replace("\\", "/"))
    parent = path.parent.name
    if parent == "1":
        return QualityLabel.GOOD
    if parent == "0":
        return QualityLabel.POOR
    return None


def resolve_quality_label(
    row: Any,
    image_col: str,
    quality_col: Optional[str] = None,
    grade_col: Optional[str] = None,
    allow_grade: bool = False,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Resolve quality label for external evaluation.

    Priority:
      1) quality_col (explicit external labels)
      2) grade_col (Gardner) if allow_grade is True
      3) folder name in image_path (0/1)
    """
    if quality_col:
        label = _normalize_quality_value(row.get(quality_col))
        if label is not None:
            return (1 if label == QualityLabel.GOOD else 0), "quality_col"

    if allow_grade and grade_col:
        label = map_gardner_to_quality(row.get(grade_col))
        if label is not None:
            return (1 if label == QualityLabel.GOOD else 0), "grade_col"

    label = _quality_from_folder(row.get(image_col))
    if label is not None:
        return (1 if label == QualityLabel.GOOD else 0), "folder"
    return None, None


def update_label_source_counts(counts: Dict[str, int], source: Optional[str]) -> None:
    if source is None:
        counts["missing"] = counts.get("missing", 0) + 1
    else:
        counts[source] = counts.get(source, 0) + 1
