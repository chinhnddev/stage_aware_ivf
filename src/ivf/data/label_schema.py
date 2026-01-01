"""
Label schema and parsing utilities for IVF embryo tasks.

Single source of truth for:
- StageLabel (cleavage/morula/blastocyst)
- QualityLabel (good/poor)
- Gardner grade parsing and mappings to quality and morphology targets

Hard rules: day metadata must not be used as model input and segmentation masks are disallowed.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class StageLabel(str, Enum):
    CLEAVAGE = "cleavage"
    MORULA = "morula"
    BLASTOCYST = "blastocyst"


class QualityLabel(str, Enum):
    GOOD = "good"
    POOR = "poor"


@dataclass(frozen=True)
class GardnerComponents:
    expansion: int
    icm: str
    te: str


# Expansion classes follow Gardner scale 1–6, IDs are zero-based to suit classifiers.
EXPANSION_CLASSES = [1, 2, 3, 4, 5, 6]
ICM_CLASSES = ["A", "B", "C"]
TE_CLASSES = ["A", "B", "C"]

EXPANSION_TO_ID: Dict[int, int] = {exp: idx for idx, exp in enumerate(EXPANSION_CLASSES)}
ICM_TO_ID: Dict[str, int] = {grade: idx for idx, grade in enumerate(ICM_CLASSES)}
TE_TO_ID: Dict[str, int] = {grade: idx for idx, grade in enumerate(TE_CLASSES)}

STAGE_CLASSES = [StageLabel.CLEAVAGE, StageLabel.MORULA, StageLabel.BLASTOCYST]
STAGE_TO_ID: Dict[StageLabel, int] = {stage: idx for idx, stage in enumerate(STAGE_CLASSES)}

QUALITY_CLASSES = [QualityLabel.POOR, QualityLabel.GOOD]
QUALITY_TO_ID: Dict[QualityLabel, int] = {label: idx for idx, label in enumerate(QUALITY_CLASSES)}

_GARDNER_PATTERN = re.compile(r"^\s*(?P<exp>[1-6])(?P<icm>[ABCabc])(?P<te>[ABCabc])\s*$")


def parse_gardner(gardner: str) -> Optional[GardnerComponents]:
    """
    Parse a Gardner string like '4AA' or '3AB'.

    Returns:
        GardnerComponents if parsable, otherwise None.
    """
    if gardner is None:
        return None

    match = _GARDNER_PATTERN.match(str(gardner))
    if not match:
        return None

    expansion = int(match.group("exp"))
    icm = match.group("icm").upper()
    te = match.group("te").upper()
    return GardnerComponents(expansion, icm, te)


def map_gardner_to_quality(gardner: str) -> Optional[QualityLabel]:
    """
    Map a Gardner grade to a binary quality label.

    Rules:
        good -> 4AA, 5AA, and any 3-4AB (e.g., 3AB, 4AB)
        poor -> <=3CC (e.g., 1CC, 2CC, 3CC)
        others -> None (unknown/ignored)
    """
    components = parse_gardner(gardner)
    if components is None:
        return None

    exp, icm, te = components.expansion, components.icm, components.te

    if (exp in {4, 5} and icm == "A" and te == "A") or (exp in {3, 4} and icm == "A" and te == "B"):
        return QualityLabel.GOOD

    if exp <= 3 and icm == "C" and te == "C":
        return QualityLabel.POOR

    return None


def gardner_to_morphology_targets(gardner: str) -> Dict[str, int]:
    """
    Convert Gardner grade into morphology class IDs for expansion, ICM, and TE.

    Raises:
        ValueError: if the Gardner string cannot be parsed.
    """
    components = parse_gardner(gardner)
    if components is None:
        raise ValueError(f"Cannot parse Gardner string: {gardner!r}")

    return {
        "exp": EXPANSION_TO_ID[components.expansion],
        "icm": ICM_TO_ID[components.icm],
        "te": TE_TO_ID[components.te],
    }
