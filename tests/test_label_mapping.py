import pytest

from ivf.data.label_schema import (
    EXPANSION_TO_ID,
    ICM_TO_ID,
    QualityLabel,
    TE_TO_ID,
    gardner_to_morphology_targets,
    map_gardner_to_quality,
)


@pytest.mark.parametrize(
    "gardner,expected",
    [
        ("4AA", QualityLabel.GOOD),
        ("5AA", QualityLabel.GOOD),
        ("3AB", QualityLabel.GOOD),
        ("4AB", QualityLabel.GOOD),
    ],
)
def test_quality_good_mappings(gardner, expected):
    assert map_gardner_to_quality(gardner) == expected


@pytest.mark.parametrize(
    "gardner,expected",
    [
        ("3CC", QualityLabel.POOR),
        ("2CC", QualityLabel.POOR),
    ],
)
def test_quality_poor_mappings(gardner, expected):
    assert map_gardner_to_quality(gardner) == expected


@pytest.mark.parametrize("gardner", ["4BB", "1AA", "5BB", "ABC", "7AA", "", None])
def test_quality_unknown_or_invalid_returns_none(gardner):
    assert map_gardner_to_quality(gardner) is None


def test_morphology_targets_mapping_is_consistent():
    targets = gardner_to_morphology_targets("4AA")
    assert targets == {
        "exp": EXPANSION_TO_ID[4],
        "icm": ICM_TO_ID["A"],
        "te": TE_TO_ID["A"],
    }


def test_morphology_targets_invalid_grade_raises():
    with pytest.raises(ValueError):
        gardner_to_morphology_targets("9ZZ")
