from ivf.eval_label_sources import resolve_quality_label


def test_quality_label_priority_quality_col():
    row = {"quality": "good", "gardner": "4AA", "image_path": "0/img.png"}
    label, source = resolve_quality_label(
        row,
        image_col="image_path",
        quality_col="quality",
        grade_col="gardner",
        allow_grade=True,
    )
    assert label == 1
    assert source == "quality_col"


def test_quality_label_from_grade_when_allowed():
    row = {"gardner": "4AA", "image_path": "0/img.png"}
    label, source = resolve_quality_label(
        row,
        image_col="image_path",
        quality_col="quality",
        grade_col="gardner",
        allow_grade=True,
    )
    assert label == 1
    assert source == "grade_col"


def test_quality_label_from_folder():
    row = {"image_path": "1/img.png"}
    label, source = resolve_quality_label(
        row,
        image_col="image_path",
        quality_col=None,
        grade_col="gardner",
        allow_grade=False,
    )
    assert label == 1
    assert source == "folder"
