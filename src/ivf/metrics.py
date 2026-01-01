"""
Metric builders for IVF tasks.
"""

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassF1Score,
)

from ivf.data.label_schema import EXPANSION_CLASSES, ICM_CLASSES, STAGE_CLASSES, TE_CLASSES


def build_morphology_metrics():
    return {
        "exp_acc": MulticlassAccuracy(num_classes=len(EXPANSION_CLASSES)),
        "icm_acc": MulticlassAccuracy(num_classes=len(ICM_CLASSES)),
        "te_acc": MulticlassAccuracy(num_classes=len(TE_CLASSES)),
    }


def build_stage_metrics():
    return {
        "stage_acc": MulticlassAccuracy(num_classes=len(STAGE_CLASSES)),
        "stage_f1": MulticlassF1Score(num_classes=len(STAGE_CLASSES), average="macro"),
    }


def build_quality_metrics():
    return {
        "quality_acc": BinaryAccuracy(),
        "quality_f1": BinaryF1Score(),
        "quality_auroc": BinaryAUROC(),
        "quality_auprc": BinaryAveragePrecision(),
    }
