"""
Transform helpers with biologically plausible augmentations.
"""

import random
from typing import Iterable, Literal, Optional

try:  # pragma: no cover - optional dependency
    from torchvision import transforms as T
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for transforms. Install torchvision to use data transforms.") from exc


class RandomRotate90:
    def __init__(self, angles=None):
        self.angles = angles or [0, 90, 180, 270]

    def __call__(self, img):
        return img.rotate(random.choice(self.angles))


def _base_transforms(
    image_size: int,
    normalize: bool,
    mean: Optional[Iterable[float]],
    std: Optional[Iterable[float]],
):
    ops = [
        T.Resize((image_size, image_size)),
    ]
    if normalize:
        mean_vals = list(mean) if mean is not None else [0.5, 0.5, 0.5]
        std_vals = list(std) if std is not None else [0.5, 0.5, 0.5]
        ops.append(T.ToTensor())
        ops.append(T.Normalize(mean=mean_vals, std=std_vals))
    else:
        ops.append(T.ToTensor())
    return ops


def get_train_transforms(
    level: Literal["light", "medium"] = "medium",
    image_size: int = 256,
    normalize: bool = False,
    mean: Optional[Iterable[float]] = None,
    std: Optional[Iterable[float]] = None,
):
    if level not in {"light", "medium"}:
        raise ValueError(f"Unsupported transform level: {level}")

    aug = [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomRotate90(),
    ]

    if level == "light":
        aug.append(T.ColorJitter(brightness=0.05, contrast=0.05))
    else:
        aug.append(T.ColorJitter(brightness=0.15, contrast=0.15))

    ops = _base_transforms(image_size, normalize, mean, std)
    return T.Compose(ops[:1] + aug + ops[1:])


def get_eval_transforms(
    image_size: int = 256,
    normalize: bool = False,
    mean: Optional[Iterable[float]] = None,
    std: Optional[Iterable[float]] = None,
):
    return T.Compose(_base_transforms(image_size, normalize, mean, std))


def has_augmentation(transform) -> bool:
    """
    Return True if transform contains any augmentation operations.
    """
    if isinstance(transform, T.Compose):
        return any(has_augmentation(t) for t in transform.transforms)
    if isinstance(
        transform,
        (
            T.RandomHorizontalFlip,
            T.RandomVerticalFlip,
            T.ColorJitter,
            RandomRotate90,
        ),
    ):
        return True
    return False


def assert_no_augmentation(transform) -> None:
    """
    Raise if augmentation is detected in a transform pipeline.
    """
    if has_augmentation(transform):
        raise ValueError("Augmentation is not allowed for evaluation transforms.")
