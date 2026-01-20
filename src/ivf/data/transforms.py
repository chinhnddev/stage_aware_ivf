"""
Transform helpers with biologically plausible augmentations.
"""

from typing import Iterable, Literal, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from torchvision import transforms as T
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for transforms. Install torchvision to use data transforms.") from exc


def get_train_transforms(
    level: Literal["light", "medium", "strong"] = "medium",
    image_size: int = 256,
    normalize: bool = False,
    mean: Optional[Iterable[float]] = None,
    std: Optional[Iterable[float]] = None,
    crop_size: Optional[int] = None,
    crop_scale: Optional[Tuple[float, float]] = None,
    crop_ratio: Optional[Tuple[float, float]] = None,
    rotation_degrees: float = 15.0,
    enable_vertical_flip: bool = False,
    translate_max: float = 0.0,
):
    if level not in {"light", "medium", "strong"}:
        raise ValueError(f"Unsupported transform level: {level}")

    crop_size = image_size if crop_size is None else crop_size
    if crop_scale is None:
        crop_scale = (0.8, 1.0)
    if crop_ratio is None:
        crop_ratio = (0.9, 1.1)

    # If crop_scale/ratio are fixed to 1.0, behave like the common "resize-only" baseline
    # used in several reference notebooks (no random cropping).
    if crop_scale == (1.0, 1.0) and crop_ratio == (1.0, 1.0):
        aug = [T.Resize((crop_size, crop_size))]
    else:
        aug = [T.RandomResizedCrop(crop_size, scale=crop_scale, ratio=crop_ratio)]
    aug.append(T.RandomHorizontalFlip(p=0.5))
    if enable_vertical_flip:
        aug.append(T.RandomVerticalFlip(p=0.5))
    if translate_max and translate_max > 0:
        aug.append(T.RandomAffine(degrees=0, translate=(translate_max, translate_max)))
    if rotation_degrees and rotation_degrees > 0:
        aug.append(T.RandomRotation(degrees=rotation_degrees))

    ops = [T.ToTensor()]
    if normalize:
        mean_vals = list(mean) if mean is not None else [0.5, 0.5, 0.5]
        std_vals = list(std) if std is not None else [0.5, 0.5, 0.5]
        ops.append(T.Normalize(mean=mean_vals, std=std_vals))
    return T.Compose(aug + ops)


def get_eval_transforms(
    image_size: int = 256,
    normalize: bool = False,
    mean: Optional[Iterable[float]] = None,
    std: Optional[Iterable[float]] = None,
    crop_size: Optional[int] = None,
):
    crop_size = image_size if crop_size is None else crop_size
    ops = [
        T.Resize(image_size),
        T.CenterCrop(crop_size),
        T.ToTensor(),
    ]
    if normalize:
        mean_vals = list(mean) if mean is not None else [0.5, 0.5, 0.5]
        std_vals = list(std) if std is not None else [0.5, 0.5, 0.5]
        ops.append(T.Normalize(mean=mean_vals, std=std_vals))
    return T.Compose(ops)


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
            T.RandomResizedCrop,
            T.RandomRotation,
            T.RandomAffine,
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
