"""
Transform helpers with biologically plausible augmentations.
"""

from typing import Iterable, Literal, Optional

try:  # pragma: no cover - optional dependency
    from torchvision import transforms as T
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for transforms. Install torchvision to use data transforms.") from exc


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
    level: Literal["light", "medium", "strong"] = "medium",
    image_size: int = 256,
    normalize: bool = False,
    mean: Optional[Iterable[float]] = None,
    std: Optional[Iterable[float]] = None,
    crop_size: Optional[int] = None,
    rotation_degrees: float = 15.0,
    enable_vertical_flip: bool = False,
):
    if level not in {"light", "medium", "strong"}:
        raise ValueError(f"Unsupported transform level: {level}")

    crop_size = image_size if crop_size is None else crop_size
    if level == "light":
        crop_scale = (0.9, 1.0)
    elif level == "medium":
        crop_scale = (0.8, 1.0)
    else:
        crop_scale = (0.7, 1.0)

    aug = [T.RandomResizedCrop(crop_size, scale=crop_scale)]
    aug.append(T.RandomHorizontalFlip())
    if enable_vertical_flip:
        aug.append(T.RandomVerticalFlip())
    if rotation_degrees and rotation_degrees > 0:
        aug.append(T.RandomRotation(degrees=rotation_degrees))

    ops = _base_transforms(image_size, normalize, mean, std)
    return T.Compose(aug + ops[1:])


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
            T.RandomResizedCrop,
            T.RandomRotation,
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
