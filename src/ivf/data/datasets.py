"""
Dataset base classes for IVF tasks.

Outputs follow the convention:
    {
        "image": tensor or PIL image,
        "targets": {...},
        "meta": {...},  # may include day for reporting only
    }
"""

from pathlib import Path
from typing import Callable, Iterable, List, Mapping, MutableMapping, Optional

from PIL import Image

try:  # pragma: no cover - optional dependency
    import torch
    from torch.utils.data import Dataset
    from torch.utils.data._utils.collate import default_collate
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore
        def __len__(self) -> int:
            return 0

        def __getitem__(self, index):
            raise ImportError("torch is required for dataset usage.")
    default_collate = None

try:  # pragma: no cover - optional dependency
    from torchvision import transforms as T
except ImportError:  # pragma: no cover
    T = None


class BaseImageDataset(Dataset):
    """
    Base dataset that loads images from disk and attaches targets/meta.

    include_meta_day controls whether 'day' is preserved in meta for reporting.
    The 'day' key is never included in the top-level output.
    """

    def __init__(
        self,
        samples: Iterable[Mapping],
        transform: Optional[Callable] = None,
        include_meta_day: bool = True,
    ) -> None:
        self.samples: List[Mapping] = list(samples)
        self.transform = transform
        self.include_meta_day = include_meta_day

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            return self.transform(image)
        if T:
            return T.ToTensor()(image)
        return image

    def __getitem__(self, index: int) -> MutableMapping:
        sample = self.samples[index]
        image_path = Path(sample["image_path"])
        image = self._load_image(image_path)

        targets = sample.get("targets", {})
        if targets is None:
            targets = {}
        if isinstance(targets, dict):
            targets = dict(targets)
            for key in TARGET_KEYS:
                targets.setdefault(key, IGNORE_INDEX)
        meta = dict(sample.get("meta", {}))
        if not self.include_meta_day:
            meta.pop("day", None)

        return {
            "image": image,
            "targets": targets,
            "meta": meta,
        }


IGNORE_INDEX = -1
TARGET_KEYS = ("exp", "icm", "te", "stage", "quality")


def make_full_target_dict(exp=None, icm=None, te=None, stage=None, quality=None) -> MutableMapping:
    """
    Ensure each sample has a full target dict so batching works across tasks.
    Missing targets use IGNORE_INDEX.
    """
    return {
        "exp": IGNORE_INDEX if exp is None else exp,
        "icm": IGNORE_INDEX if icm is None else icm,
        "te": IGNORE_INDEX if te is None else te,
        "stage": IGNORE_INDEX if stage is None else stage,
        "quality": IGNORE_INDEX if quality is None else quality,
    }


def collate_batch(batch):
    """
    Collate batch without attempting to merge heterogeneous meta dicts.
    """
    if default_collate is None:
        raise ImportError("torch is required for batch collation.")
    images = default_collate([item["image"] for item in batch])
    targets = default_collate([item["targets"] for item in batch])
    meta = [item.get("meta", {}) for item in batch]
    return {"image": images, "targets": targets, "meta": meta}
