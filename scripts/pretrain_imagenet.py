"""
Supervised ImageNet-1K pretraining for JointMorphNet backbone.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from omegaconf import OmegaConf

from ivf.models.encoder import ConvNeXtMini


class ImageNetClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int = 1000) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)


def _get_train_transforms(image_size: int, randaugment: bool) -> T.Compose:
    ops = [
        T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
    ]
    if randaugment and hasattr(T, "RandAugment"):
        ops.append(T.RandAugment())
    elif hasattr(T, "AutoAugment"):
        ops.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))
    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return T.Compose(ops)


def _get_eval_transforms(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize(int(image_size * 256 / 224)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / targets.size(0)))
        return res


def _apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
):
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, targets, targets, 1.0
    use_cutmix = cutmix_alpha > 0 and torch.rand(1).item() < 0.5
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item()) if alpha > 0 else 1.0
    rand_index = torch.randperm(images.size(0), device=images.device)
    target_a = targets
    target_b = targets[rand_index]
    if use_cutmix:
        bbx1, bby1, bbx2, bby2 = _rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
    else:
        images = lam * images + (1 - lam) * images[rand_index]
    return images, target_a, target_b, lam


def _rand_bbox(size, lam):
    w = size[2]
    h = size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = torch.randint(0, w, (1,)).item()
    cy = torch.randint(0, h, (1,)).item()
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, w)
    bby2 = min(cy + cut_h // 2, h)
    return bbx1, bby1, bbx2, bby2


def _build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def _lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain JointMorphNet backbone on ImageNet-1K.")
    parser.add_argument("--config", required=True, help="Pretraining config path.")
    parser.add_argument("--data_root", default=None, help="ImageNet root directory with train/val subfolders.")
    parser.add_argument("--output_dir", default="outputs/pretrain", help="Output directory under outputs/.")
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    data_root = Path(args.data_root or cfg.data.root_dir)
    train_dir = Path(getattr(cfg.data, "train_dir", data_root / "train"))
    val_dir = Path(getattr(cfg.data, "val_dir", data_root / "val"))

    image_size = int(cfg.training.image_size)
    batch_size = int(cfg.training.batch_size)
    epochs = int(cfg.training.epochs)
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.weight_decay)
    warmup_epochs = int(getattr(cfg.training, "warmup_epochs", 5))
    label_smoothing = float(getattr(cfg.training, "label_smoothing", 0.1))
    mixup_alpha = float(getattr(cfg.training, "mixup_alpha", 0.0))
    cutmix_alpha = float(getattr(cfg.training, "cutmix_alpha", 0.0))
    ema_decay = float(getattr(cfg.training, "ema_decay", 0.0))
    randaugment = bool(getattr(cfg.training, "randaugment", True))

    model_cfg = cfg.model
    encoder = ConvNeXtMini(
        in_channels=3,
        dims=list(model_cfg.dims),
        feature_dim=int(model_cfg.feature_dim),
        width_mult=float(getattr(model_cfg, "width_mult", 1.0)),
        depth_mult=float(getattr(model_cfg, "depth_mult", 1.0)),
    )
    model = ImageNetClassifier(encoder, feature_dim=encoder.feature_dim, num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: total={total_params}")

    device = args.device or getattr(cfg, "device", "cuda")
    device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    model.to(device)

    train_tf = _get_train_transforms(image_size, randaugment=randaugment)
    val_tf = _get_eval_transforms(image_size)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(cfg.training.num_workers), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=int(cfg.training.num_workers), pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = _build_scheduler(optimizer, warmup_steps, total_steps)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_acc1 = 0.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            images, target_a, target_b, lam = _apply_mixup_cutmix(images, targets, mixup_alpha, cutmix_alpha)
            logits = model(images)
            if mixup_alpha > 0 or cutmix_alpha > 0:
                loss = (
                    lam * F.cross_entropy(logits, target_a, label_smoothing=label_smoothing)
                    + (1.0 - lam) * F.cross_entropy(logits, target_b, label_smoothing=label_smoothing)
                )
            else:
                loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            running_loss += loss.item()
            global_step += 1
        avg_loss = running_loss / max(1, len(train_loader))

        model.eval()
        state_backup = None
        if ema is not None:
            state_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)
        top1_acc = 0.0
        top5_acc = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                acc1, acc5 = _accuracy(logits, targets, topk=(1, 5))
                top1_acc += acc1.item() * images.size(0)
                top5_acc += acc5.item() * images.size(0)
        top1_acc /= len(val_ds)
        top5_acc /= len(val_ds)
        if state_backup is not None:
            model.load_state_dict(state_backup, strict=False)
        print(f"Epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} val_top1={top1_acc:.2f} val_top5={top5_acc:.2f}")

        if top1_acc > best_acc1:
            best_acc1 = top1_acc
            ckpt_path = output_dir / "imagenet_best.ckpt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "ema_state_dict": ema.shadow if ema is not None else None,
                    "epoch": epoch + 1,
                    "best_acc1": best_acc1,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")

    final_path = output_dir / "imagenet_last.ckpt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "ema_state_dict": ema.shadow if ema is not None else None,
            "epoch": epochs,
            "best_acc1": best_acc1,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
