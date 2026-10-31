"""Dữ liệu Tiny-ImageNet cho A1 (200 lớp, 64x64).

Cấu trúc sau khi giải nén:
    data/tiny-imagenet-200/train/<wnid>/images/*.JPEG
    data/tiny-imagenet-200/val/<wnid>/images/*.JPEG      (do prepare_tiny_imagenet.py tạo)
    data/tiny-imagenet-200/wnids.txt, words.txt
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_ROOT = Path("data/tiny-imagenet-200")


def build_transforms(train: bool, img_size: int = 224) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TinyImageNetSplit(Dataset):
    """Đọc ảnh theo cấu trúc <split>/<wnid>/images/*.JPEG (hoặc <split>/<wnid>/*.JPEG)."""

    def __init__(self, root: str | Path = DEFAULT_ROOT, split: str = "train", transform=None) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        wnids_file = self.root / "wnids.txt"
        self.classes = sorted(wnids_file.read_text().split()) if wnids_file.exists() else sorted(
            p.name for p in (self.root / split).iterdir() if p.is_dir()
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for pattern in ("*/images/*.JPEG", "*/images/*.jpg", "*/*.JPEG", "*/*.jpg"):
            paths = sorted((self.root / split).glob(pattern))
            if paths:
                break
        if not paths:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh trong {self.root / split}. Chạy: python scripts/prepare_tiny_imagenet.py"
            )
        self.samples = [(p, self.class_to_idx[p.parent.parent.name if p.parent.name == "images" else p.parent.name])
                        for p in paths]

        # từ điển wnid -> tên người đọc được (nếu có)
        self.wnid_to_words = {}
        words = self.root / "words.txt"
        if words.exists():
            for line in words.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    self.wnid_to_words[parts[0]] = parts[1].split(",")[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_loaders(
    root: str | Path = DEFAULT_ROOT,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, DataLoader]:
    train_ds = TinyImageNetSplit(root, "train", build_transforms(True, img_size))
    val_ds = TinyImageNetSplit(root, "val", build_transforms(False, img_size))
    if limit:
        g = torch.Generator().manual_seed(seed)
        train_ds = Subset(train_ds, torch.randperm(len(train_ds), generator=g)[:limit].tolist())
        val_ds = Subset(val_ds, torch.randperm(len(val_ds), generator=g)[: max(limit // 5, 200)].tolist())

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True, persistent_workers=num_workers > 0),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          pin_memory=True),
    }


if __name__ == "__main__":
    ds = TinyImageNetSplit()
    print("train:", len(ds), "lớp:", len(ds.classes))
    print("ví dụ:", ds.samples[0][0].name, "->", ds.classes[ds.samples[0][1]],
          ds.wnid_to_words.get(ds.classes[ds.samples[0][1]], ""))
