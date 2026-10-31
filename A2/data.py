"""Dữ liệu Pascal VOC 2012 cho segmentation (A2).

Đọc cấu trúc chuẩn do scripts/prepare_voc2012.py tạo ra:
    data/VOC2012/JPEGImages/<id>.jpg
    data/VOC2012/SegmentationClass/<id>.png
    data/VOC2012/ImageSets/Segmentation/{train,val}.txt

Nhãn: 0..20 (21 lớp, 0 = nền), 255 = bỏ qua (viền đối tượng).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

VOC_MEAN = (0.485, 0.456, 0.406)
VOC_STD = (0.229, 0.224, 0.225)
IGNORE_INDEX = 255
NUM_CLASSES = 21
CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]
DEFAULT_ROOT = Path("data/VOC2012")


class VOCSegDataset(Dataset):
    def __init__(self, root: str | Path = DEFAULT_ROOT, split: str = "train",
                 img_size: int = 320, augment: bool = False) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        ids = (self.root / "ImageSets" / "Segmentation" / f"{split}.txt").read_text().split()
        self.ids = [i for i in ids if (self.root / "SegmentationClass" / f"{i}.png").exists()]
        if not self.ids:
            raise FileNotFoundError(
                f"Không có dữ liệu trong {self.root}. Chạy: python scripts/prepare_voc2012.py"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        sid = self.ids[index]
        img = Image.open(self.root / "JPEGImages" / f"{sid}.jpg").convert("RGB")
        mask = Image.open(self.root / "SegmentationClass" / f"{sid}.png")

        # resize giữ tỉ lệ rồi pad cho vuông (đơn giản, tránh bóp méo)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = (x - torch.tensor(VOC_MEAN).view(3, 1, 1)) / torch.tensor(VOC_STD).view(3, 1, 1)
        y = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        y = torch.where(y > NUM_CLASSES - 1, torch.full_like(y, IGNORE_INDEX), y)
        return x, y


def build_loaders(
    root: str | Path = DEFAULT_ROOT,
    batch_size: int = 8,
    img_size: int = 320,
    num_workers: int = 0,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, DataLoader]:
    train_ds = VOCSegDataset(root, "train", img_size, augment=True)
    val_ds = VOCSegDataset(root, "val", img_size, augment=False)
    if limit:
        g = torch.Generator().manual_seed(seed)
        train_ds = torch.utils.data.Subset(train_ds, torch.randperm(len(train_ds), generator=g)[:limit].tolist())
        val_ds = torch.utils.data.Subset(val_ds, torch.randperm(len(val_ds), generator=g)[: max(limit // 4, 50)].tolist())
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }


if __name__ == "__main__":
    ds = VOCSegDataset(DEFAULT_ROOT, "train")
    x, y = ds[0]
    print("train:", len(ds), "| x", tuple(x.shape), "| y", tuple(y.shape),
          "| lớp:", sorted(torch.unique(y).tolist())[:8], "...")
