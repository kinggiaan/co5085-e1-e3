"""Dữ liệu Tiny-ImageNet cho A1 (200 lớp, 64x64).

Cấu trúc sau khi giải nén:
    data/tiny-imagenet-200/train/<wnid>/images/*.JPEG
    data/tiny-imagenet-200/val/<wnid>/images/*.JPEG      (do prepare_tiny_imagenet.py tạo)
    data/tiny-imagenet-200/wnids.txt, words.txt

Ghi chú hiệu năng (số đo thật trên i5-9300H + GTX 1650):
    - đọc PIL từng ảnh  : ~15 ms/ảnh -> 350 s/epoch (20k ảnh) trong khi GPU chỉ cần 37 s
    - nạp sẵn vào RAM   : ~3 ms/ảnh, nhưng mất 180 s mỗi lần khởi động -> cache ra đĩa .pt
    - ColorJitter       : 5,2 ms/ảnh (đắt nhất trong pipeline) -> đã bỏ
    - RandomResizedCrop : 1,7 ms/ảnh với antialias=False
    => augmentation dùng: RandomResizedCrop(0.6-1.0) + lật ngang.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_ROOT = Path("data/tiny-imagenet-200")


def build_transforms(train: bool, img_size: int = 128):
    if train:
        return v2.Compose(
            [
                v2.RandomResizedCrop(img_size, scale=(0.6, 1.0), antialias=False),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return v2.Compose(
        [
            v2.Resize(int(img_size * 256 / 224), antialias=False),
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _list_samples(root: Path, split: str, classes: list[str]) -> list[tuple[Path, int]]:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for pattern in ("*/images/*.JPEG", "*/images/*.jpg", "*/*.JPEG", "*/*.jpg"):
        paths = sorted((root / split).glob(pattern))
        if paths:
            break
    if not paths:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh trong {root / split}. Chạy: python scripts/prepare_tiny_imagenet.py"
        )
    samples = []
    for p in paths:
        cls = p.parent.parent.name if p.parent.name == "images" else p.parent.name
        samples.append((p, class_to_idx[cls]))
    return samples


class TinyImageNetSplit(Dataset):
    """Ảnh Tiny-ImageNet, nạp sẵn vào RAM (có cache .pt) để tránh đọc đĩa mỗi epoch."""

    def __init__(self, root: str | Path = DEFAULT_ROOT, split: str = "train", transform=None,
                 limit: int | None = None, seed: int = 42, cache: bool = True) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        wnids_file = self.root / "wnids.txt"
        if wnids_file.exists():
            self.classes = sorted(wnids_file.read_text().split())
        else:
            self.classes = sorted(p.name for p in (self.root / split).iterdir() if p.is_dir())

        self.samples = _list_samples(self.root, split, self.classes)
        if limit is not None and limit < len(self.samples):
            g = np.random.default_rng(seed)
            idx = np.sort(g.permutation(len(self.samples))[:limit])
            self.samples = [self.samples[i] for i in idx]

        self.images = self._load_or_build(cache)

        self.wnid_to_words = {}
        words = self.root / "words.txt"
        if words.exists():
            for line in words.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    self.wnid_to_words[parts[0]] = parts[1].split(",")[0]

    def _cache_path(self) -> Path:
        full = 10000 if self.split == "val" else 100000
        tag = "full" if len(self.samples) >= full else str(len(self.samples))
        return self.root / f"_cache_{self.split}_{tag}.pt"

    def _load_or_build(self, cache: bool) -> torch.Tensor:
        path = self._cache_path()
        if cache and path.exists():
            return torch.load(path, weights_only=True)  # file chỉ chứa tensor uint8
        n = len(self.samples)
        buf = torch.empty((n, 3, 64, 64), dtype=torch.uint8)
        for i, (p, _) in enumerate(self.samples):
            arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)  # copy -> writable
            buf[i] = torch.from_numpy(arr).permute(2, 0, 1)
        if cache:
            try:
                torch.save(buf, path)
            except OSError:
                pass
        return buf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        label = self.samples[index][1]
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_loaders(
    root: str | Path = DEFAULT_ROOT,
    batch_size: int = 64,
    img_size: int = 128,
    num_workers: int = 0,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, DataLoader]:
    train_ds = TinyImageNetSplit(root, "train", build_transforms(True, img_size), limit=limit, seed=seed)
    val_limit = max((limit // 5) if limit else 0, 200) if limit else None
    val_ds = TinyImageNetSplit(root, "val", build_transforms(False, img_size), limit=val_limit, seed=seed + 1)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True, persistent_workers=num_workers > 0),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          pin_memory=True),
    }


if __name__ == "__main__":
    ds = TinyImageNetSplit()
    x, y = ds[0]
    print("train:", len(ds), "lớp:", len(ds.classes), "| x", tuple(x.shape), "| y", y)
    print("ví dụ:", ds.samples[0][0].name, "->", ds.classes[y], ds.wnid_to_words.get(ds.classes[y], ""))
