"""Nạp MNIST và chia train/val/test dùng chung cho cả E1–E3.

Chỉ dùng một split duy nhất (seed cố định) để mọi model trong ba bài tập
được so sánh trên cùng dữ liệu.

Ghi chú hiệu năng: MNIST rất nhỏ nên ta nạp sẵn toàn bộ ảnh vào RAM dưới dạng
uint8 (`PreloadedMNIST`) thay vì để torchvision giải mã PIL cho từng ảnh ở mỗi
epoch. Trên laptop i5-9300H + GTX 1650 việc này giảm thời gian mỗi epoch của CNN
từ ~18 s xuống ~8 s. DataLoader vẫn được dùng như bình thường, chỉ dataset là
khác.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
NUM_CLASSES = 10
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]


class PreloadedMNIST(Dataset):
    """MNIST nạp sẵn vào RAM; trả về ảnh đã chuẩn hoá dạng float32 (1, 28, 28).

    Nếu `download=True`, dùng torchvision để tải về `root` (lần đầu ~11 MB).
    """

    def __init__(self, root: str | Path = "data", train: bool = True, download: bool = True) -> None:
        base = datasets.MNIST(str(root), train=train, download=download)
        self.images: torch.Tensor = base.data.unsqueeze(1)  # (N, 1, 28, 28) uint8
        self.targets: torch.Tensor = base.targets
        self.train = train

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, index: int):
        x = self.images[index].float().div_(255.0).sub_(MNIST_MEAN).div_(MNIST_STD)
        return x, int(self.targets[index])


class PlainMNIST(Dataset):
    """Ảnh gốc (chưa chuẩn hoá) — dùng để vẽ mẫu lỗi cho dễ nhìn."""

    def __init__(self, root: str | Path = "data", train: bool = False) -> None:
        base = datasets.MNIST(str(root), train=train, download=True)
        self.images = base.data.unsqueeze(1)
        self.targets = base.targets

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, index: int):
        return self.images[index].float().div_(255.0), int(self.targets[index])


def build_transforms(train: bool = True) -> transforms.Compose:
    """Bản dùng PIL (tham khảo); pipeline chính dùng PreloadedMNIST."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return x * MNIST_STD + MNIST_MEAN


def load_datasets(
    data_dir: str | Path = "data",
    val_size: int = 10_000,
    seed: int = 42,
    train_subset: int | None = None,
) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    """Trả về (train, val, test, test_plain).

    train/val chia từ 60k ảnh train theo seed cố định; test là 10k ảnh test.
    `test_plain` là ảnh chưa chuẩn hoá để vẽ hình.
    """
    train_full = PreloadedMNIST(data_dir, train=True)
    test = PreloadedMNIST(data_dir, train=False)
    test_plain = PlainMNIST(data_dir, train=False)

    n_train = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, val_size], generator=generator)

    if train_subset is not None and train_subset < n_train:
        idx = torch.randperm(n_train, generator=generator)[:train_subset].tolist()
        train_set = Subset(train_set, idx)

    return train_set, val_set, test, test_plain


def make_loader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # 0 là nhanh nhất trên Windows với MNIST
        generator=g if shuffle else None,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    val_size: int = 10_000,
    seed: int = 42,
    num_workers: int = 0,
    train_subset: int | None = None,
) -> dict[str, DataLoader]:
    train_set, val_set, test_set, _ = load_datasets(data_dir, val_size, seed, train_subset)
    return {
        "train": make_loader(train_set, batch_size, shuffle=True, num_workers=num_workers, seed=seed),
        "val": make_loader(val_set, batch_size, shuffle=False, num_workers=num_workers),
        "test": make_loader(test_set, batch_size, shuffle=False, num_workers=num_workers),
    }
