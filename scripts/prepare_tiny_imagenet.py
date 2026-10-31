"""Chuẩn bị Tiny-ImageNet: giải nén và sắp lại val thành thư mục theo lớp.

    python scripts/prepare_tiny_imagenet.py

Kết quả:
    data/tiny-imagenet-200/train/<wnid>/images/*.JPEG   (100.000 ảnh, có sẵn)
    data/tiny-imagenet-200/val/<wnid>/images/*.JPEG     (10.000 ảnh, được tạo ở đây)
    data/tiny-imagenet-200/wnids.txt, words.txt
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ZIP = DATA / "_downloads" / "tiny-imagenet-200.zip"
DEST = DATA / "tiny-imagenet-200"


def extract() -> None:
    if (DEST / "train").exists() and (DEST / "val").exists():
        print("đã giải nén trước đó:", DEST)
        return
    print("giải nén", ZIP, "->", DATA, flush=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(DATA)
    print("giải nén xong", flush=True)


def reorganize_val() -> None:
    """val/images/*.JPEG + val_annotations.txt -> val/<wnid>/images/*.JPEG"""
    val_dir = DEST / "val"
    ann = val_dir / "val_annotations.txt"
    img_dir = val_dir / "images"
    if not ann.exists():
        print("không thấy val_annotations.txt — bỏ qua bước sắp val")
        return
    moved = 0
    for line in ann.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        fname, wnid = parts[0], parts[1]
        target = val_dir / wnid / "images"
        target.mkdir(parents=True, exist_ok=True)
        src = img_dir / fname
        if src.exists():
            shutil.move(str(src), str(target / fname))
            moved += 1
    print(f"đã chuyển {moved} ảnh val vào thư mục lớp")
    if img_dir.exists() and not any(img_dir.iterdir()):
        img_dir.rmdir()


def report() -> None:
    wnids = (DEST / "wnids.txt").read_text().split()
    n_train = sum(1 for _ in (DEST / "train").rglob("*.JPEG"))
    n_val = sum(1 for _ in (DEST / "val").rglob("*.JPEG"))
    print(f"  lớp: {len(wnids)}")
    print(f"  train: {n_train:,} ảnh")
    print(f"  val  : {n_val:,} ảnh")


if __name__ == "__main__":
    if not ZIP.exists():
        print("Thiếu", ZIP, "- tải trước:")
        print("  curl -L -o data\\_downloads\\tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        sys.exit(1)
    extract()
    reorganize_val()
    report()
