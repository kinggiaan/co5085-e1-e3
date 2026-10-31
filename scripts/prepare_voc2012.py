"""Tải Pascal VOC 2012 (segmentation) qua HuggingFace và lưu thành cấu trúc VOC chuẩn.

Máy chủ gốc host.robots.ox.ac.uk không truy cập được từ mạng này, nên dùng mirror
`shijli/voc2012` (config `segmentation`: 1.464 train / 1.449 val, 21 lớp).

Kết quả:
    data/VOC2012/JPEGImages/<id>.jpg
    data/VOC2012/SegmentationClass/<id>.png
    data/VOC2012/ImageSets/Segmentation/{train,val}.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "VOC2012"


def main() -> None:
    ds = load_dataset("shijli/voc2012", "segmentation")
    img_dir = DEST / "JPEGImages"
    mask_dir = DEST / "SegmentationClass"
    set_dir = DEST / "ImageSets" / "Segmentation"
    for d in (img_dir, mask_dir, set_dir):
        d.mkdir(parents=True, exist_ok=True)

    for split, out_name in (("train", "train"), ("validation", "val")):
        ids = []
        for row in ds[split]:
            sample_id = str(row["id"])
            img_path = img_dir / f"{sample_id}.jpg"
            mask_path = mask_dir / f"{sample_id}.png"
            if not img_path.exists():
                row["image"].convert("RGB").save(img_path, quality=95)
            if not mask_path.exists():
                row["mask"].save(mask_path)  # giữ nguyên palette/giá trị lớp
            ids.append(sample_id)
        (set_dir / f"{out_name}.txt").write_text("\n".join(ids) + "\n")
        print(f"{split}: {len(ids)} ảnh -> {out_name}.txt", flush=True)

    n_img = len(list(img_dir.glob("*.jpg")))
    n_mask = len(list(mask_dir.glob("*.png")))
    print(f"tổng: {n_img} ảnh, {n_mask} mask trong {DEST}")


if __name__ == "__main__":
    sys.exit(main())
