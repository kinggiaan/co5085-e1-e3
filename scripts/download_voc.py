"""Tải Pascal VOC 2012 (segmentation) về data/VOC2012."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torchvision.datasets import VOCSegmentation

root = Path(__file__).resolve().parents[1] / "data"
print("tải VOC2012 vào", root, flush=True)
ds = VOCSegmentation(root=str(root), year="2012", image_set="train", download=True)
print("xong train:", len(ds), flush=True)
ds = VOCSegmentation(root=str(root), year="2012", image_set="val", download=True)
print("xong val:", len(ds), flush=True)
