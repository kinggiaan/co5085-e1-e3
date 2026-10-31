"""Metric cho segmentation: mIoU, Dice, pixel accuracy, per-class IoU."""
from __future__ import annotations

import numpy as np

NUM_CLASSES = 21
IGNORE_INDEX = 255


class SegMetric:
    """Tích luỹ confusion matrix ở cấp pixel rồi tính mIoU / Dice / pixel acc."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.n = num_classes
        self.reset()

    def reset(self) -> None:
        self.conf = np.zeros((self.n, self.n), dtype=np.int64)

    def update(self, preds, labels) -> None:
        p = preds.detach().cpu().numpy().ravel() if hasattr(preds, "detach") else np.asarray(preds).ravel()
        t = labels.detach().cpu().numpy().ravel() if hasattr(labels, "detach") else np.asarray(labels).ravel()
        mask = t != IGNORE_INDEX
        np.add.at(self.conf, (t[mask], p[mask]), 1)

    def compute(self) -> dict:
        inter = np.diag(self.conf).astype(np.float64)
        union = self.conf.sum(0) + self.conf.sum(1) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        dice = np.where((self.conf.sum(0) + self.conf.sum(1)) > 0,
                        2 * inter / np.maximum(self.conf.sum(0) + self.conf.sum(1), 1), np.nan)
        pixel_acc = inter.sum() / max(self.conf.sum(), 1)
        return {
            "miou": float(np.nanmean(iou)),
            "dice": float(np.nanmean(dice)),
            "pixel_acc": float(pixel_acc),
            "per_class_iou": [None if np.isnan(v) else float(v) for v in iou],
            "confusion_matrix": self.conf.tolist(),
        }
