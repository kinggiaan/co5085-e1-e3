"""Metric cho segmentation: mIoU, Dice, pixel accuracy, per-class IoU."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

NUM_CLASSES = 21
IGNORE_INDEX = 255


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES,
                   ignore_index: int = IGNORE_INDEX, eps: float = 1.0) -> torch.Tensor:
    """Dice loss (soft) trên toàn batch, bỏ qua pixel nhãn 255.

    Dùng kèm CrossEntropyLoss khi lớp nền chiếm phần lớn pixel (VOC 2012: ~92%). CE một mình
    có cực tiểu tầm thường là "đoán toàn nền"; Dice tính theo phần giao/hợp nên kéo mô hình
    phải phủ đúng cả các lớp hiếm.

    Lưu ý quan trọng: chỉ lấy trung bình trên các lớp **có mặt trong target** của batch.
    Nếu lấy trung bình cả 21 lớp, các lớp vắng mặt luôn cho dice = 1 (tử = mẫu = eps) và làm
    loãng mất tín hiệu — bản đầu tiên mắc đúng lỗi này nên Dice chỉ ~0,09 dù mô hình đoán
    toàn nền, và U-Net không thoát được cực tiểu đó.
    """
    probs = F.softmax(logits, dim=1)
    valid = (target != ignore_index).unsqueeze(1).float()          # (B,1,H,W)

    tgt = target.clone()
    tgt[target == ignore_index] = 0
    onehot = F.one_hot(tgt, num_classes).permute(0, 3, 1, 2).float()  # (B,C,H,W)

    probs = probs * valid
    onehot = onehot * valid
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    card = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (card + eps)

    present = (onehot.sum(dims) > 0).float()                       # lớp có trong target
    loss = 1.0 - (dice * present).sum() / present.sum().clamp(min=1.0)
    return loss


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
