"""Kiểm tra U-Net có học nổi không: overfit 2 ảnh trong 80 bước.

Nếu mIoU trên chính 2 ảnh đó không lên cao (>0.8) thì lỗi nằm ở kiến trúc/loss,
không phải ở thiếu epoch.
"""
import sys

sys.path.insert(0, r"D:\HCMUT\DL")

import torch
import torch.nn as nn

from A2.data import IGNORE_INDEX, NUM_CLASSES, VOCSegDataset
from A2.metrics import SegMetric, soft_dice_loss
from A2.models import build_model

dev = torch.device("cuda")
ds = VOCSegDataset("data/VOC2012", "train", img_size=192, augment=False)
xs, ys = zip(*[ds[i] for i in (0, 1)])
x = torch.stack(xs).to(dev)
y = torch.stack(ys).to(dev)

model = build_model("unet").to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

for step in range(1, 81):
    model.train()
    opt.zero_grad()
    logits = model(x)
    loss = ce(logits, y) + soft_dice_loss(logits, y)
    loss.backward()
    opt.step()
    if step % 20 == 0:
        model.eval()
        with torch.no_grad():
            m = SegMetric(NUM_CLASSES)
            m.update(model(x).argmax(1), y)
            r = m.compute()
        print(f"buoc {step:3d} | loss {loss.item():.3f} | mIoU {r['miou']:.3f} | "
              f"pixel acc {r['pixel_acc']:.3f}", flush=True)
