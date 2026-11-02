# A1 — CNN vs Transformer trên Tiny-ImageNet

200 lớp, 20,000 ảnh train, val subset, ảnh 128px.

| Kiến trúc | Chế độ | Epoch | Tham số trainable / tổng | Val acc | Macro-F1 | s/epoch | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| resnet18 | freeze_partial | 6 | 8,496,328 / 11,279,112 | 58.73% | 58.32% | 45 | 318 MB |
