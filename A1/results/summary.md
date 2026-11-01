# A1 — CNN vs Transformer trên Tiny-ImageNet

200 lớp, 20,000 ảnh train, val subset, ảnh 128px.

| Kiến trúc | Chế độ | Epoch | Tham số trainable / tổng | Val acc | Macro-F1 | s/epoch | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| resnet18 | scratch | 20 | 11,279,112 / 11,279,112 | 32.25% | 31.01% | 80 | 659 MB |
| resnet18 | finetune | 8 | 11,279,112 / 11,279,112 | 62.68% | 62.44% | 81 | 659 MB |
| resnet18 | freeze_all | 6 | 102,600 / 11,279,112 | 53.35% | 52.65% | 45 | 659 MB |
| swin_t | finetune | 3 | 27,673,154 / 27,673,154 | 80.17% | 80.13% | 348 | 3241 MB |
| swin_t | freeze_partial | 3 | 14,339,192 / 27,673,154 | 75.00% | 75.03% | 181 | 3241 MB |
