# A2 — segmentation trên Pascal VOC 2012

21 lớp, ảnh 256px.

| Mô hình | Pretrain | Tham số | mIoU | Dice | Pixel acc | s/epoch | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| unet | không | 1,927,669 | 8.88% | 12.67% | 75.70% | 116 | 1680 MB |
| deeplabv3 | COCO/VOC | 42,004,074 | 67.11% | 79.18% | 91.96% | 555 | 1601 MB |
| segformer | ImageNet + ADE20K | 3,719,541 | 37.85% | 49.79% | 86.28% | 176 | 443 MB |
