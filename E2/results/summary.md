# E2 — MSA tự viết vs PyTorch, so sánh tokenization trên MNIST

Tất cả: dim=128, depth=2, heads=4, 4 epoch, 50k ảnh train.

| Tokenize | MSA | Số token | Tham số | Val acc | Test acc | Thời gian (s) | s/epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| patch | manual | 50 | 275,210 | 0.6651 | 0.6786 | 69.6 | 17.4 |
| patch | torch | 50 | 275,210 | 0.6787 | 0.6957 | 71.2 | 17.8 |
| row | manual | 29 | 274,058 | 0.8485 | 0.8543 | 84.3 | 21.1 |
| row | torch | 29 | 274,058 | 0.8848 | 0.8919 | 57.2 | 14.3 |
| cnnstem | manual | 50 | 300,266 | 0.8059 | 0.8205 | 80.9 | 20.2 |
| cnnstem | torch | 50 | 300,266 | 0.8970 | 0.9063 | 84.5 | 21.1 |
