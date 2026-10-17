# E1 — So sánh softmax / MLP / CNN trên MNIST

Số epoch: softmax=8, mlp=8, cnn=5

| Model | Tham số | Val acc (best) | Test acc | Test loss | Thời gian (s) | s/epoch |
|---|---:|---:|---:|---:|---:|---:|
| softmax | 7,850 | 0.9197 | 0.9262 | 0.2680 | 41.6 | 5.2 |
| mlp | 235,146 | 0.9757 | 0.9768 | 0.0778 | 51.1 | 6.4 |
| cnn | 468,010 | 0.9900 | 0.9910 | 0.0293 | 54.8 | 11.0 |
