# CO5085 — Bài tập E1–E3 (HK261, 2026–2027)

Bài tập coursework môn *Học sâu và ứng dụng trong thị giác máy tính*: phân loại ảnh nhỏ trên
**MNIST** với ba họ mô hình — tuyến tính/softmax, MLP, CNN (E1); Transformer với Multi-Head
Self-Attention tự viết và nhiều cách tokenize (E2); LSTM/GRU trên chuỗi ảnh (E3).

> **Bắt đầu từ đây:** [`GUIDE.md`](GUIDE.md) — lộ trình, kiến thức cần nắm và tiêu chí "xong"
> cho từng bài. README này chỉ tóm tắt cách chạy.

## Cấu trúc

```
common/            # data, training loop viết tay, utils, viz — dùng chung E1–E3
E1/                # softmax / MLP / CNN  (models.py, train.py, run_all.py)
E2/                # MSA tự viết + bản PyTorch, tokenizers, TinyViT, test_msa.py
E3/                # LSTM/GRU trên rows / cols / patches + so sánh baseline E1
scripts/smoke_test.py
docs/index.html    # GitHub Pages
reports/           # báo cáo PDF (E1–E3 trong một file)
```

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Yêu cầu: Python ≥ 3.10, PyTorch ≥ 2.2. Chạy được cả CPU và GPU (đã thử: i5-9300H + GTX 1650 4 GB,
torch 2.5.1+cu121). MNIST được tải tự động vào `data/` (đã nằm trong `.gitignore`).

## Chạy

```powershell
python scripts\smoke_test.py      # kiểm tra môi trường + shape + tốc độ (~40 s)

python -m E1.train --model cnn --epochs 5     # một model
python E1\run_all.py                          # cả 3 model + bảng + biểu đồ
python E2\run_all.py                          # MSA tự viết vs PyTorch, 3 cách tokenize
python E2\test_msa.py                         # chỉ kiểm tra MSA khớp nn.MultiheadAttention
python E3\run_all.py                          # LSTM/GRU x 3 biểu diễn + baseline
```

Mọi kết quả ghi vào `E*/results/<run>/` gồm `metrics.json`, `history.csv`, `curves.png`,
`confusion.png`; bảng tổng hợp ở `E*/results/summary.md` và `summary.json`.

## Thiết lập thí nghiệm (dùng chung)

| Thành phần | Giá trị |
|---|---|
| Dữ liệu | MNIST, 50.000 train / 10.000 val / 10.000 test (split seed 42) |
| Tiền xử lý | chuẩn hoá mean 0.1307, std 0.3081; **không** augmentation |
| Optimizer | Adam, lr 1e-3 (E1), 1e-3 + wd 0.01 (E2–E3), grad-clip 1.0 (E2–E3) |
| Loss | CrossEntropyLoss trên logits |
| Batch | 128 |
| Seed | 42 (E1–E3 dùng chung split để so sánh được với nhau) |

## Kết quả

Sinh bằng các lệnh `run_all.py` ở trên; xem `E1/results/summary.md`, `E2/results/summary.md`,
`E3/results/summary.md`.

**Báo cáo PDF — mỗi bài một file riêng** (chạy `python reports/build_report.py` để sinh lại):

| Bài | File | Trang |
|---|---|---:|
| E1 | [`reports/e1.pdf`](reports/e1.pdf) | 6 |
| E2 | [`reports/e2.pdf`](reports/e2.pdf) | 6 |
| E3 | [`reports/e3.pdf`](reports/e3.pdf) | 5 |
| A1 | [`reports/a1.pdf`](reports/a1.pdf) | 5 |
| A2 | [`reports/a2.pdf`](reports/a2.pdf) | 5 |

**E1 — softmax / MLP / CNN** (8/8/5 epoch, batch 128, seed 42):

| Model | Tham số | Val acc | Test acc | Thời gian |
|---|---:|---:|---:|---:|
| softmax | 7.850 | 91,97% | **92,62%** | 42 s |
| MLP | 235.146 | 97,57% | **97,68%** | 51 s |
| CNN | 468.010 | 99,00% | **99,10%** | 55 s |

**E2 — 3 cách tokenize × 2 bản MSA** (dim 128, depth 2, 4 head, 4 epoch):

| Tokenize | MSA tự viết | MSA PyTorch |
|---|---:|---:|
| patch 4×4 (49 token) | 67,86% | 69,57% |
| mỗi hàng (28 token) | 85,43% | 89,19% |
| CNN stem (49 token) | 82,05% | 90,63% |

MSA tự viết khớp `nn.MultiheadAttention` tới **1,0e-07** khi dùng chung trọng số
(`E2/test_msa.py`, kể cả khi có mask).

**E3 — LSTM/GRU trên chuỗi ảnh** (hidden 128, 4 epoch) so với baseline E1:

| Biểu diễn | LSTM | GRU |
|---|---:|---:|
| mỗi hàng (T=28) | 97,78% | **98,25%** |
| mỗi cột (T=28) | 97,70% | 98,11% |
| patch 4×4 (T=49) | 96,20% | 94,56% |
| baseline | MLP 97,68% | CNN **99,10%** |

Hai biến thể mở rộng: `rows/lstm` **bidirectional** (hidden 64, 49.418 tham số) đạt 97,47%;
`rows/gru` **mean-pooling** đạt 97,73%.

Đảo ngẫu nhiên thứ tự hàng khi test làm `rows/lstm` rơi từ 97,78% xuống **11,22%** — bằng chứng
mô hình thực sự dùng thông tin thứ tự.

**A1 — CNN vs Transformer trên Tiny-ImageNet** (200 lớp, 20.000 ảnh train, 128 px):

| Kiến trúc | Chế độ | Epoch | Trainable | Val acc |
|---|---|---:|---:|---:|
| ResNet-18 | from-scratch | 20 | 11.279.112 | 32,25% |
| ResNet-18 | finetune | 8 | 11.279.112 | 62,68% |
| ResNet-18 | freeze backbone | 6 | **102.600** | 53,35% |
| Swin-T | finetune | 3 | 27.673.154 | **80,17%** |
| Swin-T | freeze một phần | 3 | 14.339.192 | 75,00% |

Pretrain đáng giá **+30 điểm** với CNN; đóng băng backbone vẫn đạt 53% chỉ với 0,9% số tham số
trainable. ViT-B/16 không vừa 4 GB VRAM ở 224 px (đo được 5.048 MB) nên Transformer dùng Swin-T.

**A2 — segmentation trên VOC 2012**: U-Net tự viết (8,88% mIoU) vs DeepLabV3-ResNet50 pretrained
(67,11%) vs SegFormer-B0 (37,85%, đúng paper đang phân tích). Xem `A2/results/summary.md`.

Báo cáo: 5 file PDF riêng (bảng ở mục **Kết quả** phía trên), sinh lại bằng
`python reports/build_report.py`.

## Ghi chú

- Training loop viết tay (`common/engine.py`), không dùng API kiểu `trainer.fit()`.
- MSA của E2 tự viết từ Linear/matmul/softmax; bản `nn.MultiheadAttention` chỉ để đối chiếu và
  được kiểm tra trùng khớp số học trong `E2/test_msa.py`.
- Khai báo việc dùng công cụ AI (nếu có) theo yêu cầu của đề — xem §7 trong `GUIDE.md`.
