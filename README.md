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
`E3/results/summary.md`. Báo cáo đầy đủ: `reports/exercises.pdf`.

**E1 — đã chạy xong** (8/8/5 epoch, batch 128, seed 42):

| Model | Tham số | Val acc | Test acc | Thời gian |
|---|---:|---:|---:|---:|
| softmax | 7.850 | 91,97% | **92,62%** | 42 s |
| MLP | 235.146 | 97,57% | **97,68%** | 51 s |
| CNN | 468.010 | 99,00% | **99,10%** | 55 s |

E2/E3 chưa chạy full (mới kiểm tra 1 epoch qua `scripts/smoke_test.py`) — chạy
`python E2\run_all.py` và `python E3\run_all.py` để sinh số liệu.

## Ghi chú

- Training loop viết tay (`common/engine.py`), không dùng API kiểu `trainer.fit()`.
- MSA của E2 tự viết từ Linear/matmul/softmax; bản `nn.MultiheadAttention` chỉ để đối chiếu và
  được kiểm tra trùng khớp số học trong `E2/test_msa.py`.
- Khai báo việc dùng công cụ AI (nếu có) theo yêu cầu của đề — xem §7 trong `GUIDE.md`.
