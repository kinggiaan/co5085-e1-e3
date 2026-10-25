# Hướng dẫn làm E1–E3 từ đầu — CO5085 (HK261)

Tài liệu này dành cho nhóm muốn **tự viết lại từ số 0** (đề yêu cầu hiểu và bảo vệ được code,
nên đây là cách nên làm). Trong repo đã có sẵn một bản tham chiếu chạy được ở `common/`, `E1/`,
`E2/`, `E3/` — dùng để đối chiếu khi bí, nhưng hãy tự gõ lại theo lộ trình ở §4.

---

## 0. TL;DR — môi trường đã sẵn sàng

```powershell
cd D:\HCMUT\DL
.\.venv\Scripts\activate          # Python 3.11 + torch 2.5.1+cu121
python scripts\smoke_test.py      # kiểm tra toàn bộ: ~50s, in ra bảng tốc độ
```

`smoke_test.py` là **bước số 0 của mọi buổi làm việc**: nó nạp MNIST, chạy thử vài batch cho
cả 7 model của E1–E3, kiểm tra MSA tự viết khớp với `nn.MultiheadAttention`, và in thời gian
mỗi epoch. Nếu lệnh này xanh thì mọi thứ phía sau chỉ còn là vấn đề ý tưởng, không phải môi trường.

**Đã cài:** `torch 2.5.1+cu121` (dùng được GTX 1650), `torchvision`, `numpy`, `matplotlib`,
`reportlab`. MNIST đã tải sẵn trong `data/` (11 MB, nằm trong `.gitignore`).

**Tốc độ thực đo** (GTX 1650 4 GB, batch 128, 50k ảnh train, đã chạy thật):

| Model | Tham số | s/epoch | Cấu hình đã chạy | Test acc |
|---|---:|---:|---|---:|
| E1 softmax (784→10) | 7.850 | 5,2 | 8 epoch | 92,62% |
| E1 MLP (784→256→128→10) | 235.146 | 6,4 | 8 epoch | 97,68% |
| E1 CNN (4 khối conv + BN) | 468.010 | 11,0 | 5 epoch | 99,10% |
| E2 TinyViT patch/4, d=128, depth=2 | 275.210 | ~12 | 4 epoch (ước tính) | — |
| E3 LSTM trên 28 hàng, h=128 | 82.186 | ~7 | 4 epoch (ước tính) | — |

→ Cả ba bài chạy hết **dưới 30 phút GPU**. Đừng ngại thử nhiều cấu hình.

> Lưu ý: `num_workers>0` **chậm hơn** trên Windows với MNIST (mỗi epoch 280 s thay vì 18 s), nên
> `common/data.py` cố ý để `num_workers=0` và nạp sẵn ảnh vào RAM bằng `PreloadedMNIST`.

---

## 1. Bản đồ repo

```
common/            # dùng chung cho cả 3 bài
  data.py          # PreloadedMNIST + split train/val/test cố định (seed 42)
  engine.py        # training loop viết tay: train_one_epoch / evaluate / fit
  utils.py         # set_seed, count_parameters, save_json, Timer, get_device
  viz.py           # đường cong học, confusion matrix, mẫu lỗi, attention
E1/  models.py  train.py  run_all.py
E2/  msa.py  tokenizers.py  models.py  train.py  test_msa.py  run_all.py
E3/  models.py  train.py  run_all.py
scripts/smoke_test.py
docs/index.html    # trang GitHub Pages
```

Đọc theo thứ tự: `common/data.py` → `common/engine.py` → `E1/models.py` → `E1/train.py`
→ `E2/msa.py` → `E2/tokenizers.py` → `E3/models.py`.

**Điểm quan trọng nhất của toàn bộ repo**: `common/engine.py::fit` chính là "vòng lặp huấn luyện
tự viết" mà đề bắt buộc. Chỉ ~20 dòng nhưng là thứ giảng viên sẽ hỏi:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(x)
loss = criterion(logits, y)      # CrossEntropyLoss nhận logits, KHÔNG softmax trước
loss.backward()
optimizer.step()
```

---

## 2. Kiến thức cần nắm

Đánh dấu ✅ khi bạn giải thích được cho người khác mà không nhìn tài liệu.

### 2.1 Nền chung (làm trước, ~1 ngày)

- ✅ Tensor & shape: `(B, C, H, W)` cho ảnh, `(B, N, d)` cho chuỗi token, `(B, 10)` cho logits.
- ✅ Autograd: `requires_grad`, `loss.backward()` cộng dồn gradient vào `.grad`, vì sao phải
  `zero_grad()` mỗi bước, vì sao phải `torch.no_grad()` khi đánh giá.
- ✅ `CrossEntropyLoss` = `log_softmax` + `NLLLoss`; đầu vào là **logits thô**.
- ✅ Gradient của softmax + CE rút gọn thành `p − y_onehot` — hiểu vì sao nó "đẹp".
- ✅ Optimizer: SGD → momentum → Adam (moment bậc 1/2, bias correction), `lr`, `weight_decay`.
- ✅ Overfitting: đọc đường cong train/val; dấu hiệu nào là underfit, dấu hiệu nào là overfit.
- ✅ Metric: accuracy chưa đủ — đọc thêm confusion matrix, precision/recall/F1 per-class.
- ✅ Tái lập: `set_seed`, ghi config, lưu `metrics.json` + `history.csv` (trong repo đã có).

### 2.2 E1 — softmax, MLP, CNN

**Phải hiểu**

- Hồi quy softmax = **một lớp tuyến tính + softmax**; nó tương đương học 10 "template" ảnh,
  dự đoán = mẫu nào khớp nhất. Vì sao nó bão hoà quanh 92–93% trên MNIST.
- MLP: lớp ẩn + phi tuyến (ReLU) tăng capacity; đếm tham số:
  `(784+1)·256 + (256+1)·128 + (128+1)·10 = 235.146`.
- CNN và **inductive bias**: chia sẻ trọng số + locality → học đặc trưng cục bộ (nét, góc, vòng),
  ít tham số hơn MLP cùng độ chính xác.
- Công thức kích thước sau conv: `H_out = floor((H + 2P − K)/S) + 1`.
  Với `K=3, P=1, S=1` thì giữ nguyên; `MaxPool2d(2)` thì chia đôi: 28 → 14 → 7.
- Số tham số một lớp conv: `(C_in·K·K + 1)·C_out`.
- BatchNorm: chuẩn hoá theo batch khi train, dùng running mean/var khi eval →
  **quên `model.eval()` là một trong những lỗi hay gặp nhất**, làm val acc sai lệch.
- Pooling làm tăng receptive field và giảm chiều; vì sao CNN không cần flatten sớm.

**Câu hỏi tự kiểm tra**

1. Vì sao softmax + CE không cần softmax tường minh trong `forward`?
2. Vì sao CNN 468k tham số lại tốt hơn MLP 235k tham số trên cùng dữ liệu?
3. Nếu bỏ toàn bộ BatchNorm thì accuracy và tốc độ hội tụ thay đổi ra sao? (thử đi, 1 dòng sửa)
4. Confusion matrix của bạn sai nhiều nhất ở cặp lớp nào, và **vì sao** (nhìn ảnh mẫu lỗi)?

### 2.3 E2 — Transformer: MSA và tokenization

**Công thức phải viết lại được**

```
Q = X·W_Q,  K = X·W_K,  V = X·W_V              # (B, N, d)
head_h = softmax(Q_h·K_hᵀ / √d_k)·V_h          # d_k = d / H
MSA(X) = concat(head_1..head_H)·W_O
```

- **Vì sao chia cho `√d_k`**: nếu Q,K ~ N(0,1) độc lập thì `q·k` có phương sai `d_k`; chia cho
  `√d_k` đưa phương sai về 1 để softmax không bị bão hoà (gradient ≈ 0).
- Softmax theo **trục key** (chiều cuối), không phải trục query — sai trục là lỗi kinh điển.
- Số tham số MSA: `4d² + 4d` (3 ma trận Q/K/V + 1 W_O, mỗi cái `d×d`).
- **Tokenization** (đề yêu cầu ≥ 2, nên làm 3):
  - `patch`: `unfold(kernel=4, stride=4)` → 49 token, mỗi token 16 chiều → `Linear(16, d)`.
  - `row`: mỗi hàng 28 điểm ảnh là 1 token (28 token) → `Linear(28, d)`.
  - `cnnstem`: 2 lớp conv stride 2 → feature map 7×7×64 → 49 token → `Linear(64, d)`.
  - So sánh: số token (chi phí `O(N²d)`) vs lượng thông tin mỗi token (locality).
- Vì sao Transformer cần **positional embedding** còn CNN thì không.
- `LayerNorm` vs `BatchNorm`: LN chuẩn hoá theo **từng token** (không phụ thuộc batch) →
  ổn định với chuỗi dài/ngắn khác nhau. Đó là lý do Transformer dùng LN.
- Residual + Pre-LN: `x = x + MSA(LN(x))`, `x = x + FFN(LN(x))` — vì sao cần residual (gradient
  chảy thẳng, học được mạng sâu).
- `nn.MultiheadAttention` lưu `in_proj_weight` = `[W_Q; W_K; W_V]` xếp chồng theo chiều 0,
  `in_proj_bias` tương tự, `batch_first=True`, `need_weights=True` khi muốn lấy attention.

**Điểm ăn điểm rõ ràng nhất của E2**: chứng minh MSA tự viết **trùng khớp số học** với bản
PyTorch bằng cách copy trọng số rồi so output. Repo có sẵn `E2/test_msa.py` (sai khác ~0.0e+00).
Hãy tự viết lại phép kiểm tra này — nó chứng minh bạn hiểu, không chỉ chạy được.

**Bẫy**: quên scale; softmax sai trục; `reshape` sai thứ tự head (phải `reshape(B,N,H,D).transpose(1,2)`,
không phải `reshape(B,N,H,D)` rồi coi chiều 2 là head); `pos_embed` lệch số token (kèm CLS);
đánh giá khi model còn `train()` (dropout bật).

**Kỳ vọng hợp lý** (đo thật, 1 epoch, patch/4, d=128, depth=2): test ≈ 60–65%.
Nếu epoch đầu chỉ ~10–20% → attention hoặc positional embedding có vấn đề.
4 epoch nên đạt ~93–96% (bản `torch` nhanh hơn bản tự viết ~20% thời gian).

### 2.4 E3 — LSTM / GRU trên chuỗi ảnh

**Công thức**

```
RNN : h_t = tanh(W_xh·x_t + W_hh·h_{t−1} + b)
LSTM: i,f,o = σ(...),  g = tanh(...),  c_t = f⊙c_{t−1} + i⊙g,  h_t = o⊙tanh(c_t)
GRU : z,r = σ(...),  h̃ = tanh(W_xh·x_t + W_hh·(r⊙h_{t−1})),  h_t = (1−z)⊙h_{t−1} + z⊙h̃
```

- Vì sao LSTM giảm **vanishing gradient**: cell state cộng dồn (`c_t = f⊙c_{t−1} + ...`) tạo
  "đường cao tốc" gradient; cổng `f` học được cách giữ/quên.
- GRU: 2 cổng, ít tham số hơn LSTM (~25%), thường tương đương trên chuỗi ngắn.
- Số tham số LSTM 1 lớp: `4·((input + hidden)·hidden + hidden)`; GRU: `3·(...)`.
  Ví dụ `rows` (input 28, hidden 128): LSTM = 4·((28+128)·128 + 128) = 80.896 (+ head 1.290).
- Ba cách biểu diễn chuỗi: `rows` (T=28, F=28), `cols` (T=28, F=28), `patches` (T=49, F=16).
  Đây là chỗ thể hiện hiểu biết: **thứ tự** nào mang thông tin, và tại sao ảnh không phải chuỗi tự nhiên.
- `batch_first=True`; hidden state `(num_layers, B, H)`; `dropout` của RNN chỉ có tác dụng khi
  `num_layers > 1` (bẫy rất hay gặp).
- Bidirectional: hợp lý vì ta có cả ảnh, không cần causal → nhưng làm tăng tham số gấp đôi.
- Vì sao LSTM **không vượt** CNN trên MNIST: MNIST gần như không có phụ thuộc xa, CNN có
  inductive bias đúng hơn và tính song song tốt hơn. Kết luận này (kèm số liệu) là một nhận xét
  tốt cho báo cáo — đừng cố "làm cho LSTM thắng".

**Thí nghiệm nên có**: huấn luyện `rows/lstm`, sau đó **đảo ngẫu nhiên thứ tự hàng** khi test.
Nếu accuracy tụt mạnh → mô hình thực sự dùng thứ tự, không chỉ cộng các hàng. Repo có sẵn
hàm `evaluate_permuted_rows` trong `E3/run_all.py` để bạn tham khảo cách làm.

**Kỳ vọng** (đo thật, 1 epoch rows/lstm): test ≈ 95%. CNN vẫn cao hơn (~99,1% sau 5 epoch).

---

## 3. Lộ trình tới hạn nộp

Hôm nay **27/10**. E1+E2 hạn **02/11 23:59**, E3 hạn **09/11 23:59**.

| Ngày | Việc | Xong khi |
|---|---|---|
| 27/10 (tối nay) | Chạy `smoke_test.py`; đọc `common/engine.py`, `E1/models.py`; đọc lại đề | Giải thích được training loop từng dòng |
| 28/10 | Tự viết `E1/models.py` (3 model) + `E1/train.py`; chạy 3 model | Bảng accuracy 3 model + 3 hình curves |
| 29/10 | Viết `E2/msa.py` + `test_msa.py` cho khớp tuyệt đối; viết `patch` tokenizer | `test_msa` in sai khác < 1e-4 |
| 30/10 | Viết thêm `row` + `cnnstem`; chạy 6 cấu hình E2 | Bảng so sánh có cột thời gian |
| 31/10 | Viết `E3/models.py` (rows/cols/patches × lstm/gru); chạy | Bảng E3 + baseline MLP/CNN |
| 01/11 | Viết báo cáo PDF (E1+E2), cập nhật GitHub Pages, commit/tag | PDF + link Pages chạy được |
| 02/11 | Buffer + nộp trước 23:59 | Tag `e1e2-final` |
| 03–07/11 | Thí nghiệm thêm cho E3 (bidirectional, mean-pooling, đảo thứ tự) | Có 2–3 nhận xét định lượng |
| 08/11 | Bổ sung chương E3 vào PDF, cập nhật README | PDF v2 |
| 09/11 | Nộp E3 trước 23:59 | Tag `e3-final` |

Nếu chỉ có một người làm: bỏ phần "thí nghiệm thêm", giữ đúng yêu cầu tối thiểu (đề cho phép
làm một mình).

---

## 4. Viết từ scratch: thứ tự và tiêu chí "xong"

1. **`common/data.py`** — load MNIST, chia 50k/10k/10k bằng `random_split` với generator cố định.
   *Xong khi*: hai lần chạy liên tiếp cho cùng chỉ số ảnh đầu tiên (kiểm tra bằng cách in nhãn 5 ảnh).
2. **`common/engine.py`** — `train_one_epoch`, `evaluate`, `fit`. *Xong khi*: loss giảm đơn điệu
   trên 3 epoch với softmax; `evaluate` không bật gradient.
3. **`E1/models.py`** — 3 model, kiểm tra shape bằng một batch giả:
   `assert model(torch.randn(2,1,28,28)).shape == (2,10)`. *Xong khi*: số tham số khớp bảng §0
   (7.850 / 235.146 / 468.010) — lệch nghĩa là cấu trúc khác.
4. **`E1/train.py` + `run_all.py`** — lưu `metrics.json`, `history.csv`, hình. *Xong khi*: 3 model
   đạt xấp xỉ 92,6% / 97,7% / 99,1% test (số của nhóm bạn sẽ lệch ±0,3%).
5. **`E2/msa.py`** — bản tự viết + bản `nn.MultiheadAttention`. *Xong khi*: `test_msa.py` PASS.
6. **`E2/tokenizers.py`** — bắt đầu bằng `patch`, rồi `row`, rồi `cnnstem`.
   *Xong khi*: in ra `(B, N, d)` đúng `N` mong đợi (50, 29, 50).
7. **`E2/train.py`** — dùng lại `fit`. *Xong khi*: 6 cấu hình chạy xong, có bảng so sánh.
8. **`E3/models.py`** — `ImageSequenceClassifier` với 3 encoder. *Xong khi*: 6 cấu hình chạy,
   và bạn giải thích được vì sao `patches` (T=49) chậm hơn `rows` (T=28).
9. **`run_all.py` cho E3** — thêm baseline MLP/CNN (dùng lại kết quả E1 cùng split).
10. **Báo cáo** — §5.

Mẹo debug chuẩn: in shape ở mỗi bước, chạy `--epochs 1` trên 2.000 ảnh trước khi chạy full
(`build_loaders(..., train_subset=2000)` đã hỗ trợ).

---

## 5. Viết báo cáo (một PDF cho cả E1–E3)

Khung đề gợi ý — mỗi bài một chương, dùng chung phần dữ liệu ở đầu:

1. **Giới thiệu & dữ liệu**: MNIST, 50k/10k/10k, chuẩn hoá `mean=0.1307, std=0.3081`,
   không augmentation (để so sánh công bằng), seed 42.
2. **E1**: kiến trúc 3 model (kèm bảng số tham số) → đường cong loss/acc → bảng so sánh →
   confusion matrix của model tốt nhất **và** của softmax (để thấy khác biệt) → 8–16 ảnh sai + nhận xét.
3. **E2**: công thức MSA, code tự viết, **bảng chứng minh trùng khớp với PyTorch** (sai khác ~1e-6),
   3 cách tokenize (bảng: số token, tham số, acc, s/epoch), nhận xét về chi phí `O(N²d)`.
4. **E3**: công thức LSTM/GRU, 3 cách biểu diễn, bảng so với baseline CNN/MLP, thí nghiệm đảo thứ tự.
5. **Kết luận**: mỗi bài 3–5 câu, nêu rõ **cái gì hiệu quả, cái gì không, vì sao** — đừng chỉ
   liệt kê accuracy.
6. **Phụ lục**: cách chạy (`python E1/run_all.py`), phần cứng, thư viện, seed, khai báo công cụ AI.

Chất lượng hình: mỗi hình phải có nhãn trục + đơn vị; confusion matrix ghi số trong ô; đường cong
có cả train và val trên cùng hình. Đề chấm "bảng + biểu đồ + phân tích lỗi rõ ràng" — một hình
đẹp kèm 3 câu nhận xét ăn điểm hơn 10 hình không nhận xét.

---

## 6. Checklist nộp bài

- [ ] `E1/`, `E2/`, `E3/` mỗi thư mục chạy lại được từ đầu (`python E1/run_all.py`).
- [ ] `README.md`: môi trường, lệnh chạy, cấu trúc, kết quả tóm tắt.
- [ ] `reports/exercises.pdf` (một file cho cả ba bài).
- [ ] GitHub Pages trỏ tới repo + PDF (`docs/index.html`).
- [ ] Commit/tag trước 23:59 GMT+7 đúng hạn từng mốc.
- [ ] Bảng khai báo công cụ AI (đề yêu cầu ghi rõ; xem §7).
- [ ] Mỗi thành viên giải thích được phần mình làm.

## 7. Về liêm chính học thuật

Đề nói rõ: công cụ AI chỉ để **hỗ trợ** (tìm tài liệu, gợi ý, ngữ pháp), phải ghi rõ, và không
được nộp sản phẩm do AI tạo phần lớn; phải hiểu và bảo vệ được code + số liệu.

Cách làm an toàn và vẫn nhanh: dùng tài liệu/bản tham chiếu để **hiểu**, rồi tự gõ lại và tự chạy
số liệu của mình. Nếu dùng AI để gợi ý cấu trúc, ghi một dòng trong báo cáo:
"nhóm dùng [công cụ] để gợi ý bố cục code/hình thức trình bày; toàn bộ code và số liệu do nhóm
tự viết và tự chạy". Commit nên phản ánh đúng quá trình thật của nhóm.

## 8. Câu hỏi vấn đáp nên chuẩn bị

1. `optimizer.zero_grad()` để làm gì, nếu bỏ thì sao?
2. Vì sao `CrossEntropyLoss` nhận logits mà không phải xác suất?
3. BatchNorm khác gì `model.eval()`; vì sao quên `eval()` lại làm kết quả val sai?
4. Vì sao chia attention cho `√d_k`?
5. Multi-head khác gì "một head to hơn"? Lợi ích thực tế?
6. Token patch 4×4 vs 7×7: ảnh hưởng gì tới accuracy và thời gian? (trả lời bằng số của bạn)
7. Vì sao Transformer cần positional embedding?
8. LSTM giải quyết vanishing gradient thế nào? GRU khác gì?
9. Vì sao CNN vẫn thắng LSTM trên MNIST?
10. Số tham số của model bạn là bao nhiêu, tính tay được không?
11. Split val/test khác nhau thế nào; vì sao không tune trên test?
12. Nếu accuracy test cao hơn val nhiều thì bạn nghĩ gì?

## 9. Tài nguyên

- Dive into Deep Learning (d2l.ai) — chương Linear/MLP/CNN/Attention/RNN, có code PyTorch.
- CS231n notes: CNN, backprop, training tips.
- "The Illustrated Transformer" (Jay Alammar) — trực quan MSA.
- "The Unreasonable Effectiveness of RNNs" (Karpathy) + `minGPT` để đọc code attention gọn.
- PyTorch docs: `nn.MultiheadAttention`, `nn.LSTM`, `nn.GRU`, `torch.nn.functional.unfold`.
- Bài giảng CO5085 trên LMS (ưu tiên — giảng viên chấm theo đó).
