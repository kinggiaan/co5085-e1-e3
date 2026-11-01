# Nhật ký làm việc

Nguyên tắc: mỗi dòng ứng với một commit trong ngày ghi ở cột Ngày. Cột Kiểm chứng chỉ ghi
file/hình có thật trong repo. Ngày nào không làm thì ghi là không làm.

Bài tập: **E1–E3** (môn CO5085) và **A1–A2** (bài tập lớn). Bắt đầu 10/10/2026.

---

## 10–13/10 — dựng repo và hạ tầng dùng chung

| Ngày | Việc | Kiểm chứng |
|---|---|---|
| 10/10 | Tải và đọc đề `exercise-vne (1).pdf`, chốt làm E1–E3 một mình. Khởi tạo repo, `.gitignore`, `requirements.txt` | `requirements.txt`, `exercise-vne (1).pdf` |
| 11/10 | Viết `common/utils.py`: đặt seed, đếm tham số, lưu metrics, đo thời gian | `common/utils.py` |
| 11/10 | `common/data.py`: `PreloadedMNIST`, chia 50k/10k/10k theo seed 42 | `common/data.py` |
| 12/10 | `common/engine.py`: vòng lặp train tự viết, giữ trọng số tốt nhất theo val acc | `common/engine.py` |
| 13/10 | `common/viz.py`: đường cong học, confusion matrix, ảnh dự đoán sai | `common/viz.py` |

Ghi chú: `num_workers > 0` trên Windows chậm gấp nhiều lần với MNIST nên phải nạp sẵn ảnh
vào RAM (18 → 8 giây/epoch với CNN). Lý do `common/data.py` dùng `PreloadedMNIST`.

## 14/10 — nghỉ, không commit

## 15–17/10 — E1

| Ngày | Việc | Kiểm chứng |
|---|---|---|
| 15/10 | `E1/models.py`: softmax tuyến tính, MLP hai lớp ẩn, CNN ba khối conv + BN | `E1/models.py` |
| 16/10 | `E1/train.py`: train từng mô hình, lưu `metrics.json` + `history.csv` | `E1/train.py` |
| 17/10 | Chạy cả ba mô hình, sinh bảng + biểu đồ so sánh | `E1/results/summary.md` |

Kết quả E1: **softmax 92,62% · MLP 97,68% · CNN 99,10%** test accuracy.

## 18–22/10 — E2

| Ngày | Việc | Kiểm chứng |
|---|---|---|
| 18/10 | `E2/msa.py`: MSA tự viết từ Linear/einsum/softmax, kèm bản `nn.MultiheadAttention` để đối chiếu | `E2/msa.py` |
| 18/10 | `E2/test_msa.py`: so sai khác hai bản, có cả trường hợp mask | `E2/test_msa.py` |
| 19/10 | `E2/tokenizers.py`: patch 4×4, mỗi hàng, CNN stem 7×7 | `E2/tokenizers.py` |
| 20/10 | `E2/models.py`: TinyViT có CLS token + pos-embed, 2 block pre-LN | `E2/models.py` |
| 22/10 | `E2/train.py`, `E2/run_all.py`: 3 tokenizer × 2 bản MSA | `E2/run_all.py` |

## 23–24/10 — E3 và công cụ kiểm tra

| Ngày | Việc | Kiểm chứng |
|---|---|---|
| 23/10 | `E3/models.py`: LSTM/GRU trên chuỗi hàng/cột/patch, pooling last hoặc mean | `E3/models.py` |
| 24/10 | `E3/train.py`, `E3/run_all.py`: 6 cấu hình + baseline MLP/CNN, thêm `evaluate_permuted_rows` | `E3/run_all.py` |
| 24/10 | `scripts/smoke_test.py`: kiểm tra shape, tốc độ, MSA khớp PyTorch | `scripts/smoke_test.py` |
| 24/10 | `docs/index.html`: trang GitHub Pages | `docs/index.html` |

## 25–26/10 — tài liệu và script chạy hàng loạt

| Ngày | Việc | Kiểm chứng |
|---|---|---|
| 25/10 | Viết `README.md` (cách chạy) và `GUIDE.md` (lộ trình, kiến thức, tiêu chí) | `README.md`, `GUIDE.md` |
| 25/10 | Điền số liệu E1 thực đo vào README và bảng tốc độ trong GUIDE | `README.md`, `GUIDE.md` |
| 25/10 | Sửa tham chiếu chéo trong GUIDE | `GUIDE.md` |
| 26/10 | Chuẩn hoá line-ending về LF | `.gitattributes` |
| 26/10 | `run_all.ps1` chạy toàn bộ, thư mục `reports/` | `run_all.ps1` |

## 27–29/10 — chạy E2, E3 lấy số liệu

| Ngày | Việc | Kiểm chứng |
|---|---|---|
| 27/10 | Phát hiện `E2/test_msa.py` tắt grad toàn cục làm mọi run sau chết; bỏ dòng đó, sửa luôn `viz.plot_bar_comparison` (trục thời gian bị lật) và cho `plot_attention` chọn head nhọn nhất | `E2/test_msa.py`, `common/viz.py` |
| 28/10 | Chạy 6 cấu hình E2, commit kết quả + tool vẽ attention | `E2/results/summary.md`, `E2/plot_attention.py` |
| 29/10 | Chạy 6 cấu hình E3, commit kết quả + thí nghiệm đảo thứ tự hàng | `E3/results/summary.md`, `E3/results/order_probe.json` |
| 29/10 | `reports/build_report.py` → `reports/exercises.pdf` 11 trang | `reports/exercises.pdf` |

Số liệu: E2 patch 67,9/69,6 · row 85,4/89,2 · cnnstem 82,1/90,6 (MSA tự viết/PyTorch, test acc);
MSA tự viết khớp `nn.MultiheadAttention` tới 1,0e-07. E3 rows/gru 98,25% cao nhất; đảo thứ tự
hàng làm rows/lstm rơi 97,78% → 11,22%.

## 30–31/10 — chốt E1–E3, mở A1/A2

| Ngày | Việc | Kiểm chứng |
|---|---|---|
| 30/10 | Bảng kết quả E2/E3 vào README, viết `TODO.md` | `README.md`, `TODO.md` |
| 30/10 | Viết `docs/worklog.md` | `docs/worklog.md` |
| 30/10 | Ghi mốc 10/10 vào worklog | `docs/worklog.md` |
| 31/10 | Trỏ link repo/PDF đúng tên thật cho GitHub Pages, xuất bản PDF sang `docs/` | `docs/index.html`, `docs/exercises.pdf` |
| 31/10 | Thêm đề A1, A2 vào repo | `assignment1-vne (1).pdf`, `assignment2-vne.pdf` |
| 31/10 | Build lại PDF (chỉ đổi metadata) | `reports/exercises.pdf` |
| 31/10 | Đọc đề A1/A2; dựng `A1/` (Tiny-ImageNet, ResNet-18 vs Swin-T, 4 chế độ) và `A2/` (VOC2012, U-Net vs DeepLabV3) + trang A1/A2 | `A1/`, `A2/`, `docs/a1.html`, `docs/a2.html` |
| 31/10 | Bỏ `scripts/download_voc.py`, chuyển sang mirror HuggingFace | `scripts/prepare_voc2012.py` |
| 31/10 | A1 chậm bất thường (350 s/epoch, GPU chỉ cần 37 s): đo tách phần, nạp ảnh vào RAM + cache `.pt`, bỏ `ColorJitter`, `antialias=False` → **73,3 s/epoch**. Sửa luôn lỗi `imshow` với tensor đã chuẩn hoá | `A1/data.py`, `A1/train.py`, `_a1_log.txt` |

Đo VRAM trước khi chạy: ViT-B/16 @224 cần 5.048 MB (> 4 GB, tràn shared memory, 31 s/bước) nên
đổi Transformer sang **Swin-T @128** (3.243 MB, 0,78 s/bước).

## 01/11 — xong A1, A2 và chốt báo cáo

| Ngày | Việc | Kiểm chứng |
|---|---|---|
| 01/11 | Thêm chương A1, A2 vào báo cáo; `A1/run_all.py` có resume, bỏ `swin_t/scratch` (đo lại 1.663 s/epoch, 9 h cho 20 epoch, vượt ngân sách) | `reports/build_report.py`, `A1/run_all.py` |
| 01/11 | Thêm SegFormer-B0 theo paper: hạ `transformers<5`, cache HF vào `.hf/` | `A2/segformer.py`, `requirements.txt` |
| 01/11 | A1 xong 5 cấu hình; E3 chạy thêm 2 biến thể (bidirectional, GRU mean-pooling) | `A1/results/summary.md`, `E3/results/summary.md` |
| 01/11 | A2: U-Net sụp về nền với CE thường (mIoU 4%, pixel acc 73% = tỉ lệ nền) → thêm trọng số lớp nghịch tần suất, chặn 20×, có `--no-class-weight` để tắt | `A2/train.py`, `A2/run_all.py` |
| 01/11 | Bảng kết quả A1 vào README và trang A1 | `README.md`, `docs/a1.html` |
| 01/11 | A2: lưu metrics/history **sau mỗi epoch** — DeepLab đạt 67,11% nhưng mất file vì tiến trình kết thúc trước khi ghi | `A2/train.py`, `A2/run_all.py` |
| 01/11 | A2 xong: DeepLabV3 **67,11% mIoU** (5 epoch), SegFormer **37,85%** (4 epoch) trên VOC2012 256px; báo cáo lên 16 trang | `A2/results/summary.md`, `reports/exercises.pdf` |
| 01/11 | Worklog cập nhật | `docs/worklog.md` |
| 01/11 | Điền tên + MSSV (Dương Gia An — 2470293) vào bìa báo cáo, trang chủ, worklog | `reports/build_report.py`, `docs/index.html` |
| 01/11 | Viết lại mục khai báo AI (mục 1.6) theo giọng cá nhân | `reports/build_report.py` |
| 01/11 | Sửa mục khai báo AI cho đúng mức độ hiểu code, bỏ câu khẳng định quá | `reports/build_report.py` |
| 01/11 | Gọn lại mục khai báo AI, chỉ nêu phạm vi hỗ trợ | `reports/exercises.pdf` |

Kết quả A1: resnet scratch **32,25%** · finetune **62,68%** · freeze_all **53,35%** ·
swin finetune **80,17%** · freeze_partial **75,00%**. E3 thêm: bidirectional 97,47%, GRU
mean-pooling 97,73%.

Việc còn lại xem `TODO.md`.

---

## Khai báo sử dụng công cụ AI (đề mục 1.6)

Theo mục 1.6 của đề: tôi dùng một trợ lý AI để đọc và tóm tắt đề, gợi ý bố cục thư mục và API
PyTorch phù hợp, soạn bản nháp ban đầu cho phần lớn mã nguồn trong `common/`, `E1/`–`E3/`,
`A1/`, `A2/`, gợi ý dàn ý báo cáo và cách chẩn đoán một số lỗi shape/lỗi huấn luyện.

Phần tôi trực tiếp làm: cài môi trường, tải toàn bộ dữ liệu, chạy mọi thí nghiệm trên máy của
mình, đối chiếu từng bảng số và từng hình trong báo cáo với `metrics.json` do các script sinh
ra. Không có con số nào trong báo cáo được nhập tay.
