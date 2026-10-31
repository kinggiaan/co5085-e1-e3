# Nhật ký làm việc

Nguyên tắc: chỉ ghi việc **đã thực sự làm**, mỗi dòng gắn được với một commit hoặc một
file/hình kiểm chứng được trong repo. Ngày nào không làm thì ghi là không làm — một nhật ký
có khoảng trống vẫn đáng tin hơn một nhật ký đẹp mà không khớp với `git log`.

---

## Ghi nhận theo lời kể (chưa có artifact trong repo)

> Mục này ghi lại điều nhóm **cho biết đã làm**, không kèm bằng chứng trong repository.
> Người đọc cần đối chiếu với phần nhật ký có kiểm chứng ở dưới.

| Ngày | Nội dung theo lời kể | Trạng thái bằng chứng |
|---|---|---|
| 10/10/2026 | Tải đề, dựng môi trường và bắt đầu làm trên **một máy khác** | Chưa có artifact: repo này chỉ chứa `.venv` tạo 27/10 và commit đầu tiên 27/10. Nếu tìm lại được thư mục cũ, chép sang bằng `robocopy /E /COPY:DAT /DCOPY:T` (giữ nguyên ngày file) rồi bổ sung vào đây. |

---

## 27/10/2026 (tối) — đọc đề, dựng môi trường, xong E1

| Giờ | Việc | Kiểm chứng |
|---|---|---|
| 19:28 | Đọc đề `exercise-vne (1).pdf` (6 trang), chốt phạm vi E1–E3 + hạn nộp | — |
| 19:36 | Tạo `.venv` (Python 3.11), cài `torch 2.5.1+cu121` | thư mục `.venv` tạo 19:36 |
| 20:00 | Kiểm tra CUDA (GTX 1650 4 GB), tải MNIST, đo tốc độ | `scripts/smoke_test.py` |
| 20:10 | `common/data.py`: `PreloadedMNIST`, split 50k/10k/10k seed 42 | commit `93d1004` |
| 20:20 | `common/engine.py`: vòng lặp train viết tay; `common/viz.py` | commit `7a60b22`, `20c243a` |
| 20:30 | `E1/models.py`: softmax / MLP / CNN; `E1/train.py` | commit `ba63149`, `6cb6490` |
| 20:35 | Chạy E1: **softmax 92,62% · MLP 97,68% · CNN 99,10%** | commit `9783f00`, `E1/results/` |

Ghi chú kỹ thuật: `num_workers>0` trên Windows chậm gấp 15 lần với MNIST → nạp sẵn ảnh vào RAM
(45 → 16 ms/batch). Đây là lý do `common/data.py` dùng `PreloadedMNIST`.

## 28–30/10 — không có commit

Không làm gì trên repo (bận việc khác). Không ghi công cho ba ngày này.

## 31/10/2026 — E2, E3, báo cáo

| Giờ | Việc | Kiểm chứng |
|---|---|---|
| 16:46 | Rà lại code cũ, phát hiện bug: `E2/test_msa.py` tắt grad toàn cục làm mọi run sau chết | commit `cb7ec80` |
| 17:00 | Sửa `viz.plot_bar_comparison` (trục thời gian bị lật), thêm chọn head attention | commit `cb7ec80` |
| 17:05 | E2: 6 run (patch / row / cnnstem × MSA tự viết / PyTorch) | commit `6893654` |
| 17:05 | Kiểm tra MSA tự viết khớp `nn.MultiheadAttention` tới 1,0e-07 | `E2/results/msa_equivalence.json` |
| 17:25 | E3: 6 run LSTM/GRU + baseline + thí nghiệm đảo thứ tự hàng | commit `c578fbc` |
| 17:35 | `reports/build_report.py` → `reports/exercises.pdf` (11 trang) | commit `e3e6fd0` |
| 17:45 | Cập nhật README, viết `TODO.md` cho ngày mai | commit `b83b6bc` |

Kết quả: E2 patch 67,9/69,6 · row 85,4/89,2 · cnnstem 82,1/90,6; E3 rows/gru 98,25%,
đảo thứ tự hàng làm rows/lstm rơi 97,78% → 11,22%.

## 31/10/2026 (đêm) → 01/11 (rạng sáng) — A1 + A2: dữ liệu, code, chạy nền

| Giờ | Việc | Kiểm chứng |
|---|---|---|
| 17:55 | Đọc đề A1, A2; chốt dataset A1 = Tiny-ImageNet, A2 = VOC2012 + paper SegFormer | commit `422a6db` |
| 18:00 | Tải Tiny-ImageNet (237 MB), giải nén, sắp 10k ảnh val thành thư mục lớp | `scripts/prepare_tiny_imagenet.py` |
| 18:00 | VOC2012: máy chủ gốc `robots.ox.ac.uk` timeout → chuyển sang mirror HuggingFace, dựng cấu trúc VOC chuẩn (1.464 train / 1.449 val) | `scripts/prepare_voc2012.py` |
| 18:10 | Đo VRAM: ViT-B/16 @224 cần 5.048 MB (> 4 GB → tràn shared memory, 31 s/bước) → đổi Transformer sang **Swin-T @128** (3.243 MB, 0,78 s/bước) | `A1/README.md` |
| 18:20 | Viết `A1/` (data, models, train, run_all) + `A2/` (data, models, metrics, train, run_all) | commit `080fd15` |
| 18:25 | Chạy thử: U-Net 1 epoch; DeepLabV3 pretrained đạt **56,4% mIoU** trên 40 ảnh val (chưa finetune) | `_smoke/` (đã xoá) |
| 18:30 | Trang GitHub Pages: `a1.html`, `a2.html`, trang chung + `style.css` | commit `080fd15` |
| 19:00 | Phát hiện A1 chậm bất thường: 350 s/epoch trong khi GPU chỉ cần 37 s | log bên dưới |
| 19:10 | Đo tách phần: đọc PIL 15 ms/ảnh · `ColorJitter` 5,2 ms/ảnh · `RandomResizedCrop` 1,7 ms/ảnh | script profiling (đã xoá) |
| 19:20 | Sửa `A1/data.py`: nạp RAM + cache `.pt`, bỏ ColorJitter, `antialias=False` | commit sau |
| 19:35 | Sửa lỗi matplotlib: tensor `(3,H,W)` đã chuẩn hoá → phải đảo chuẩn hoá trước khi `imshow` | commit sau |
| 19:53 | Đo lại trong vòng lặp thật: **73,3 s/epoch** (chờ data 27,9 · compute 39,1 · val 6,3) | `_a1_log.txt` |
| 20:15 | Bật `A1/run_all.py --limit 20000` (6 cấu hình, ~3 h) — chạy qua đêm | `A1/results/` |

Ghi chú: job A1 chạy **đêm 31/10 sang 01/11**; kết quả ghi vào `A1/results/` khi từng cấu hình
xong. A2 chạy sau (cùng 4 GB VRAM nên không chạy song song được).

## 01/11/2026 — (điền khi làm xong)

- [ ] Xem `A1/results/summary.md` (job đêm) và chạy `A2/run_all.py`.
- [ ] Điền họ tên + MSSV ở trang bìa báo cáo và trang chủ.
- [ ] Điền mục khai báo công cụ AI (đề mục 1.6).
- [ ] 2 thí nghiệm E3 thêm: `--bidirectional --hidden 64`, `--pooling mean`.
- [ ] Viết nhận xét riêng cho từng chương.
- [ ] Tag `a1-1`, `a2-1` và đẩy lên GitHub.

## 02/11/2026 — nộp E1 + E2

- [ ] Tag `e1e2-final`, push tag trước 23:59 GMT+7.
- [ ] Kiểm tra link Pages mở được PDF.

---

## Phần có hỗ trợ của công cụ AI

Đề (mục 1.6) yêu cầu khai báo rõ. Thực tế đã dùng: một trợ lý AI để đọc đề, dựng khung repo,
viết phần lớn code mẫu, chạy thí nghiệm và soạn bản nháp báo cáo trong hai buổi 27/10 và 31/10.
Nhóm chịu trách nhiệm đọc lại, chạy lại và bảo vệ từng phần khi được hỏi.

<!-- Khi nộp: sửa lại đoạn trên cho đúng phạm vi thực tế, đừng để nguyên nếu không còn đúng. -->
