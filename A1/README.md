# A1 — CNN vs Transformer trên Tiny-ImageNet

Mốc **A1.1** (dataset + protocol) và **A1.2** (so sánh cuối, hạn 09/11).

## Dataset: Tiny-ImageNet

200 lớp, ảnh 64×64, 100.000 train / 10.000 val (nhãn có sẵn), ~237 MB.
Nguồn: [cs231n.stanford.edu](http://cs231n.stanford.edu/tiny-imagenet-200.zip) — dẫn xuất từ
ImageNet, dùng cho nghiên cứu/giáo dục.

```powershell
curl -L -o data\_downloads\tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
python scripts\prepare_tiny_imagenet.py      # giải nén + sắp val thành thư mục lớp
```

**Vì sao chọn (so với CIFAR-100 / Food-101):**

| Tiêu chí | Tiny-ImageNet | CIFAR-100 | Food-101 |
|---|---|---|---|
| Độ phân giải | 64×64 | 32×32 | 512px |
| Lớp / ảnh | 200 / 110k | 100 / 60k | 101 / 101k |
| Pretrained 224px transfer | có nghĩa (phóng 3,5×) | kém tin cậy (phóng 7×) | tốt |
| Vừa 4 GB VRAM | có | có | khó (tải ~5 GB, batch rất nhỏ) |
| Kết luận freeze/pretrain | đáng tin | bị nhiễu bởi domain shift | đáng tin nhưng quá thời gian |

## Kiến trúc

| Họ | Mô hình | Tham số | Ghi chú |
|---|---|---|---|
| CNN | ResNet-18 | ~11,3M | hội tụ nhanh khi from-scratch |
| Transformer | Swin-T | ~27,7M | phân cấp, cửa sổ dịch |

**Vì sao không dùng ViT-B/16:** đo thực tế, ViT-B/16 @224 cần **5.048 MB** (vượt 4 GB → tràn sang
shared memory, chậm ~30×: 31 s/bước so với 0,3 s/bước của ResNet-18). Swin-T @128 cần 3,2 GB.

## Protocol

| # | Mô hình | Chế độ | Trainable | Ghi chú |
|---|---|---|---|---|
| 1 | ResNet-18 | scratch | toàn bộ | mốc CNN không pretrain |
| 2 | Swin-T | scratch | toàn bộ | kiểm chứng "Transformer cần pretrain" |
| 3 | ResNet-18 | finetune | toàn bộ | trần trên CNN pretrained |
| 4 | Swin-T | finetune | toàn bộ | trần trên Transformer pretrained |
| 5 | ResNet-18 | freeze_all | chỉ head | chi phí tối thiểu |
| 6 | Swin-T | freeze_partial | stage cuối + norm + head | cân bằng |

Ảnh **128×128**, batch 64, **50.000 ảnh train** (subset, cùng subset cho cả 6 cấu hình), val 10k.
AdamW + cosine, wd 0,05, label smoothing 0,1, clip 1,0. Epoch: scratch 20, finetune 8, freeze 6–8.

## Chạy

```powershell
python -m A1.train --arch resnet18 --mode scratch --epochs 20   # một cấu hình
python A1\run_all.py                                            # cả 6 + bảng + biểu đồ
python A1\run_all.py --limit 2000 --epochs 1                    # chạy thử nhanh
```

Kết quả vào `A1/results/<arch>_<mode>/` (`metrics.json`, `history.csv`, `curves.png`,
`worst_classes.png`, `errors.png`), bảng tổng hợp ở `A1/results/summary.md`.

## Đo tốc độ thực tế (GTX 1650 4 GB, 128px, bs 64)

| Mô hình | ms/bước | phút/epoch (100k) | peak VRAM |
|---|---:|---:|---:|
| ResNet-18 | 118 | 3,1 | 658 MB |
| Swin-T | 781 | 20,3 | 3.243 MB |

Với 50k ảnh: ResNet-18 ≈ 1,6 phút/epoch, Swin-T ≈ 10 phút/epoch.
