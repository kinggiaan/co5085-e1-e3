# A2 — Semantic segmentation trên Pascal VOC 2012

Chủ đề tự chọn + phân tích paper. Mốc **A2.1** (chủ đề + paper + dataset, hạn 02/11) và
**A2.2** (pipeline + phân tích, hạn 16/11).

## Chủ đề

Phân đoạn ngữ nghĩa 21 lớp trên Pascal VOC 2012. So sánh ba biến thể để tách hai yếu tố thường bị
trộn: **kiến trúc** (CNN vs Transformer) và **pretrain** (có vs không):

| Biến thể | Pretrain | Vai trò |
|---|---|---|
| U-Net (tự viết, 1,9M) | không | baseline CNN from-scratch — thể hiện inductive bias |
| DeepLabV3-ResNet50 (42M) | COCO/VOC (21 lớp có sẵn) | CNN mạnh + pretrain |
| SegFormer-B0 (3,7M) | ImageNet-1k → ADE20K (encoder) | đúng kiến trúc của paper đang phân tích |

**Vì sao chọn:** không trùng bài toán phân loại của A1; nối tiếp mạch CNN-vs-Transformer sang bài
toán dự đoán dày; vừa 4 GB VRAM; metric chuẩn tự cài được.

## Paper

> Xie, Wang, Yu, Anandkumar, Alvarez, Luo. **SegFormer: Simple and Efficient Design for Semantic
> Segmentation with Transformers.** NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203) ·
> [code](https://github.com/NVlabs/SegFormer)

Ba ý tưởng: encoder Transformer phân cấp; **bỏ positional encoding** (tránh nội suy khi đổi độ phân
giải test) với Mix-FFN dùng conv 3×3; decoder MLP nhẹ.

**Điểm phê bình dự kiến:** paper đánh giá trên ADE20K/Cityscapes với pretrain ImageNet-1k. VOC 2012
chỉ ~1,5k ảnh — đúng vùng dữ liệu ít mà paper chưa khảo sát. Câu hỏi của nhóm: *trên tập nhỏ,
Transformer còn giữ lợi thế khi cả hai họ đều pretrained?*

## Dataset và metric

| Hạng mục | Giá trị |
|---|---|
| Dataset | Pascal VOC 2012 segmentation: 1.464 train / 1.449 val, 21 lớp |
| Nguồn | mirror HuggingFace `shijli/voc2012` (config `segmentation`) — máy chủ gốc robots.ox.ac.uk không truy cập được |
| Tiền xử lý | resize 320×320, chuẩn hoá ImageNet; augmentation: flip ngang |
| Metric | mIoU (chính), Dice, pixel accuracy, per-class IoU |

```powershell
python scripts\prepare_voc2012.py     # tải + dựng cấu trúc VOC chuẩn vào data/VOC2012
```

## Chạy

```powershell
python -m A2.train --arch unet --epochs 30
python -m A2.train --arch deeplabv3 --epochs 10 --lr 1e-4
python A2\run_all.py                 # cả hai + bảng + biểu đồ
python A2\run_all.py --limit 100 --epochs 1   # chạy thử nhanh
```

## Kết quả (ảnh 256×256, GTX 1650 4 GB)

| Mô hình | Pretrain | Tham số | Epoch | mIoU | Dice | Pixel acc | s/epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepLabV3-ResNet50 | COCO/VOC | 42.004.074 | 5 | **67,11%** | 79,18% | 91,96% | 555 |
| SegFormer-B0 | ImageNet + ADE20K | 3.719.541 | 4 | 37,85% | 49,79% | 86,28% | 176 |
| U-Net | không | 1.927.669 | 35 | 8,88% | 12,67% | 75,70% | 116 |

**Nhận xét:** DeepLabV3 khởi đầu đã ở 50,27% mIoU ngay epoch 1 vì trọng số VOC có sẵn 21 lớp —
phần lớn khoảng cách đến từ **pretrain**, không phải kiến trúc. SegFormer-B0 (3,7M tham số, nhỏ hơn
11×) đạt 37,85% chỉ sau 4 epoch, nhanh hơn 3× mỗi epoch.

**U-Net train từ đầu: 8,88% sau 35 epoch** (CE + Dice, lr 5e-4). Đây là kết quả thật, không phải
lỗi cấu hình — hai bằng chứng:

1. `A2/check_overfit.py` overfit 2 ảnh: mIoU **71,5%**, pixel accuracy 99,4% sau 80 bước → kiến
   trúc và loss học được.
2. VOC 2012 chỉ có 1.464 ảnh train; 35 epoch ≈ 6.400 bước, trong khi U-Net từ đầu trên VOC thường
   cần hàng chục nghìn bước mới hội tụ.

Bài học rút ra (đã ghi trong báo cáo): với ~1.500 ảnh, học từ pixel không thể cạnh tranh với đặc
trưng pretrained — đúng luận điểm cần chứng minh ở A2.

## Cấu trúc

```
A2/data.py      # VOCSegDataset, loaders
A2/models.py    # U-Net (tự viết, 1,9M) + DeepLabV3-ResNet50 (torchvision, pretrained)
A2/metrics.py   # SegMetric: mIoU / Dice / pixel acc từ confusion matrix
A2/train.py     # train loop, lưu metrics + curves + ảnh so sánh mặt nạ
A2/run_all.py   # chạy 2 cấu hình + bảng + biểu đồ
```
