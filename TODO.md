# Việc còn lại

Cập nhật 05/11. **E1–E3 xong. A1 xong 6 cấu hình. A2 xong 3 biến thể.**

## Trạng thái từng mốc

| Mốc | Hạn | Trạng thái |
|---|---|---|
| E1 + E2 | 02/11 | xong nội dung, đã tag `e1e2-final` |
| E3 | 09/11 | xong |
| A1.1 | 26/10 | xong (protocol + trang A1) |
| A1.2 | 09/11 | xong — 6 cấu hình, đủ 3 kiểu đóng băng trên ResNet-18 |
| A2.1 | 02/11 | xong (chủ đề + paper + dataset) |
| A2.2 | 16/11 | xong — 3 biến thể + video demo + phân tích paper |

## Còn lại (không bắt buộc)

- [ ] Nếu muốn U-Net đẹp hơn: chạy ≥100 epoch
      (`python -m A2.train --arch unet --epochs 100 --lr 5e-4 --img-size 256`) — hiện 8,88% sau 35
      epoch, kiến trúc đã được kiểm chứng đúng bằng `A2/check_overfit.py`.
- [ ] Đọc lại 5 báo cáo riêng (`reports/e1.pdf`, `e2.pdf`, `e3.pdf`, `a1.pdf`, `a2.pdf`) trước khi nộp.
- [ ] Kiểm tra trang Pages sau khi push: https://kinggiaan.github.io/co5085-e1-e3/

## Ghi chú kỹ thuật

- Dice loss phải lấy trung bình **trên các lớp có mặt trong target**; lấy trung bình cả 21 lớp thì
  lớp vắng mặt luôn cho dice = 1 và làm loãng tín hiệu (đã mắc và sửa).
- `A1/run_all.py` và `A2/run_all.py` đều có resume: chạy lại chỉ tính cấu hình còn thiếu và gộp vào
  `summary.json` thay vì ghi đè.
- Máy throttle còn ~1.150 MHz khi nóng (75 °C) → epoch chậm 2–3×. Job dài nên chạy lúc máy nguội.
- `num_workers > 0` luôn chậm hơn trên Windows với các dataset này — giữ 0 và nạp sẵn RAM.
- Không dùng `Get-Content`/`Set-Content` của PowerShell để sửa file tiếng Việt: đã 3 lần làm hỏng UTF-8.
