# Việc còn lại

Cập nhật 01/11 chiều. Trạng thái: **E1–E3 xong**, **A1 xong**, **A2 xong 2/3 biến thể**.

## Còn lại cho A2.2 (hạn 16/11)

- [ ] **U-Net from-scratch đang sụp về nền** (mIoU ~3,7%, pixel acc 73% = đúng tỉ lệ nền).
      Mask đã kiểm tra đúng (nhãn 0–20 + 255); đã thử lr 1e-3 và 3e-4 trong 2–3 epoch. Hướng sửa:
      1. Trọng số lớp `w = 1/√freq`, chuẩn hoá rồi chặn `[0.2, 10]` — **đừng** chặn sau khi
         chuẩn hoá như bản cũ (bản cũ cho nền trọng số 0,01 nên mô hình loạn).
      2. Hoặc loss `CE + Dice` (Dice chịu mất cân bằng tốt hơn CE).
      3. Chạy ≥30 epoch, lr 3e-4, ảnh 256.
      ```powershell
      python -m A2.train --arch unet --epochs 30 --lr 3e-4 --img-size 256
      ```
- [ ] Cập nhật báo cáo: `python reports/build_report.py` (chương 6 sẽ tự thêm dòng U-Net).
- [ ] (khuyến nghị) video demo ngắn cho A2 — đề mục 7.1.2 có gợi ý.

## Còn lại để nộp

- [ ] Điền **họ tên + MSSV** ở: `reports/build_report.py` (`cover()`), `docs/index.html`,
      `docs/worklog.md`.
- [ ] Đọc lại mục **khai báo công cụ AI** trong PDF, sửa cho đúng phạm vi bạn muốn khai.
- [ ] Đọc lại PDF một lượt, thêm 2–3 câu nhận xét riêng cho mỗi chương.
- [ ] `git push` lần cuối + `git tag -f e1e2-final` (hạn E1+E2: **02/11 23:59**); E3: 09/11.
- [ ] Kiểm tra https://kinggiaan.github.io/co5085-e1-e3/

## Ghi chú kỹ thuật (để không lặp lại)

- Máy throttle còn ~1.150 MHz khi nóng (75 °C) → epoch chậm 2–3×. Job dài nên chạy lúc máy nguội.
- `num_workers > 0` **luôn chậm hơn** trên Windows với các dataset này — giữ 0 và nạp sẵn RAM.
- Không dùng `Get-Content`/`Set-Content` của PowerShell để sửa file tiếng Việt trong repo:
  đã 3 lần làm hỏng UTF-8. Dùng editor hoặc Python với `encoding="utf-8"`.
