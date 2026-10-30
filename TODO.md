# Việc để lại cho ngày mai (01/11)

Hôm nay (31/10) đã xong: E1, E2, E3 chạy đủ, báo cáo PDF v1, code + kết quả đã commit.
Phần dưới đây là việc **bạn** làm — mỗi mục là một commit, đừng gộp.

## Bắt buộc trước khi nộp (02/11)

- [ ] **Đọc hết `reports/exercises.pdf`** một lượt, ghi lại chỗ nào bạn thấy khó hiểu — đó chính là chỗ giảng viên sẽ hỏi.
- [ ] Trang bìa: điền mã nhóm + họ tên 2 thành viên (`reports/build_report.py`, biến `rows` trong `cover()`), chạy lại `python reports/build_report.py`.
- [ ] §6 "Khai báo sử dụng công cụ AI": điền đúng những gì đã dùng (đề mục 1.6 bắt buộc).
- [ ] Mỗi chương thêm 2–3 câu nhận xét **của bạn** — chỗ nào bất ngờ, chỗ nào khác dự đoán ban đầu.
  Gợi ý: patch tokenize chỉ 68% trong khi cnnstem 90%; đảo thứ tự hàng làm rows/lstm rơi từ 97,8% xuống 11,2%.

## Nên làm (ăn điểm, không tốn nhiều thời gian)

- [ ] Thêm 2 thí nghiệm E3 rồi cập nhật bảng:
  ```powershell
  python -m E3.train --repr rows --cell lstm --bidirectional --hidden 64 --out-dir E3/results/rows_lstm_bidir
  python -m E3.train --repr rows --cell gru  --pooling mean --out-dir E3/results/rows_gru_mean
  ```
  Rồi so sánh: bidirectional có đáng 2× tham số không? mean-pooling vs last-hidden?
- [ ] `docs/worklog.md`: nhật ký theo ngày (10/10 nếu bạn có bằng chứng thật, 27/10–02/11 phần còn lại).

## Đưa lên GitHub (mục 1.4 của đề)

- [ ] Tạo repo trên GitHub, rồi:
  ```powershell
  git remote add origin https://github.com/<username>/co5085-e1-e3.git
  git push -u origin main
  ```
- [ ] Sửa `docs/index.html`: thay `<username>` bằng tài khoản thật.
- [ ] Settings → Pages → Source: `main` / thư mục `/docs`.
- [ ] Kiểm tra link Pages mở được repo + `reports/exercises.pdf`.
- [ ] `git tag e1e2-final` + push tag (đề chấm theo commit trước 23:59 GMT+7).

## Trước khi hết hạn: tự kiểm tra hiểu bài

Trả lời được 6 câu này mà không nhìn code là bạn sẵn sàng vấn đáp:

- [ ] Vì sao `CrossEntropyLoss` nhận logits, không nhận xác suất?
- [ ] BatchNorm khác gì giữa `train()` và `eval()`?
- [ ] Vì sao chia attention cho `√d_k`?
- [ ] Vì sao phải `reshape(B,N,H,D).transpose(1,2)` thay vì `reshape(B,N,H,D)`?
- [ ] LSTM giảm vanishing gradient nhờ đâu?
- [ ] Vì sao CNN vẫn thắng LSTM trên MNIST dù LSTM "hiện đại" hơn?

## Còn lại cho tuần sau (E3, hạn 09/11)

- [ ] Viết chương E3 kỹ hơn: giải thích cổng LSTM/GRU bằng lời của bạn, không chép công thức.
- [ ] Thử thêm biểu diễn chuỗi thứ tư (ví dụ patch 7×7 → 16 token) và so sánh.
- [ ] Cập nhật `reports/exercises.pdf` bản cuối, tag `e3-final`.
