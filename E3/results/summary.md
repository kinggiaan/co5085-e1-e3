# E3 — LSTM/GRU trên chuỗi ảnh so với baseline MLP/CNN (MNIST)

hidden=128, 1 lớp, 4 epoch. Baseline lấy từ E1 (cùng split).

| Biểu diễn | Cell | Tham số | Val acc | Test acc | Thời gian (s) | s/epoch |
|---|---|---:|---:|---:|---:|---:|
| rows | lstm | 82,186 | 0.9755 | 0.9778 | 42.5 | 10.6 |
| rows | gru | 61,962 | 0.9798 | 0.9825 | 45.0 | 11.3 |
| cols | lstm | 82,186 | 0.9758 | 0.9770 | 45.1 | 11.3 |
| cols | gru | 61,962 | 0.9769 | 0.9811 | 40.9 | 10.2 |
| patches | lstm | 76,042 | 0.9572 | 0.9620 | 47.7 | 11.9 |
| patches | gru | 57,354 | 0.9447 | 0.9456 | 44.4 | 11.1 |
| rows | lstm | 49,418 | 0.9741 | 0.9747 | 34.3 | 8.6 |
| rows | gru | 61,962 | 0.9762 | 0.9773 | 33.8 | 8.5 |
| baseline mlp | — | 235,146 | 0.9757 | 0.9768 | 51.1 | 6.4 |
| baseline cnn | — | 468,010 | 0.9900 | 0.9910 | 54.8 | 11.0 |
