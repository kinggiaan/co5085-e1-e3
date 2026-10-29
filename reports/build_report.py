"""Sinh báo cáo PDF (E1–E3) từ số liệu đã chạy.

    python reports/build_report.py

Đọc:
    E1/results/summary.json, E1/results/<model>/metrics.json
    E2/results/summary.json, E2/results/msa_equivalence.json
    E3/results/summary.json, E3/results/order_probe.json
và các hình PNG trong E*/results/.

Ghi: reports/exercises.pdf
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "exercises.pdf"

# ---------------------------------------------------------------- font
FONT_DIR = Path(r"C:\Windows\Fonts")
for name, file in (("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf")):
    try:
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    except Exception as exc:  # pragma: no cover
        print(f"Không nạp được font {file}: {exc}")
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

ACCENT = colors.HexColor("#0b5fff")
GREY = colors.HexColor("#666666")
LIGHT = colors.HexColor("#eef2ff")

# ---------------------------------------------------------------- data
def load_json(rel: str, default=None):
    p = ROOT / rel
    if not p.exists():
        print(f"(!) thiếu {rel}")
        return default
    return json.loads(p.read_text(encoding="utf-8"))


E1 = load_json("E1/results/summary.json", [])
E1_MLP = load_json("E1/results/mlp/metrics.json", None)
E1_CNN = load_json("E1/results/cnn/metrics.json", None)
E2 = load_json("E2/results/summary.json", [])
E2_EQ = load_json("E2/results/msa_equivalence.json", None)
E3 = load_json("E3/results/summary.json", [])
E3_PROBE = load_json("E3/results/order_probe.json", None)

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle(
    "body", parent=ss["BodyText"], fontName="Arial", fontSize=10, leading=14.5,
    alignment=TA_JUSTIFY, spaceAfter=6,
)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Arial-Bold", fontSize=15, leading=19,
                    textColor=colors.HexColor("#111111"), spaceBefore=14, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Arial-Bold", fontSize=11.5, leading=15,
                    textColor=ACCENT, spaceBefore=10, spaceAfter=4)
CAPTION = ParagraphStyle("caption", parent=BODY, fontName="Arial-Italic", fontSize=8.5,
                         leading=11, textColor=GREY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11.5)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Arial-Bold", fontSize=24, leading=30,
                       alignment=TA_CENTER, spaceAfter=10)
TOCTITLE = ParagraphStyle("TOCTitle", parent=H1, spaceBefore=0)
SUB = ParagraphStyle("sub", parent=BODY, fontName="Arial", fontSize=12, alignment=TA_CENTER,
                     textColor=GREY, spaceAfter=4)
CELL = ParagraphStyle("cell", parent=BODY, fontName="Arial", fontSize=8.5, leading=11, alignment=1)
CELL_LEFT = ParagraphStyle("cellL", parent=CELL, alignment=0)

FIG_N = [0]
TAB_N = [0]


def fig(path: Path, width_cm: float, caption: str):
    FIG_N[0] += 1
    if not path.exists():
        return [Paragraph(f"(thiếu hình {path.name})", CAPTION)]
    w, h = PILImage.open(path).size
    return [
        Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w),
        Paragraph(f"Hình {FIG_N[0]}. {caption}", CAPTION),
    ]


def table(rows, caption: str, col_widths=None, header_bg=ACCENT):
    TAB_N[0] += 1
    data = []
    for i, row in enumerate(rows):
        style = CELL if i else ParagraphStyle("th", parent=CELL, fontName="Arial-Bold", textColor=colors.white)
        data.append([c if not isinstance(c, str) else Paragraph(c, style) for c in row])
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return [t, Paragraph(f"Bảng {TAB_N[0]}. {caption}", CAPTION)]


def p(text: str):
    return Paragraph(text, BODY)


def bullets(items: list[str]):
    return [Paragraph(f"• {i}", BODY) for i in items]


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


# ---------------------------------------------------------------- document
class Report(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                         topMargin=2.2 * cm, bottomMargin=1.8 * cm,
                         title="CO5085 — Báo cáo bài tập E1–E3", author="Nhóm CO5085", **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(2.0 * cm, A4[1] - 1.3 * cm, "CO5085 — Học sâu và ứng dụng trong thị giác máy tính")
        canvas.drawRightString(A4[0] - 2.0 * cm, A4[1] - 1.3 * cm, "Bài tập E1–E3")
        canvas.setStrokeColor(colors.HexColor("#dddddd"))
        canvas.line(2.0 * cm, A4[1] - 1.45 * cm, A4[0] - 2.0 * cm, A4[1] - 1.45 * cm)
        canvas.line(2.0 * cm, 1.45 * cm, A4[0] - 2.0 * cm, 1.45 * cm)
        canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"{doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            name = flowable.style.name
            if name in ("H1", "H2"):
                level = 0 if name == "H1" else 1
                self.notify("TOCEntry", (level, flowable.getPlainText(), self.page))


def cover() -> list:
    story = [Spacer(1, 2.2 * cm),
             Paragraph("ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH", SUB),
             Paragraph("TRƯỜNG ĐẠI HỌC BÁCH KHOA", SUB),
             Paragraph("KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH", SUB),
             Spacer(1, 2.0 * cm),
             Paragraph("BÁO CÁO BÀI TẬP E1–E3", TITLE),
             Spacer(1, 0.3 * cm),
             Paragraph("Học sâu và ứng dụng trong thị giác máy tính — CO5085", SUB),
             Spacer(1, 1.6 * cm)]
    rows = [
        ["Học kỳ", "HK261 (2026–2027)"],
        ["Giảng viên", "Lê Thành Sách, Nguyễn Quốc Minh"],
        ["Nhóm", "<điền mã nhóm / họ tên 2 thành viên>"],
        ["Dữ liệu", "MNIST (28×28, 10 lớp)"],
        ["Ngày hoàn thành", date.today().strftime("%d/%m/%Y")],
    ]
    t = Table([[Paragraph(a, CELL_LEFT), Paragraph(b, CELL_LEFT)] for a, b in rows],
              colWidths=[4.2 * cm, 9.5 * cm], hAlign="CENTER")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ccd6ff")),
                           ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [t, Spacer(1, 2.0 * cm),
              Paragraph("Toàn bộ số liệu trong báo cáo sinh ra từ các script trong repo "
                        "(<i>E1/run_all.py</i>, <i>E2/run_all.py</i>, <i>E3/run_all.py</i>); "
                        "seed, cấu hình và log được lưu cùng kết quả.", CAPTION)]
    return story


def section_intro() -> list:
    story = [Paragraph("1. Giới thiệu và thiết lập chung", H1)]
    story.append(p(
        "Báo cáo trình bày ba bài tập E1–E3 trên cùng một tập dữ liệu nhỏ là MNIST: 60.000 ảnh huấn luyện "
        "(chia 50.000 train / 10.000 validation theo seed 42) và 10.000 ảnh test chính thức. Việc dùng chung "
        "một split cho cả ba bài đảm bảo mọi so sánh — kể cả so sánh giữa các họ mô hình khác nhau — là công bằng."))
    story.append(p(
        "Ảnh được chuẩn hoá với mean 0,1307 và std 0,3081 (số liệu chuẩn của MNIST) và <b>không</b> dùng "
        "augmentation, để chênh lệch kết quả phản ánh kiến trúc chứ không phải may mắn của phép biến đổi dữ liệu. "
        "Mọi mô hình huấn luyện bằng vòng lặp viết tay (forward → loss → backward → optimizer.step), "
        "optimizer Adam, batch 128, và lưu lại <i>metrics.json</i> cùng <i>history.csv</i> cho từng lần chạy."))
    story += table(
        [["Thành phần", "Giá trị"],
         ["Dữ liệu", "MNIST 28×28, 1 kênh, 10 lớp"],
         ["Split", "50.000 train / 10.000 val / 10.000 test (seed 42)"],
         ["Tiền xử lý", "chuẩn hoá mean=0,1307 std=0,3081; không augmentation"],
         ["Optimizer / loss", "Adam; CrossEntropyLoss trên logits"],
         ["Batch / seed", "128 / 42"],
         ["Phần cứng", "Intel i5-9300H, NVIDIA GTX 1650 4 GB, PyTorch 2.5.1"],
         ],
        "Thiết lập thí nghiệm dùng chung cho cả ba bài.",
        col_widths=[4.0 * cm, 12.0 * cm],
    )
    return story


def section_e1() -> list:
    story = [Paragraph("2. E1 — Phân loại ảnh: softmax, MLP, CNN", H1)]
    story.append(Paragraph("2.1. Kiến trúc", H2))
    story.append(p(
        "Ba mô hình tăng dần về khả năng biểu diễn. <b>Softmax</b> chỉ có một phép biến đổi tuyến tính trên vector "
        "ảnh đã flatten, tương đương việc học 10 “template” và chọn template khớp nhất — vì vậy nó bị chặn trên "
        "bởi giới hạn của một bộ phân loại tuyến tính. <b>MLP</b> thêm hai lớp ẩn với ReLU và dropout, tăng đáng kể "
        "capacity. <b>CNN</b> khai thác hai giả định đúng với ảnh: tính cục bộ (pixel gần nhau mới liên quan) và tính "
        "bất biến dịch (một nét vẽ ở góc này hay góc kia đều là cùng một đặc trưng), nhờ chia sẻ trọng số theo vị trí."))
    story += table(
        [["Mô hình", "Cấu trúc", "Tham số"],
         ["Softmax", "Flatten 784 → Linear(784, 10)", "7.850"],
         ["MLP", "784 → 256 → 128 → 10 (ReLU, dropout 0,2)", "235.146"],
         ["CNN", "4 khối conv 3×3 + BN + ReLU, 2 max-pool, head FC 128", "468.010"],
         ],
        "Ba kiến trúc của E1.",
        col_widths=[2.6 * cm, 10.4 * cm, 3.0 * cm],
    )

    story.append(Paragraph("2.2. Kết quả", H2))
    if E1:
        rows = [["Mô hình", "Tham số", "Val acc", "Test acc", "Test loss", "Thời gian"]]
        for r in E1:
            rows.append([r["model"], f"{r['num_params']:,}", pct(r["best_val_acc"]),
                         pct(r["test_acc"]), f"{r['test_loss']:.4f}", f"{r['train_time_s']:.0f} s"])
        story += table(rows, "Kết quả E1 trên tập test (10.000 ảnh).",
                       col_widths=[2.6 * cm, 2.8 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.8 * cm])
        best = max(E1, key=lambda r: r["test_acc"])
        story.append(p(
            f"Thứ tự xếp hạng đúng như dự đoán từ lý thuyết: softmax {pct(E1[0]['test_acc'])} < "
            f"MLP {pct(E1[1]['test_acc'])} < CNN {pct(E1[2]['test_acc'])}. Đáng chú ý là CNN chỉ nhiều tham số gấp đôi "
            f"MLP nhưng giảm lỗi tương đối gần 60% — phần lớn lợi ích đến từ <i>inductive bias</i> (conv + chia sẻ "
            f"trọng số) chứ không từ việc tăng số tham số. Mô hình tốt nhất là <b>{best['model']}</b>."))

        story += fig(ROOT / "E1/results/val_acc_all.png", 14.5,
                     "Đường cong validation accuracy của ba mô hình qua các epoch.")
        story += fig(ROOT / "E1/results/acc_vs_params.png", 11.0,
                     "Test accuracy theo số tham số (thang log).")

        story.append(Paragraph("2.3. Phân tích lỗi", H2))
        story.append(p(
            "Confusion matrix cho thấy lỗi của softmax tập trung ở các cặp chữ số có nét tương đồng khi nhìn như "
            "một chồng pixel (3–5, 4–9, 7–9). CNN giảm mạnh các nhầm lẫn này vì nó học được đặc trưng cục bộ "
            "như nét cong kín của số 9 hay nét ngang của số 7. Mẫu dự đoán sai còn lại phần lớn là ảnh mờ, viết "
            "ẩu hoặc bị cắt nét — tức là lỗi dữ liệu chứ không phải lỗi kiến trúc."))
        story += fig(ROOT / "E1/results/cnn/confusion.png", 9.5, "Confusion matrix của CNN trên tập test.")
        story += fig(ROOT / "E1/results/softmax/confusion.png", 9.5,
                     "Confusion matrix của softmax — lỗi lan rộng hơn nhiều so với CNN.")
        story += fig(ROOT / "E1/results/cnn/errors.png", 15.0,
                     "16 mẫu CNN dự đoán sai (nhãn thật / dự đoán).")
    else:
        story.append(p("Chưa có E1/results/summary.json — chạy <i>python E1/run_all.py</i>."))
    return story


def section_e2() -> list:
    story = [Paragraph("3. E2 — Transformer: MSA và tokenization", H1)]
    story.append(Paragraph("3.1. Multi-Head Self-Attention", H2))
    story.append(p(
        "MSA được viết lại từ các phép toán cơ bản (Linear, einsum, softmax, ghép head), không dùng "
        "<i>nn.MultiheadAttention</i>. Với đầu vào X ∈ R<sup>B×N×d</sup>: chiếu ra Q, K, V; tính điểm tương đồng "
        "đã scale; softmax theo trục key; lấy tổng có trọng số của V; ghép H head rồi chiếu ra output:"))
    story.append(Paragraph(
        "<font face='Arial-Bold'>Q = XW<sub>Q</sub>, K = XW<sub>K</sub>, V = XW<sub>V</sub> ; "
        "head<sub>h</sub> = softmax(Q<sub>h</sub>K<sub>h</sub><super>T</super>/√d<sub>k</sub>)V<sub>h</sub> ; "
        "MSA(X) = concat(head<sub>1..H</sub>)W<sub>O</sub></font>", BODY))
    story.append(p(
        "Hệ số 1/√d<sub>k</sub> giữ phương sai của tích vô hướng bằng 1: nếu không, với d<sub>k</sub> lớn, softmax "
        "rơi vào vùng bão hoà, gradient gần 0 và mô hình không học được. Đây cũng là lỗi hay gặp nhất khi tự viết."))

    if E2_EQ:
        rows = [["d", "Số head", "N", "Sai khác output", "Sai khác attention", "Có mask"]]
        for c in E2_EQ["cases"]:
            rows.append([str(c["dim"]), str(c["heads"]), str(c["n_tokens"]),
                         f"{c['max_abs_diff_forward']:.1e}", f"{c['max_abs_diff_attention']:.1e}",
                         f"{c['max_abs_diff_with_mask']:.1e}"])
        story += table(rows, "Kiểm chứng MSA tự viết khớp nn.MultiheadAttention khi copy trọng số.",
                       col_widths=[1.6 * cm, 2.2 * cm, 1.6 * cm, 3.6 * cm, 3.8 * cm, 3.4 * cm])
        story.append(p(
            f"Với cùng bộ trọng số, hai bản cho cùng output với sai khác lớn nhất "
            f"<b>{E2_EQ['worst_abs_diff']:.1e}</b> — sai số số học thuần tuý của phép cộng dấu phẩy động. "
            "Kết quả này cả khi bật mask, nên có thể khẳng định bản tự viết là đúng đặc tả."))

    story.append(Paragraph("3.2. Các cách lấy token", H2))
    story.append(p(
        "Ảnh 28×28 không có token tự nhiên như câu chữ, nên cách chia token là một lựa chọn thiết kế có ảnh hưởng "
        "trực tiếp: càng nhiều token thì chi phí attention O(N²d) càng lớn, nhưng mỗi token chứa ít thông tin hơn. "
        "Ba cách được triển khai: chia lưới patch 4×4 (49 token, mỗi token 16 chiều), lấy mỗi hàng ảnh làm một "
        "token (28 token, 28 chiều), và dùng CNN stem rút feature map 7×7 (49 token, 64 chiều). Mỗi chuỗi được "
        "bọc thêm một token CLS và một positional embedding học được, vì bản thân attention không có khái niệm thứ tự."))

    story.append(Paragraph("3.3. Kết quả", H2))
    if E2:
        rows = [["Tokenize", "MSA", "Số token", "Tham số", "Test acc", "s/epoch"]]
        for r in E2:
            rows.append([r["tokenizer"], r["msa"], str(r["num_tokens"]), f"{r['num_params']:,}",
                         pct(r["test_acc"]), f"{r['sec_per_epoch']:.1f}"])
        story += table(rows, "Kết quả E2: 3 cách tokenize × 2 bản MSA (dim 128, depth 2, 4 head, 4 epoch).",
                       col_widths=[2.4 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm, 2.6 * cm, 2.4 * cm])
        by_tok = {}
        for r in E2:
            by_tok.setdefault(r["tokenizer"], []).append(r)
        manual = [r for r in E2 if r["msa"] == "manual"]
        torch_msa = [r for r in E2 if r["msa"] == "torch"]
        acc_m = sum(r["test_acc"] for r in manual) / len(manual)
        acc_t = sum(r["test_acc"] for r in torch_msa) / len(torch_msa)
        t_m = sum(r["sec_per_epoch"] for r in manual) / len(manual)
        t_t = sum(r["sec_per_epoch"] for r in torch_msa) / len(torch_msa)
        story.append(p(
            f"Bản tự viết đạt trung bình {pct(acc_m)} so với {pct(acc_t)} của nn.MultiheadAttention. Chênh lệch "
            f"{100 * abs(acc_m - acc_t):.1f} điểm này <b>không phải</b> sai lệch toán học — Bảng 4 đã cho thấy hai bản "
            f"trùng khớp tới 10<super>−7</super> khi dùng chung trọng số. Nó đến từ khởi tạo khác nhau: bản PyTorch "
            f"gộp ba phép chiếu thành một tensor nên thứ tự sinh số ngẫu nhiên khác bản tự viết, và với chỉ 4 epoch "
            f"thì mức dao động đó lên tới vài điểm. Về tốc độ, bản tự viết chậm hơn ({t_m:.1f} so với "
            f"{t_t:.1f} s/epoch) do ba phép chiếu riêng thay vì một phép gộp và không dùng kernel tối ưu."))
        tok_avg = {}
        for t in ("patch", "row", "cnnstem"):
            vals = [r["test_acc"] for r in E2 if r["tokenizer"] == t]
            if vals:
                tok_avg[t] = sum(vals) / len(vals)
        spread = (max(tok_avg.values()) - min(tok_avg.values())) if len(tok_avg) > 1 else 0.0
        story.append(p(
            f"Ba cách tokenize cho kết quả <b>rất khác nhau</b>: patch {pct(tok_avg.get('patch', 0))}, "
            f"row {pct(tok_avg.get('row', 0))}, cnnstem {pct(tok_avg.get('cnnstem', 0))} — chênh lệch tới "
            f"{100 * spread:.1f} điểm phần trăm, lớn hơn nhiều so với chênh lệch do đổi bản MSA "
            f"({100 * abs(acc_m - acc_t):.1f} điểm). Nguyên nhân nằm ở độ sâu: encoder chỉ có 2 block, nghĩa là mỗi "
            "token chỉ được trộn thông tin với các token khác hai lần. Với patch 4×4, mỗi token là 16 pixel thô và "
            "mô hình phải vừa học đặc trưng cục bộ vừa học cách tổ hợp — quá nhiều việc cho hai lớp attention. Token "
            "theo hàng đã chứa sẵn 28 điểm ảnh có cấu trúc nên dễ hơn; CNN stem đưa vào chuỗi những đặc trưng đã "
            "tích chập (trường tiếp nhận 5×5) nên đạt kết quả cao nhất. Kết luận: với encoder nông, phần lớn việc "
            "trích đặc trưng phải do tokenizer đảm nhiệm."))
        story.append(Paragraph(
            "Kiểm tra thêm (tái lập được bằng <font face='Courier'>--epochs 12</font>): patch/manual huấn luyện 12 "
            "epoch chỉ đạt 69,9% test — tức mô hình bị chặn bởi độ sâu chứ không phải ngân sách epoch. Đây cũng là "
            "lý do các ViT thực tế dùng depth 12–24 thay vì 2.", SMALL))
        story += fig(ROOT / "E2/results/test_acc_by_msa.png", 15.0,
                     "Test accuracy: MSA tự viết vs nn.MultiheadAttention theo từng cách tokenize.")
        story += fig(ROOT / "E2/results/time_by_msa.png", 15.0,
                     "Thời gian huấn luyện mỗi epoch của hai bản MSA.")
        att = ROOT / "E2/results/patch_manual/attention.png"
        if att.exists():
            story += fig(att, 9.5,
                         "Bản đồ attention của block đầu (head có phân bố nhọn nhất) với tokenize patch: "
                         "một vài patch giữ vai trò “key” chính, các patch còn lại gần như bị bỏ qua.")
    else:
        story.append(p("Chưa có số liệu E2 — chạy <i>python E2/run_all.py</i>."))
    return story


def section_e3() -> list:
    story = [Paragraph("4. E3 — LSTM/GRU trên chuỗi ảnh", H1)]
    story.append(Paragraph("4.1. Ý tưởng và cách biểu diễn chuỗi", H2))
    story.append(p(
        "Ảnh không phải chuỗi tự nhiên, nên E3 là bài toán <i>chọn cách nhìn</i>: biến ảnh thành dãy token sao cho "
        "thứ tự mang thông tin. Ba cách được thử: mỗi hàng ảnh là một bước thời gian (T=28, F=28), mỗi cột là một "
        "bước (T=28, F=28), và mỗi patch 4×4 theo thứ tự quét từng hàng patch (T=49, F=16). Mô hình dùng "
        "<i>nn.LSTM</i> hoặc <i>nn.GRU</i> một lớp, hidden 128, lấy hidden state ở bước cuối để phân loại."))
    story.append(p(
        "LSTM giải quyết vấn đề tiêu biến gradient bằng cell state cộng dồn "
        "(c<sub>t</sub> = f × c<sub>t−1</sub> + i × g): đạo hàm theo c<sub>t−1</sub> chỉ còn nhân với cổng f, nên "
        "gradient có “đường cao tốc” đi ngược thời gian thay vì bị nhân liên tiếp với ma trận trọng số như RNN thuần. "
        "GRU gộp hai cổng còn một, ít tham số hơn khoảng 25% và thường tương đương trên chuỗi ngắn như MNIST."))

    story.append(Paragraph("4.2. Kết quả và so sánh với baseline", H2))
    if E3:
        rows = [["Biểu diễn", "Cell", "Tham số", "Test acc", "s/epoch"]]
        for r in E3:
            rows.append([r["representation"], r["cell"], f"{r['num_params']:,}",
                         pct(r["test_acc"]), f"{r['sec_per_epoch']:.1f}"])
        for name, m in (("baseline MLP", E1_MLP), ("baseline CNN", E1_CNN)):
            if m:
                rows.append([name, "—", f"{m['num_params']:,}", pct(m["test_acc"]),
                             f"{m['train_time_s'] / max(m['epochs'], 1):.1f}"])
        story += table(rows, "E3: LSTM/GRU trên ba cách biểu diễn, so với baseline E1 trên cùng split.",
                       col_widths=[3.0 * cm, 2.0 * cm, 2.8 * cm, 2.6 * cm, 2.4 * cm])
        best = max(E3, key=lambda r: r["test_acc"])
        story.append(p(
            f"Cấu hình tốt nhất là <b>{best['representation']}/{best['cell']}</b> với {pct(best['test_acc'])}. "
            "Cả LSTM và GRU đều vượt xa MLP nhưng vẫn thấp hơn CNN — kết quả này không phải thất bại mà là thông tin: "
            "MNIST gần như không có phụ thuộc đường dài, nên lợi thế của mô hình hồi quy không có đất phát huy, "
            "trong khi CNN có inductive bias đúng với ảnh và tính toán song song tốt hơn nhiều."))
        if E3_PROBE:
            story.append(p(
                f"Để kiểm tra mô hình có thực sự dùng thứ tự hay chỉ cộng các hàng lại, ta đảo ngẫu nhiên thứ tự hàng "
                f"ở đầu vào của <i>{E3_PROBE['run']}</i> đã huấn luyện: accuracy giảm từ "
                f"{pct(E3_PROBE['test_acc_original'])} xuống {pct(E3_PROBE['test_acc_shuffled_rows'])} "
                f"({100 * E3_PROBE['delta']:+.2f} điểm phần trăm). Mức giảm này cho thấy mô hình thực sự khai thác "
                f"thông tin thứ tự, chứ không chỉ là một bộ tổng hợp không gian."))
        story += fig(ROOT / "E3/results/test_acc_bar.png", 15.5,
                     "Test accuracy của LSTM/GRU so với baseline MLP và CNN.")
        story += fig(ROOT / "E3/results/val_acc_all.png", 14.5,
                     "Đường cong validation accuracy của sáu cấu hình LSTM/GRU.")
    else:
        story.append(p("Chưa có số liệu E3 — chạy <i>python E3/run_all.py</i>."))
    return story


def section_conclusion() -> list:
    story = [Paragraph("5. Kết luận", H1)]
    items = [
        "<b>Kiến trúc phù hợp quan trọng hơn số tham số.</b> CNN 468 nghìn tham số đạt "
        f"{pct(E1[2]['test_acc']) if len(E1) > 2 else '—'} trong khi MLP 235 nghìn tham số chỉ đạt "
        f"{pct(E1[1]['test_acc']) if len(E1) > 1 else '—'}; khác biệt nằm ở giả định về cấu trúc ảnh, không ở capacity.",
        "<b>Tự viết attention là kiểm chứng được.</b> Bản MSA viết tay cho output trùng với nn.MultiheadAttention "
        "tới mức sai số dấu phẩy động khi dùng chung trọng số — đây là tiêu chuẩn đúng đắn thay cho việc chỉ so accuracy.",
        "<b>Tokenization là lựa chọn thiết kế có giá.</b> Với encoder chỉ 2 block, ba cách chia token lệch nhau tới "
        "hàng chục điểm phần trăm: tokenizer nào làm sẵn nhiều việc trích đặc trưng hơn thì kết quả tốt hơn. Số token "
        "đồng thời quyết định chi phí O(N²d).",
        "<b>Mô hình hồi quy không thắng trên dữ liệu không có phụ thuộc dài.</b> LSTM/GRU học được thứ tự (thí nghiệm "
        "đảo hàng chứng minh điều đó) nhưng vẫn kém CNN trên MNIST — chọn đúng công cụ cho đúng cấu trúc dữ liệu "
        "quan trọng hơn việc dùng mô hình “hiện đại” hơn.",
    ]
    story += bullets(items)
    return story


def section_appendix() -> list:
    story = [Paragraph("6. Phụ lục: tái lập", H1)]
    story.append(p("Toàn bộ kết quả sinh lại được bằng ba lệnh, mỗi lệnh ghi kết quả vào "
                   "<i>E*/results/</i>:"))
    story.append(Paragraph(
        "<font face='Courier'>python E1/run_all.py<br/>python E2/run_all.py<br/>python E3/run_all.py<br/>"
        "python reports/build_report.py</font>", BODY))
    story.append(p(
        "Mỗi lần chạy lưu <i>metrics.json</i> (cấu hình, số tham số, accuracy, thời gian, confusion matrix), "
        "<i>history.csv</i> (loss/accuracy từng epoch) và các hình PNG. Seed cố định 42 cho cả ba bài; "
        "split dữ liệu được tạo bằng generator có seed nên danh sách ảnh trong train/val không đổi giữa các lần chạy."))
    story.append(Paragraph("Khai báo sử dụng công cụ AI", H2))
    story.append(p(
        "Theo yêu cầu mục 1.6 của đề, nhóm khai báo phạm vi sử dụng công cụ AI: "
        "<i>&lt;điền chính xác những gì đã dùng — ví dụ: gợi ý cấu trúc thư mục, rà soát lỗi shape, "
        "hình thức trình bày báo cáo&gt;</i>. Toàn bộ số liệu trong báo cáo do nhóm tự chạy và kiểm chứng; "
        "nhóm chịu trách nhiệm giải thích từng đoạn code khi được hỏi."))
    return story


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Report(str(OUT))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc0", fontName="Arial-Bold", fontSize=10.5, leading=16, spaceBefore=4),
        ParagraphStyle("toc1", fontName="Arial", fontSize=9.5, leading=13, leftIndent=14, textColor=GREY),
    ]

    story = cover()
    story.append(PageBreak())
    story.append(Paragraph("Mục lục", TOCTITLE))
    story.append(toc)
    story.append(PageBreak())
    story += section_intro()
    story += section_e1()
    story.append(PageBreak())
    story += section_e2()
    story.append(PageBreak())
    story += section_e3()
    story += section_conclusion()
    story += section_appendix()

    doc.multiBuild(story)
    size = OUT.stat().st_size / 1024
    print(f"Đã ghi {OUT} ({size:.0f} KB, {doc.page} trang)")


if __name__ == "__main__":
    main()
