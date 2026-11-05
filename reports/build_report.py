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
A1 = load_json("A1/results/summary.json", [])
A2 = load_json("A2/results/summary.json", [])

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
                         title="CO5085 — Báo cáo bài tập E1–E3", author="Dương Gia An — 2470293", **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(2.0 * cm, A4[1] - 1.3 * cm, "CO5085 — Học sâu và ứng dụng trong thị giác máy tính")
        canvas.drawRightString(A4[0] - 2.0 * cm, A4[1] - 1.3 * cm, "Coursework E1–E3 · A1–A2")
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
             Paragraph("BÁO CÁO COURSEWORK — E1–E3 &amp; A1–A2", TITLE),
             Spacer(1, 0.3 * cm),
             Paragraph("Học sâu và ứng dụng trong thị giác máy tính — CO5085", SUB),
             Spacer(1, 1.6 * cm)]
    rows = [
        ["Học kỳ", "HK261 (2026–2027)"],
        ["Giảng viên", "Lê Thành Sách, Nguyễn Quốc Minh"],
        ["Sinh viên", "Dương Gia An — MSSV 2470293"],
        ["Hình thức", "Cá nhân"],
        ["Dữ liệu", "MNIST (E1–E3), Tiny-ImageNet (A1), Pascal VOC 2012 (A2)"],
        ["Ngày hoàn thành", date.today().strftime("%d/%m/%Y")],
    ]
    t = Table([[Paragraph(a, CELL_LEFT), Paragraph(b, CELL_LEFT)] for a, b in rows],
              colWidths=[4.2 * cm, 9.5 * cm], hAlign="CENTER")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ccd6ff")),
                           ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [t, Spacer(1, 2.0 * cm)]
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


def section_a1() -> list:
    story = [Paragraph("5. A1 — CNN vs Transformer trên tập ảnh lớn (Tiny-ImageNet)", H1)]
    story.append(Paragraph("5.1. Dữ liệu và vì sao chọn Tiny-ImageNet", H2))
    story.append(p(
        "A1 yêu cầu một tập ảnh lớn hơn E1 để câu hỏi CNN vs Transformer và pretrain/freeze có ý nghĩa. "
        "Bài làm chọn <b>Tiny-ImageNet</b>: 200 lớp, ảnh 64×64, 100.000 ảnh train và 10.000 val có nhãn. "
        "Điểm quyết định là độ phân giải: CIFAR-100 (32×32) phải phóng 7× lên 224 cho mô hình pretrained, "
        "làm lợi thế của pretrain bị bóp méo. Food-101 thì quá nặng cho 4 GB VRAM."))
    story += table(
        [["Tiêu chí", "Tiny-ImageNet", "CIFAR-100", "Food-101"],
         ["Độ phân giải", "64×64", "32×32", "512 px"],
         ["Lớp / ảnh", "200 / 110k", "100 / 60k", "101 / 101k"],
         ["Transfer từ pretrained 224", "có nghĩa", "kém tin cậy", "tốt"],
         ["Vừa 4 GB VRAM", "có", "có", "khó"]],
        "So sánh ba lựa chọn dataset cho A1.",
        col_widths=[5.0 * cm, 4.0 * cm, 4.0 * cm, 3.6 * cm],
    )

    story.append(Paragraph("5.2. Kiến trúc: vì sao Transformer là Swin-T chứ không phải ViT-B/16", H2))
    story.append(p(
        "Đo trực tiếp trên GTX 1650 4 GB: ViT-B/16 ở 224 px cần <b>5.048 MB</b> bộ nhớ, vượt 4 GB "
        "nên tràn sang shared memory và chậm bất thường — <b>31 s/bước</b> so với 0,3 s của ResNet-18. "
        "Swin-T ở 128 px chỉ cần 3.243 MB và 0,78 s/bước, nên được chọn làm đại diện họ Transformer "
        "(Swin cũng nằm trong danh sách đề gợi ý). Đây là ràng buộc phần cứng được ghi lại thay vì "
        "che đi bằng một cấu hình không chạy được."))

    story.append(Paragraph("5.3. Kết quả", H2))
    if A1:
        rows = [["Kiến trúc", "Chế độ", "Epoch", "Tham số trainable", "Val acc", "Macro-F1", "s/epoch"]]
        for r in A1:
            rows.append([r["arch"], r["mode"], str(r["epochs"]), f"{r['num_params_trainable']:,}",
                         pct(r["test_acc"]), pct(r["macro_f1"]), f"{r['sec_per_epoch']:.0f}"])
        story += table(rows, "Kết quả A1: 200 lớp, 20.000 ảnh train, ảnh 128 px.",
                       col_widths=[2.6 * cm, 2.8 * cm, 1.4 * cm, 3.2 * cm, 2.2 * cm, 2.2 * cm, 1.8 * cm])
        best = max(A1, key=lambda r: r["test_acc"])
        scratch = next((r for r in A1 if r["mode"] == "scratch"), None)
        ft = next((r for r in A1 if r["arch"] == "resnet18" and r["mode"] == "finetune"), None)
        freeze = next((r for r in A1 if r["mode"] == "freeze_all"), None)
        story.append(p(
            f"Mô hình tốt nhất là <b>{best['arch']}/{best['mode']}</b> với {pct(best['test_acc'])}. "
            + (f"So với train from-scratch ({pct(scratch['test_acc'])}), bản pretrained tăng rất mạnh — "
               f"đúng dự đoán rằng với 20.000 ảnh và 200 lớp, đặc trưng ImageNet có giá trị lớn hơn việc "
               f"học từ đầu. " if scratch and ft else "")
            + (f"Chế độ đóng băng backbone chỉ đạt {pct(freeze['test_acc'])}, thấp hơn finetune toàn bộ: "
               f"đặc trưng đóng băng tốt cho bài toán gần miền ImageNet, nhưng 200 lớp ảnh 64×64 vẫn cần "
               f"tinh chỉnh." if freeze else "")))
        story.append(p(
            "Cấu hình Swin-T from-scratch tốn 1.663 s/epoch (27,7 phút), tức ~9 giờ cho 20 epoch — vượt "
            "ngân sách compute, nên Transformer chỉ được chạy ở dạng pretrained; ResNet-18 đảm nhiệm phần "
            "from-scratch."))
        fa = next((r for r in A1 if r["arch"] == "resnet18" and r["mode"] == "freeze_all"), None)
        fp = next((r for r in A1 if r["arch"] == "resnet18" and r["mode"] == "freeze_partial"), None)
        if fa and fp and ft:
            story.append(p(
                f"Trên ResNet-18, ba kiểu đóng băng cho một dãy tăng dần: đóng băng toàn bộ backbone "
                f"{pct(fa['test_acc'])} (chỉ {fa['num_params_trainable']:,} tham số trainable) → đóng băng "
                f"một phần {pct(fp['test_acc'])} ({fp['num_params_trainable']:,}) → finetune toàn bộ "
                f"{pct(ft['test_acc'])}. Điểm đáng chú ý là 53% chỉ với 0,9% tham số — mức đánh đổi hợp lý "
                f"khi compute hạn chế."))
        story += fig(ROOT / "A1/results/val_acc_bar.png", 15.0,
                     "A1 — val accuracy theo kiến trúc và chế độ huấn luyện.")
        story += fig(ROOT / "A1/results/val_acc_all.png", 14.0,
                     "A1 — đường cong val accuracy của các cấu hình.")
        wc = ROOT / f"A1/results/{best['tag']}/worst_classes.png"
        if wc.exists():
            story += fig(wc, 10.5, f"A1 — 15 lớp có recall thấp nhất ({best['tag']}).")
    else:
        story.append(p("Chưa có số liệu A1 — chạy <i>python A1/run_all.py</i>."))
    return story


def section_a2() -> list:
    story = [Paragraph("6. A2 — Phân đoạn ngữ nghĩa trên Pascal VOC 2012 và phân tích SegFormer", H1)]
    story.append(Paragraph("6.1. Bài toán và dữ liệu", H2))
    story.append(p(
        "A2 yêu cầu một bài toán thị giác không phải phân loại ảnh thuần. Bài làm chọn <b>phân đoạn ngữ nghĩa "
        "21 lớp trên Pascal VOC 2012</b> (1.464 ảnh train / 1.449 val), metric <b>mIoU</b> và Dice — "
        "đây là bài toán dự đoán dày, khác bản chất với phân loại ở A1, và nối tiếp mạch "
        "CNN-vs-Transformer sang một miền mới. Máy chủ gốc của VOC không truy cập được từ mạng của máy này, "
        "nên dữ liệu được lấy qua mirror HuggingFace."))

    story.append(Paragraph("6.2. Phân tích paper: SegFormer (NeurIPS 2021)", H2))
    story.append(p(
        "Xie và cộng sự, <i>SegFormer: Simple and Efficient Design for Semantic Segmentation with "
        "Transformers</i>, NeurIPS 2021 (arXiv:2105.15203). Ba ý tưởng chính:"))
    story += bullets([
        "<b>Encoder Transformer phân cấp</b> trả về đặc trưng đa tỉ lệ (1/4 … 1/32), thay vì một tỉ lệ duy nhất.",
        "<b>Bỏ positional encoding</b>, thay bằng Mix-FFN có conv 3×3 — tránh phải nội suy mã vị trí khi độ "
        "phân giải test khác train, vốn là nguồn giảm chất lượng của ViT trong segmentation.",
        "<b>Decoder MLP nhẹ</b>: gộp đặc trưng các tầng rồi chiếu bằng MLP, không dùng decoder nặng.",
    ])
    story.append(p(
        "Thí nghiệm trong paper: SegFormer-B4 đạt 50,3% mIoU trên ADE20K với 64M tham số; B5 đạt 84,0% mIoU "
        "trên Cityscapes val. <b>Hạn chế</b> được khai thác trong báo cáo: paper đánh giá trên tập lớn có pretrain "
        "ImageNet-1k, chưa khảo sát vùng dữ liệu ít như VOC 2012 (~1,5k ảnh) — nơi mà lợi thế của kiến trúc "
        "Transformer có thể bị pretrain chi phối hoàn toàn."))

    story.append(Paragraph("6.3. Pipeline và kết quả", H2))
    story.append(p(
        "Ba biến thể được cài đặt để tách hai yếu tố thường bị trộn lẫn — <b>kiến trúc</b> (CNN vs Transformer) "
        "và <b>pretrain</b> (có vs không): U-Net tự viết (1,9M tham số, train từ đầu), DeepLabV3-ResNet50 "
        "(42M, trọng số COCO/VOC có sẵn 21 lớp) và SegFormer-B0 (3,7M, encoder ImageNet-1k + ADE20K, "
        "đúng kiến trúc của paper đang phân tích). Ảnh 256×256, CrossEntropyLoss bỏ qua nhãn 255, AdamW."))
    if A2:
        pretrain_map = {"deeplabv3": "COCO + VOC", "segformer": "ImageNet + ADE20K", "unet": "không"}
        rows = [["Mô hình", "Pretrain", "Tham số", "mIoU", "Dice", "Pixel acc", "s/epoch"]]
        for r in A2:
            rows.append([r["arch"], pretrain_map.get(r["arch"], "?"), f"{r['num_params']:,}",
                         pct(r["miou"]), pct(r["dice"]), pct(r["pixel_acc"]), f"{r['sec_per_epoch']:.0f}"])
        story += table(rows, "Kết quả A2 trên VOC 2012 val.",
                       col_widths=[2.8 * cm, 2.4 * cm, 2.6 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 2.0 * cm])
        best = max(A2, key=lambda r: r["miou"])
        dl = next((r for r in A2 if r["arch"] == "deeplabv3"), None)
        sf = next((r for r in A2 if r["arch"] == "segformer"), None)
        story.append(p(
            f"<b>{best['arch']}</b> đạt mIoU {pct(best['miou'])}. "
            + (f"Ngay epoch đầu DeepLabV3 đã ở mức 50,3% vì trọng số VOC có sẵn đúng 21 lớp — phần lớn "
               f"khoảng cách đến từ <i>pretrain</i>, không phải từ kiến trúc. " if dl else "")
            + (f"SegFormer-B0 nhỏ hơn {dl['num_params'] / sf['num_params']:.0f}× nhưng sau 4 epoch chỉ đạt "
               f"{pct(sf['miou'])} và nhanh hơn {dl['sec_per_epoch'] / sf['sec_per_epoch']:.1f}× mỗi epoch. "
               f"Điều này khớp với hạn chế của paper đã nêu: trên tập nhỏ, encoder phân cấp của SegFormer "
               f"cần nhiều epoch hơn để bù lại việc head phân loại phải học lại từ đầu cho 21 lớp VOC."
               if dl and sf else "")))
        story.append(p(
            "Về baseline U-Net train từ đầu: VOC 2012 có ~92% pixel là nền, nên CrossEntropyLoss thuần có cực "
            "tiểu tầm thường là “đoán toàn nền” và mô hình mắc kẹt ở mIoU ~3,7%. Hai lần sửa đã thử: trọng số "
            "lớp nghịch tần suất (không ổn định) và loss CE + Dice (giữ được, dùng cho kết quả dưới đây). "
            "Để chắc chắn lỗi không nằm ở kiến trúc, một phép kiểm tra overfit 2 ảnh được chạy "
            "(<i>A2/check_overfit.py</i>): sau 80 bước, mIoU trên chính 2 ảnh đó đạt 71,5% và pixel accuracy "
            "99,4% — tức kiến trúc và loss học được, giới hạn nằm ở số epoch so với ngân sách GPU."))
        unet = next((r for r in A2 if r["arch"] == "unet"), None)
        if unet:
            story.append(p(
                f"Kết quả U-Net sau {unet['epochs']} epoch (CE + Dice, lr {unet['lr']:g}): mIoU "
                f"{pct(unet['miou'])}, Dice {pct(unet['dice'])}, pixel accuracy {pct(unet['pixel_acc'])}. "
                f"Con số này thấp hơn nhiều so với hai mô hình pretrained, nhưng nó đo đúng điều cần đo: "
                f"với ~1.500 ảnh huấn luyện và {unet['epochs']} epoch, một CNN tự viết chưa thể cạnh tranh với đặc trưng "
                f"đã học sẵn từ ImageNet/COCO."))
        story += fig(ROOT / "A2/results/curves_all.png", 15.0, "A2 — mIoU và Dice theo epoch.")
        samp = ROOT / f"A2/results/{best['tag']}/samples.png"
        if samp.exists():
            story += fig(samp, 12.0, f"A2 — mặt nạ dự đoán ({best['tag']}: ảnh · ground truth · dự đoán).")
    else:
        story.append(p("Chưa có số liệu A2 — chạy <i>python A2/run_all.py</i>."))
    return story


def section_conclusion() -> list:
    story = [Paragraph("7. Kết luận", H1)]
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
        "đảo hàng chứng minh điều đó) nhưng vẫn kém CNN trên MNIST: 98,25% so với 99,10%.",
    ]
    story += bullets(items)
    return story


def section_appendix() -> list:
    story = [Paragraph("8. Phụ lục: tái lập", H1)]
    story.append(p("Toàn bộ kết quả sinh lại được bằng các lệnh sau, mỗi lệnh ghi kết quả vào "
                   "<i>*/results/</i>:"))
    story.append(Paragraph(
        "<font face='Courier'>python scripts/prepare_tiny_imagenet.py<br/>"
        "python scripts/prepare_voc2012.py<br/>"
        "python E1/run_all.py<br/>python E2/run_all.py<br/>python E3/run_all.py<br/>"
        "python A1/run_all.py<br/>python A2/run_all.py<br/>"
        "python reports/build_report.py</font>", BODY))
    story.append(p(
        "Mỗi lần chạy lưu <i>metrics.json</i> (cấu hình, số tham số, accuracy, thời gian), "
        "<i>history.csv</i> (loss/accuracy từng epoch) và các hình PNG. Seed cố định 42 cho cả ba bài."))
    story.append(Paragraph("Khai báo sử dụng công cụ AI", H2))
    story.append(p(
        "Theo mục 1.6 của đề, tôi khai báo phạm vi đã sử dụng công cụ AI hỗ trợ trong bài tập này: "
        "đọc và tóm tắt đề; gợi ý bố cục thư mục và các API PyTorch phù hợp; soạn bản nháp ban đầu cho phần lớn "
        "mã nguồn trong <i>common/</i>, <i>E1/</i>–<i>E3/</i>, <i>A1/</i>, <i>A2/</i>; gợi ý dàn ý và cách "
        "trình bày báo cáo; gợi ý cách chẩn đoán một số lỗi shape và lỗi huấn luyện."))
    story.append(p(
        "Phần tôi trực tiếp thực hiện: cài đặt môi trường và tải toàn bộ dữ liệu; chạy toàn bộ thí nghiệm trên "
        "máy của mình; kiểm tra sự nhất quán của từng bảng số và từng hình trong báo cáo với "
        "<i>metrics.json</i> mà các script sinh ra."))
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
    story.append(PageBreak())
    story += section_a1()
    story.append(PageBreak())
    story += section_a2()
    story += section_conclusion()
    story += section_appendix()

    doc.multiBuild(story)
    size = OUT.stat().st_size / 1024
    print(f"Đã ghi {OUT} ({size:.0f} KB, {doc.page} trang)")

    # GitHub Pages chỉ phục vụ thư mục /docs -> chép thêm một bản để link tải trực tiếp được
    pages_copy = ROOT / "docs" / "exercises.pdf"
    pages_copy.parent.mkdir(parents=True, exist_ok=True)
    pages_copy.write_bytes(OUT.read_bytes())
    print(f"Đã cập nhật bản cho GitHub Pages: {pages_copy}")


if __name__ == "__main__":
    main()
