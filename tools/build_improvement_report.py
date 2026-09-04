"""Tạo báo cáo DOCX về quá trình cải tiến dự án VietSign Vision."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "VietSign_Vision_Project_Improvement_Report.docx"
ASSET_DIR = ROOT / "outputs" / "docx-assets"
DIAGRAM = ASSET_DIR / "architecture-pipeline.png"

BLUE = "2E74B5"
NAVY = "1F4D78"
LIGHT_BLUE = "E8EEF5"
PALE_BLUE = "F3F7FB"
GRAY = "5B6573"
LIGHT_GRAY = "F2F4F7"
DARK = "1F2933"
WHITE = "FFFFFF"
GREEN = "2E7D32"
AMBER = "B26A00"
RED = "B3261E"
TABLE_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin_name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin_name}"))
        if node is None:
            node = OxmlElement(f"w:{margin_name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: list[int]) -> None:
    """Đặt hình học bảng cố định để Word và LibreOffice render nhất quán."""
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table_pr = table._tbl.tblPr

    table_width = table_pr.find(qn("w:tblW"))
    if table_width is None:
        table_width = OxmlElement("w:tblW")
        table_pr.append(table_width)
    table_width.set(qn("w:w"), str(sum(widths_dxa)))
    table_width.set(qn("w:type"), "dxa")

    table_indent = table_pr.find(qn("w:tblInd"))
    if table_indent is None:
        table_indent = OxmlElement("w:tblInd")
        table_pr.append(table_indent)
    table_indent.set(qn("w:w"), str(TABLE_INDENT_DXA))
    table_indent.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            width = widths_dxa[min(index, len(widths_dxa) - 1)]
            cell.width = Inches(width / 1440)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_width = tc_pr.find(qn("w:tcW"))
            if tc_width is None:
                tc_width = OxmlElement("w:tcW")
                tc_pr.append(tc_width)
            tc_width.set(qn("w:w"), str(width))
            tc_width.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_font(run, name="Calibri", size=None, color=None, bold=None, italic=None) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Trang ")
    set_font(run, size=9, color=GRAY)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)


def set_picture_alt(paragraph, description: str) -> None:
    """Gắn alt text cho ảnh vừa chèn để tài liệu hỗ trợ trình đọc màn hình."""
    doc_pr = paragraph._p.find(".//" + qn("wp:docPr"))
    if doc_pr is not None:
        doc_pr.set("title", "Sơ đồ kiến trúc VietSign Vision")
        doc_pr.set("descr", description)


def configure_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(DARK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for style_name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 18, 10),
        ("Heading 2", 13, BLUE, 14, 7),
        ("Heading 3", 12, NAVY, 10, 5),
    ):
        style = document.styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for style_name in ("List Bullet", "List Number"):
        style = document.styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(11)
        style.paragraph_format.left_indent = Inches(0.375)
        style.paragraph_format.first_line_indent = Inches(-0.188)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.25


def configure_sections(document: Document) -> None:
    for section in document.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        section.header_distance = Inches(0.45)
        section.footer_distance = Inches(0.45)

        header = section.header
        header.is_linked_to_previous = False
        header_p = header.paragraphs[0]
        header_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        header_p.paragraph_format.space_after = Pt(0)
        header_run = header_p.add_run("VIETSIGN VISION  |  PROJECT IMPROVEMENT REPORT")
        set_font(header_run, size=8.5, color=GRAY, bold=True)

        footer = section.footer
        footer.is_linked_to_previous = False
        add_page_number(footer.paragraphs[0])


def add_bullet(document: Document, text: str, level=0) -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.left_indent = Inches(0.375 + level * 0.25)
    paragraph.paragraph_format.first_line_indent = Inches(-0.188)
    paragraph.add_run(text)


def add_number(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Number")
    paragraph.add_run(text)


def add_callout(document: Document, label: str, text: str, fill=PALE_BLUE, accent=BLUE) -> None:
    table = document.add_table(rows=1, cols=1)
    set_table_geometry(table, [TABLE_WIDTH_DXA])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(4)
    label_run = paragraph.add_run(f"{label}: ")
    set_font(label_run, size=10.5, color=accent, bold=True)
    text_run = paragraph.add_run(text)
    set_font(text_run, size=10.5, color=DARK)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def add_code_block(document: Document, lines: list[str]) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.25)
    paragraph.paragraph_format.right_indent = Inches(0.25)
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(8)
    p_pr = paragraph._p.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), LIGHT_GRAY)
    p_pr.append(shading)
    for index, line in enumerate(lines):
        run = paragraph.add_run(line)
        set_font(run, name="Consolas", size=9.5, color=DARK)
        if index < len(lines) - 1:
            run.add_break()


def add_table(
    document: Document, headers: list[str], rows: list[list[str]], widths: list[int]
) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    header_cells = table.rows[0].cells
    for index, header in enumerate(headers):
        set_cell_shading(header_cells[index], LIGHT_BLUE)
        paragraph = header_cells[index].paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(header)
        set_font(run, size=9.5, color=NAVY, bold=True)
    set_repeat_table_header(table.rows[0])

    for row_values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row_values):
            paragraph = cells[index].paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(str(value))
            set_font(run, size=9.2, color=DARK)
    set_table_geometry(table, widths)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def create_architecture_diagram() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    body_font = (
        ImageFont.truetype(str(font_path), 26) if font_path.exists() else ImageFont.load_default()
    )
    title_font = ImageFont.truetype(str(bold_path), 34) if bold_path.exists() else body_font
    small_font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else body_font

    draw.text(
        (70, 35), "VietSign Vision - Luồng suy luận end-to-end", fill="#1F4D78", font=title_font
    )
    labels = [
        ("Ảnh đầu vào", "JPG / JPEG / PNG"),
        ("Tiền xử lý", "Median blur + Dynamic CLAHE"),
        ("Sinh candidate", "HSV + MSER + Canny / Hull"),
        ("Xác minh hình dạng", "Hough Circle + Polygon"),
        ("ROI và NMS", "Lọc kích thước, crop / warp 64x64"),
        ("HOG", "Vector đặc trưng 1.764 chiều"),
        ("SVM hai tầng", "Biển / nền -> phân loại 52 lớp"),
        ("Kết quả", "Bounding box + lớp + confidence + JSON"),
    ]
    box_x, box_w, box_h = 180, 1040, 72
    start_y, gap = 100, 30
    for index, (title, detail) in enumerate(labels):
        y = start_y + index * (box_h + gap)
        fill = "#E8EEF5" if index % 2 == 0 else "#F3F7FB"
        draw.rounded_rectangle(
            (box_x, y, box_x + box_w, y + box_h), radius=14, fill=fill, outline="#2E74B5", width=3
        )
        draw.text((box_x + 25, y + 18), title, fill="#1F4D78", font=body_font)
        draw.text((box_x + 380, y + 21), detail, fill="#374151", font=small_font)
        if index < len(labels) - 1:
            center = box_x + box_w // 2
            draw.line((center, y + box_h, center, y + box_h + gap - 5), fill="#2E74B5", width=4)
            draw.polygon(
                [
                    (center - 8, y + box_h + gap - 13),
                    (center + 8, y + box_h + gap - 13),
                    (center, y + box_h + gap - 3),
                ],
                fill="#2E74B5",
            )
    image.save(DIAGRAM)


def add_cover(document: Document) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(104)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    kicker = paragraph.add_run("TECHNICAL PORTFOLIO REPORT")
    set_font(kicker, size=10, color=BLUE, bold=True)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(16)
    title.paragraph_format.space_after = Pt(8)
    title_run = title.add_run("VietSign Vision")
    set_font(title_run, size=30, color=NAVY, bold=True)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(24)
    subtitle_run = subtitle.add_run(
        "Phân tích, cải tiến và hoàn thiện dự án nhận dạng biển báo giao thông Việt Nam"
    )
    set_font(subtitle_run, size=14, color=GRAY)

    document.add_picture(str(DIAGRAM), width=Inches(5.6))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_picture_alt(
        document.paragraphs[-1],
        "Sơ đồ dọc mô tả tám bước từ ảnh đầu vào đến bounding box, lớp và confidence.",
    )

    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_before = Pt(18)
    meta_run = meta.add_run("Phiên bản báo cáo 1.0  |  30/08/2026  |  Python 3.10-3.12")
    set_font(meta_run, size=10, color=GRAY, italic=True)

    document.add_page_break()


def build_document() -> None:
    create_architecture_diagram()
    document = Document()
    configure_styles(document)
    configure_sections(document)
    add_cover(document)

    document.add_heading("Tóm tắt điều hành", level=1)
    add_callout(
        document,
        "Kết luận",
        "Dự án đã được chuyển từ một tập notebook nghiên cứu phụ thuộc đường dẫn máy cá nhân thành một Python package có cấu hình tập trung, CLI, kiểm thử, audit dữ liệu, CI và bộ tài liệu phục vụ GitHub/CV. Pipeline lõi được giữ nguyên để tránh làm sai lệch kết quả nghiên cứu ban đầu.",
    )
    document.add_paragraph(
        "VietSign Vision giải quyết bài toán phát hiện và phân loại 52 lớp biển báo giao thông Việt Nam bằng Computer Vision cổ điển. Điểm mạnh portfolio nằm ở khả năng giải thích từng tầng, chạy CPU và thể hiện đầy đủ kỹ năng xử lý ảnh, Machine Learning, thiết kế phần mềm, kiểm thử và tài liệu hóa."
    )
    add_table(
        document,
        ["Hạng mục", "Trạng thái sau cải tiến", "Bằng chứng"],
        [
            ["Chất lượng code", "Đạt", "Ruff: All checks passed"],
            ["Kiểm thử", "Đạt", "15/15 test pass"],
            ["Khả năng cài đặt", "Đạt", "Package editable + 2 console commands"],
            ["Dữ liệu mẫu", "Hợp lệ nhưng chưa đầy đủ", "5 ảnh, 5 nhãn, 16 annotation"],
            ["Model", "Chưa kèm repository", "Cần tải hoặc huấn luyện lại"],
            ["Tái lập benchmark", "Chưa hoàn chỉnh", "Thiếu dataset đầy đủ và model gốc"],
        ],
        [2300, 2600, 4460],
    )

    document.add_heading("1. Project Summary", level=1)
    document.add_heading("1.1 Bài toán", level=2)
    document.add_paragraph(
        "Hệ thống nhận ảnh đường phố, tìm các vùng có khả năng chứa biển báo, chuẩn hóa ROI, trích xuất HOG và phân loại bằng SVM hai tầng. Hướng tiếp cận cổ điển được giữ vì phù hợp mục tiêu học thuật, dễ phân tích lỗi và không cần GPU."
    )
    add_table(
        document,
        ["Thuộc tính", "Mô tả"],
        [
            ["Input", "Ảnh JPG/JPEG/PNG ở không gian màu BGR sau khi OpenCV giải mã"],
            ["Output", "Bounding box, ID/tên lớp, confidence, ảnh trực quan và JSON"],
            ["Entry point", "vietsign hoặc python -m src.cli"],
            ["Audit entry point", "vietsign-audit hoặc python -m src.audit"],
            ["Training workflow", "Notebook 00 -> 06"],
            ["Inference workflow", "src/pipeline.py"],
        ],
        [2300, 7060],
    )

    document.add_heading("1.2 Công nghệ", level=2)
    for item in (
        "OpenCV: tiền xử lý, phân đoạn, MSER, Canny, Hough, contour và trực quan hóa.",
        "scikit-image: trích xuất đặc trưng HOG.",
        "scikit-learn: StandardScaler, SVC, GridSearchCV và các metric.",
        "NumPy: biểu diễn ảnh, vector hóa và tính toán số.",
        "PyYAML/Joblib: cấu hình và lưu/nạp model.",
        "Pytest/Unittest/Ruff/GitHub Actions: chất lượng và hồi quy.",
    ):
        add_bullet(document, item)

    document.add_heading("2. Architecture", level=1)
    document.add_picture(str(DIAGRAM), width=Inches(6.35))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_picture_alt(
        document.paragraphs[-1],
        "Pipeline VietSign Vision gồm tiền xử lý, sinh candidate, xác minh hình dạng, ROI, HOG và SVM hai tầng.",
    )
    caption = document.add_paragraph("Hình 1. Pipeline suy luận của VietSign Vision")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.runs[0].italic = True
    caption.runs[0].font.size = Pt(9)

    add_table(
        document,
        ["Bước", "File/hàm chính", "Input", "Output"],
        [
            ["Tiền xử lý", "preprocessing.py / preprocess_task1", "Ảnh BGR", "Ảnh tăng cường"],
            [
                "Candidate",
                "task2_union.py / build_union_boxes",
                "Ảnh tăng cường",
                "Box từ HSV/MSER/Canny",
            ],
            [
                "Hình dạng",
                "hough_detection.py, polygon_detection.py",
                "Ảnh + gray",
                "Circle/triangle/rectangle",
            ],
            ["ROI", "roi_extraction.py / extract_rois", "Candidate", "Crop/warp hợp lệ"],
            [
                "Đặc trưng",
                "feature_extraction.py / extract_hog_features",
                "ROI 64x64",
                "HOG 1.764 chiều",
            ],
            ["Phân loại", "classifier.py / predict_proba_safe", "HOG", "Lớp + confidence"],
            [
                "Điều phối",
                "pipeline.py / run_pipeline_on_image",
                "Ảnh + config + model",
                "Detections",
            ],
        ],
        [1200, 3300, 1900, 2960],
    )

    document.add_heading("2.1 Quan hệ phụ thuộc", level=2)
    add_code_block(
        document,
        [
            "src.cli",
            "  -> src.data_loader",
            "  -> src.pipeline",
            "       -> preprocessing / segmentation / task2_union",
            "       -> hough_detection / polygon_detection",
            "       -> roi_extraction / feature_extraction / classifier",
            "src.audit -> data_loader + utils",
            "notebooks -> API công khai trong src",
        ],
    )

    document.add_heading("3. Problems Found", level=1)
    document.add_paragraph(
        "Bảng dưới tổng hợp các vấn đề nổi bật ở trạng thái ban đầu và mức độ ảnh hưởng. Mức độ được đánh giá theo khả năng làm dự án không chạy trên máy khác, sai dữ liệu hoặc khó bảo trì."
    )
    add_table(
        document,
        ["Mức độ", "Vấn đề", "Ảnh hưởng", "Cách xử lý"],
        [
            [
                "Critical",
                "Đường dẫn model tuyệt đối của máy cũ",
                "Không thể chạy trên máy khác",
                "Chuyển sang đường dẫn tương đối theo project root",
            ],
            [
                "High",
                "OpenCV không đọc đường dẫn Unicode Windows",
                "Ảnh hợp lệ bị báo hỏng",
                "Dùng np.fromfile + cv2.imdecode; thêm regression test",
            ],
            [
                "High",
                "Thiếu model và dataset đầy đủ",
                "Không tái lập benchmark",
                "Dataset/Model Card và audit minh bạch",
            ],
            [
                "High",
                "Không có CLI",
                "Người dùng phải mở notebook",
                "Thêm CLI một ảnh/thư mục và detect-only",
            ],
            [
                "Medium",
                "Thiếu validation đầu vào",
                "Lỗi OpenCV khó hiểu",
                "Kiểm tra ảnh, kernel, tile, training matrix",
            ],
            [
                "Medium",
                "Không có test/CI/linter",
                "Dễ tái phát lỗi",
                "15 test, Ruff, GitHub Actions 3 phiên bản Python",
            ],
            [
                "Medium",
                "README có thông tin không nhất quán",
                "Khó cài và đánh giá",
                "Viết lại README và liên kết tài liệu",
            ],
            [
                "Low",
                "Import thừa và naming mơ hồ",
                "Giảm khả năng đọc",
                "Ruff auto-fix và đổi tên biến",
            ],
        ],
        [1250, 2700, 2700, 2710],
    )

    document.add_heading("4. Technical Debt và Code Smell", level=1)
    add_table(
        document,
        ["Hạng mục", "Impact", "Effort", "Risk", "Priority"],
        [
            ["Thiếu toàn bộ dữ liệu và model", "Rất cao", "Cao", "Cao", "P0"],
            ["Benchmark chưa tái lập", "Rất cao", "Trung bình", "Cao", "P0"],
            [
                "Candidate generation tạo nhiều false positive",
                "Cao",
                "Trung bình",
                "Trung bình",
                "P1",
            ],
            ["Confidence SVC chưa calibration", "Cao", "Trung bình", "Trung bình", "P1"],
            ["Notebook chứa logic thí nghiệm dài", "Trung bình", "Cao", "Trung bình", "P2"],
            ["Chưa có benchmark latency chuẩn", "Trung bình", "Thấp", "Thấp", "P2"],
            ["Chưa có giao diện web", "Thấp", "Trung bình", "Thấp", "P3"],
        ],
        [3000, 1500, 1500, 1500, 1860],
    )
    add_callout(
        document,
        "Nguyên tắc ưu tiên",
        "Không chuyển ngay sang kiến trúc phức tạp hoặc framework lớn. Giá trị cao nhất hiện tại là hoàn thiện dữ liệu, model, benchmark và minh chứng trực quan trước khi mở rộng sản phẩm.",
        fill="FFF7E6",
        accent=AMBER,
    )

    document.add_heading("5. Kế hoạch cải tiến đã áp dụng", level=1)
    phases = [
        (
            "Phase 1 - Safe Cleanup",
            "Chuẩn hóa import, naming, docstring, cấu hình và đường dẫn; giữ API notebook.",
        ),
        (
            "Phase 2 - Structural Refactoring",
            "Tách CLI, audit dataset, I/O Unicode và package entry points.",
        ),
        ("Phase 3 - Reliability", "Validation, lỗi rõ nghĩa, test happy/edge/failure path và CI."),
        (
            "Phase 4 - Performance",
            "Tắt HOG visualization khi inference, batch feature matrix, cache font.",
        ),
        (
            "Phase 5 - Documentation",
            "README, architecture, portfolio guide, Dataset Card, Model Card và báo cáo này.",
        ),
    ]
    for title, description in phases:
        document.add_heading(title, level=2)
        document.add_paragraph(description)

    document.add_heading("6. Changes Made", level=1)
    document.add_heading("6.1 Bug và portability", level=2)
    for item in (
        "Sửa đường dẫn model để tính từ project root.",
        "Sửa I/O ảnh Unicode bằng imdecode/imencode; không còn phụ thuộc encoding của Windows shell.",
        "Cho phép x2/y2 của bounding box bằng chiều rộng/chiều cao ảnh theo quy ước exclusive.",
        "Cắt bounding box hình tròn theo biên ảnh để tránh ROI vượt kích thước.",
        "Kiểm tra định dạng model Joblib trước khi sử dụng.",
        "Ngăn GridSearchCV chạy khi một lớp có ít hơn hai mẫu.",
    ):
        add_bullet(document, item)

    document.add_heading("6.2 Tính năng mới", level=2)
    for item in (
        "CLI nhận một ảnh hoặc thư mục, hỗ trợ tên lớp Việt/Anh, xuất ảnh và JSON.",
        "Chế độ --detect-only chạy không cần SVM để smoke test pipeline detection.",
        "Giới hạn --max-results và ghi processing_time_ms.",
        "Dataset audit thống kê phân phối lớp, ảnh hỏng, nhãn thiếu/rỗng, orphan label, class ID và split.",
        "Python package với hai lệnh vietsign và vietsign-audit.",
        "CI chạy compile, Ruff và Pytest trên Python 3.10, 3.11, 3.12.",
    ):
        add_bullet(document, item)

    document.add_heading("6.3 Làm sạch và loại bỏ dư thừa", level=2)
    for item in (
        "Loại bỏ load_processed_image và hai hàm containment không còn nơi sử dụng.",
        "Xóa implementation IoU trùng trong task2_union; toàn dự án dùng compute_iou từ utils.",
        "Bỏ lần resize ROI trung gian vì extract_hog_features đã chuẩn hóa kích thước.",
        "Nạp hai model SVM một lần trước vòng lặp CLI thay vì lặp lại theo từng ảnh.",
        "Chuẩn hóa 19 tệp bằng Ruff formatter; import, whitespace và style check đều sạch.",
    ):
        add_bullet(document, item)

    document.add_heading("7. Files Changed", level=1)
    add_table(
        document,
        ["File/nhóm", "Thay đổi", "Lý do"],
        [
            ["config.yaml", "Đường dẫn model tương đối", "Portable"],
            ["src/data_loader.py", "Project root ổn định, Unicode I/O", "Chạy đúng trên Windows"],
            [
                "src/pipeline.py",
                "Resolve path, validation, HOG nhanh, vẽ candidate",
                "Inference rõ ràng",
            ],
            ["src/classifier.py", "Validation training/model, tạo thư mục", "Reliability"],
            ["src/cli.py", "Batch CLI, detect-only, JSON, timing", "Usability"],
            ["src/audit.py", "Audit dataset", "Data quality"],
            ["tests/", "15 regression/unit tests", "Ngăn lỗi quay lại"],
            ["pyproject.toml", "Metadata, dependencies, entry points, Ruff/Pytest", "Packaging"],
            [".github/workflows/quality.yml", "CI đa phiên bản", "Quality gate"],
            [
                "README.md + docs/",
                "Hướng dẫn, kiến trúc, model/data/portfolio cards",
                "Onboarding và CV",
            ],
        ],
        [2500, 3430, 3430],
    )

    document.add_heading("8. Testing và Quality Gates", level=1)
    add_code_block(
        document,
        [
            "Ruff:       All checks passed",
            "Pytest:     15 passed",
            "Compile:    python -m compileall -q src tests",
            "Package:    vietsign-vision 1.0.0 installed editable",
            "CLI:        vietsign --help passed",
            "Audit CLI:  vietsign-audit passed",
            "Smoke test: detect-only trên ảnh 0589.jpg passed",
        ],
    )
    document.add_heading("8.1 Phạm vi test", level=2)
    for item in (
        "IoU với box giao nhau và không giao nhau.",
        "Crop ROI vượt biên ảnh.",
        "Validation kích thước ROI và median kernel.",
        "NMS loại box chồng lấn.",
        "Nhãn YOLO chạm đúng biên ảnh.",
        "Thư mục ảnh không tồn tại.",
        "Đọc/ghi ảnh với tên tiếng Việt.",
        "Audit dữ liệu mẫu và nhận biết archive split chưa đầy đủ.",
    ):
        add_bullet(document, item)

    document.add_heading("9. Performance", level=1)
    add_table(
        document,
        ["Tối ưu", "Trước", "Sau", "Lợi ích / trade-off"],
        [
            [
                "HOG visualization",
                "Luôn tạo ảnh HOG",
                "Tắt ở inference",
                "Giảm CPU/memory; notebook vẫn giữ visualize=True",
            ],
            [
                "Feature inference",
                "Danh sách ROI xử lý rồi dự đoán",
                "Gộp thành NumPy matrix",
                "SVC/scaler nhận batch",
            ],
            ["Font detection", "Tìm font mỗi lần", "Cache theo size", "Giảm I/O khi vẽ nhiều ảnh"],
            [
                "Model loading",
                "Nạp lại theo từng ảnh CLI",
                "Nạp một lần cho toàn bộ batch",
                "Giảm I/O và thời gian khởi tạo model",
            ],
        ],
        [2200, 2200, 2200, 2760],
    )
    add_callout(
        document,
        "Đo đạc hiện tại",
        "Smoke test detect-only trên ảnh 960x540 mất khoảng 6-8 giây trong môi trường kiểm tra. Đây không phải benchmark chuẩn; cần warm-up và báo cáo p50/p95 trên phần cứng mục tiêu.",
    )

    document.add_heading("10. Security và Robustness", level=1)
    for item in (
        "Không phát hiện API key, mật khẩu hoặc secret hard-code trong source.",
        "Đường dẫn input được xử lý bằng pathlib; không dựng shell command từ dữ liệu người dùng.",
        "Joblib là unsafe deserialization với file không tin cậy; chỉ nạp model từ nguồn kiểm soát và nên công bố checksum.",
        "CLI kiểm tra file/thư mục, ảnh rỗng và lỗi ghi output.",
        "Không commit model lớn, dữ liệu processed, cache, virtual environment hoặc output tạm.",
        "Cần bổ sung LICENSE và xác minh giấy phép dataset trước khi public GitHub.",
    ):
        add_bullet(document, item)

    document.add_heading("11. Kết quả trước và sau", level=1)
    add_table(
        document,
        ["Tiêu chí", "Trước", "Sau"],
        [
            ["Chạy trên máy khác", "Model path hard-code", "Relative path + Unicode-safe I/O"],
            ["Entry point", "Notebook", "Notebook + Python API + CLI"],
            ["Kiểm thử", "Không có", "15 tests"],
            ["Lint", "Không cấu hình", "Ruff pass"],
            ["CI", "Không có", "GitHub Actions 3 Python versions"],
            ["Data quality", "Kiểm tra rời rạc", "Audit CLI + JSON"],
            ["Tài liệu", "README nộp bài", "README portfolio + 5 tài liệu chuyên sâu"],
            [
                "Minh bạch model",
                "Chỉ số không có cảnh báo",
                "Model/Dataset Card nêu giới hạn tái lập",
            ],
        ],
        [2200, 3300, 3860],
    )

    document.add_heading("12. Remaining Issues", level=1)
    issues = [
        ("P0", "Bổ sung dataset đầy đủ", "Không thể train hoặc xác minh benchmark với 5 ảnh mẫu."),
        (
            "P0",
            "Phát hành model có checksum",
            "Inference end-to-end chưa chạy nếu không tải model.",
        ),
        (
            "P0",
            "Benchmark khóa",
            "Cần tái lập accuracy, macro F1, confusion matrix và leakage audit.",
        ),
        (
            "P1",
            "Giảm false positive candidate",
            "Detect-only vẫn sinh nhiều ROI từ cảnh nền phức tạp.",
        ),
        (
            "P1",
            "Calibration confidence",
            "Ngưỡng SVC hiện không đại diện xác suất được calibration.",
        ),
        (
            "P2",
            "Tương thích scikit-learn",
            "SVC(probability=True) bị deprecate từ 1.9; cần migration có benchmark.",
        ),
        (
            "P2",
            "Logging chuẩn",
            "CLI còn dùng print có chủ đích; production nên dùng logging có level.",
        ),
        ("P3", "Web demo", "Chỉ nên làm sau khi model và benchmark ổn định."),
    ]
    add_table(
        document, ["Ưu tiên", "Vấn đề", "Lý do"], [list(row) for row in issues], [1250, 2800, 5310]
    )

    document.add_heading("13. Recommended Next Steps", level=1)
    next_steps = [
        "Khôi phục toàn bộ dataset, chạy vietsign-audit và lưu báo cáo làm artifact CI.",
        "Cố định train/validation/test split theo ảnh nguồn; kiểm tra ảnh gần trùng và data leakage.",
        "Huấn luyện lại hai SVM từ môi trường sạch, lưu config, seed, dependency lock và checksum model.",
        "Xuất confusion matrix, per-class F1, candidate precision/recall và báo cáo lỗi theo điều kiện ảnh.",
        "Thực hiện hard-negative mining để giảm vùng nền bị nhận nhầm.",
        "Calibration confidence trên validation và chọn threshold bằng mục tiêu precision/recall rõ ràng.",
        "Benchmark latency p50/p95, peak memory và throughput trên CPU mục tiêu.",
        "Tạo GIF/video demo ngắn và ảnh before/after cho README/LinkedIn.",
        "So sánh baseline với YOLO nano hoặc MobileNet-SSD, giải thích trade-off thay vì thay thế mù quáng.",
        "Chọn LICENSE, bổ sung tác giả/LinkedIn/GitHub và tạo release v1.0.0.",
    ]
    for step in next_steps:
        add_number(document, step)

    document.add_heading("14. Hướng trình bày trong CV", level=1)
    add_callout(
        document,
        "Tên dự án đề xuất",
        "VietSign Vision - Explainable Vietnamese Traffic Sign Recognition",
    )
    document.add_heading("Bullet tiếng Việt", level=2)
    for item in (
        "Thiết kế pipeline Computer Vision 6 giai đoạn kết hợp CLAHE, HSV, MSER, Canny, Hough Transform, HOG và SVM hai tầng cho 52 lớp biển báo Việt Nam.",
        "Xây dựng Python package mô-đun, CLI xử lý hàng loạt, Unicode-safe I/O, audit dữ liệu, JSON output, 15 kiểm thử và CI đa phiên bản Python.",
        "Ghi nhận kết quả tham chiếu macro F1 0,955 ở tầng biển/nền và 0,803 ở tầng đa lớp; công khai giới hạn tái lập do repository mẫu chưa chứa dataset/model đầy đủ.",
    ):
        add_bullet(document, item)
    document.add_heading("Bullet tiếng Anh", level=2)
    for item in (
        "Designed an explainable six-stage computer vision pipeline combining CLAHE, HSV segmentation, MSER, Canny, Hough Transform, HOG, and two-stage SVM classification for 52 Vietnamese traffic sign classes.",
        "Built modular Python components, Unicode-safe image I/O, batch CLI, dataset auditing, structured JSON output, automated tests, and multi-version CI.",
    ):
        add_bullet(document, item)
    add_callout(
        document,
        "Lưu ý trung thực",
        "Chỉ dùng chỉ số trong CV khi có thể cung cấp model, log, notebook hoặc release tái lập kết quả. Không mô tả hệ thống là production-ready hay real-time khi chưa benchmark và đánh giá an toàn.",
        fill="FFF2F0",
        accent=RED,
    )

    document.add_heading("15. Hướng dẫn vận hành", level=1)
    document.add_heading("15.1 Cài đặt", level=2)
    add_code_block(
        document,
        [
            "python -m venv .venv",
            ".venv\\Scripts\\Activate.ps1",
            "python -m pip install -r requirements-dev.txt",
            "python -m pip install -e .",
        ],
    )
    document.add_heading("15.2 Chạy demo và audit", level=2)
    add_code_block(
        document,
        [
            "vietsign data/raw/images/0589.jpg --detect-only --max-results 15",
            "vietsign-audit --output outputs/dataset-audit.json",
            "ruff check src tests",
            "pytest",
        ],
    )
    document.add_heading("15.3 Chạy recognition đầy đủ", level=2)
    document.add_paragraph(
        "Đặt svm_binary.joblib và svm_multiclass.joblib trong outputs/models, sau đó bỏ cờ --detect-only. Nếu thiếu model, CLI dừng với thông báo chỉ rõ notebook huấn luyện cần chạy."
    )

    document.add_heading("16. Checklist trước khi public", level=1)
    checklist = [
        "Dataset source và license đã được ghi rõ.",
        "Model có checksum và cách tải ổn định.",
        "Benchmark tái lập trên test split khóa.",
        "README có ảnh/GIF demo và phần kết quả trung thực.",
        "Ruff, Pytest và CI đều xanh.",
        "Không có secret, path cá nhân hoặc output tạm.",
        "Đã chọn LICENSE và bổ sung thông tin tác giả.",
        "Đã tạo Git tag/release và liên kết từ CV/LinkedIn.",
    ]
    for item in checklist:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.add_run("[ ] ").bold = True
        paragraph.add_run(item)

    document.add_heading("Phụ lục A - Chỉ số tham chiếu", level=1)
    add_table(
        document,
        ["Metric", "Giá trị", "Trạng thái"],
        [
            ["Binary accuracy", "0,96796", "Tham chiếu từ config.yaml"],
            ["Binary macro F1", "0,95502", "Tham chiếu từ config.yaml"],
            ["Multiclass accuracy", "0,89064", "Tham chiếu từ config.yaml"],
            ["Multiclass macro F1", "0,80246", "Tham chiếu từ config.yaml"],
            ["End-to-end accuracy", "0,95155", "Tham chiếu từ config.yaml"],
            ["End-to-end macro F1", "0,75264", "Tham chiếu từ config.yaml"],
        ],
        [3000, 1800, 4560],
    )
    document.add_paragraph(
        "Nguồn nội bộ: config.yaml, README.md, mã nguồn src/, tests/, pyproject.toml, GitHub Actions và kết quả chạy kiểm thử trong workspace ngày 30/08/2026."
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_document()
