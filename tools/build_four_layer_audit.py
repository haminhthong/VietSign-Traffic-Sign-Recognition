"""Tạo báo cáo kiểm định dự án VietSign Vision theo bốn tầng."""

from pathlib import Path

from build_improvement_report import (
    BLUE,
    GRAY,
    LIGHT_BLUE,
    PALE_BLUE,
    RED,
    add_bullet,
    add_callout,
    add_table,
    configure_sections,
    configure_styles,
    set_font,
)
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "VietSign_Vision_4_Layer_Audit.docx"


def add_status_summary(document: Document) -> None:
    add_table(
        document,
        ["Kết luận", "Số mục", "Ý nghĩa"],
        [
            ["Đạt", "3", "Có bằng chứng trực tiếp và đủ dùng trong phạm vi hiện tại."],
            ["Đạt một phần", "10", "Thiết kế hợp lý nhưng bằng chứng hoặc phạm vi còn thiếu."],
            ["Chưa đạt", "4", "Có lỗi phương pháp hoặc claim chưa được chứng minh."],
            ["Không áp dụng", "1", "Repository không cung cấp API/UI dịch vụ."],
        ],
        [1900, 1300, 6160],
    )


def add_layer(document: Document, title: str, verdict: str, rows: list[list[str]]) -> None:
    document.add_heading(title, level=1)
    add_callout(document, "Kết luận tầng", verdict, fill=PALE_BLUE, accent=BLUE)
    add_table(
        document,
        ["Trạng thái", "Tiêu chí", "Bằng chứng và kết luận", "Hành động bắt buộc"],
        rows,
        [1350, 2400, 3510, 2100],
    )


def build_document() -> None:
    document = Document()
    configure_styles(document)
    configure_sections(document)

    section = document.sections[0]
    section.header.paragraphs[0].clear()
    header_run = section.header.paragraphs[0].add_run("VIETSIGN VISION  |  FOUR-LAYER AUDIT")
    set_font(header_run, size=8.5, color=GRAY, bold=True)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(70)
    title.paragraph_format.space_after = Pt(12)
    run = title.add_run("Kiểm định dự án theo 4 tầng")
    set_font(run, size=25, color=BLUE, bold=True)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run(
        "Problem → AI/ML Correctness → Software Engineering → Production / Business Value"
    )
    subtitle.runs[0].italic = True
    subtitle.runs[0].font.size = Pt(12)

    document.add_paragraph()
    add_callout(
        document,
        "Kết luận tổng thể",
        "Dự án phù hợp làm portfolio học thuật về Classical Computer Vision, nhưng chưa đủ bằng chứng để gọi là production-grade, edge-ready hay báo cáo benchmark test. Hai lỗi ML quan trọng là dùng test_files.txt làm validation để chọn tham số/ngưỡng và chuẩn hóa trước cross-validation trong tune_svm().",
        fill=LIGHT_BLUE,
        accent=RED,
    )

    document.add_heading("Phạm vi và cách chấm", level=1)
    document.add_paragraph(
        "Audit dựa trên mã nguồn src/, notebook 00–06, config.yaml, README, Dataset/Model Card, 5 ảnh mẫu, 5 nhãn, danh sách split, 15 test và kết quả chạy công cụ audit trong workspace ngày 01/09/2026. Không coi metric trong config là đã xác minh vì thiếu dataset đầy đủ và model gốc."
    )
    add_status_summary(document)

    add_layer(
        document,
        "Tầng 1 — Problem",
        "Bài toán và luồng xử lý có logic, nhưng phạm vi thực tế hiện chỉ được chứng minh ở mức demo/offline. Một số claim đang mạnh hơn bằng chứng.",
        [
            [
                "Đạt một phần",
                "Có thực sự giải quyết đúng bài toán?",
                "Pipeline nhắm đúng phát hiện + phân loại biển báo 52 lớp. Tuy nhiên repository thiếu model và gần như toàn bộ dữ liệu, nên chỉ chứng minh được candidate detection/smoke test, chưa chứng minh nhận dạng thực tế end-to-end.",
                "Đính kèm model có checksum, dataset hợp lệ hoặc script tải; chạy benchmark khóa.",
            ],
            [
                "Đạt",
                "Input → Output có logic?",
                "Ảnh BGR → tăng cường → candidate → xác minh hình dạng → ROI → HOG → SVM hai tầng → box/lớp/confidence. Kiến trúc mô-đun và kiểu dữ liệu đầu ra nhất quán.",
                "Bổ sung schema JSON và ví dụ output thật.",
            ],
            [
                "Chưa đạt",
                "Có kiến thức/công thức AI tự bịa?",
                "Công thức HOG 1.764 chiều là đúng. Nhưng ngưỡng CLAHE theo std 50/100 và clipLimit 4/2/1 được gắn 'Vieira et al., 2024' mà không có citation; tìm kiếm nguồn học thuật không xác minh được quy tắc này. Phải xem đây là heuristic nội bộ, không phải công thức nghiên cứu đã chứng minh.",
                "Bỏ attribution Vieira hoặc bổ sung DOI/trang/công thức đúng; ablation heuristic trên validation.",
            ],
            [
                "Đạt một phần",
                "Có phần chỉ để trông giống AI?",
                "Cụm 'Explainable AI', 'production-grade', 'edge-ready' và 'Type Annotations 100%' chưa có phép đo/công cụ chứng minh. Pipeline dễ quan sát không đồng nghĩa đã triển khai phương pháp XAI.",
                "Đổi thành 'interpretable classical CV baseline'; chỉ giữ claim có artifact đo được.",
            ],
        ],
    )

    add_layer(
        document,
        "Tầng 2 — AI/ML Correctness",
        "Chưa đạt chuẩn đánh giá ML nghiêm túc. Metric hiện tại là validation đã dùng để chọn siêu tham số/ngưỡng, không phải test độc lập; còn có leakage trong helper GridSearchCV.",
        [
            [
                "Chưa đạt",
                "Data có leakage không?",
                "Có hai rủi ro xác nhận được: 04_task4_roi dùng test_files.txt làm val; 06_task6_svm chọn C và threshold trên tập này. Ngoài ra tune_svm() fit StandardScaler trên toàn bộ X_train trước khi chia fold CV.",
                "Tạo train/val/test theo ảnh nguồn/sequence; đặt StandardScaler trong sklearn Pipeline bên trong CV.",
            ],
            [
                "Chưa đạt",
                "Train/validation/test có đúng?",
                "Không có test độc lập. Danh sách test được tái sử dụng làm validation. Bản repo chỉ có 4/2.552 ảnh train và 1/639 ảnh test nên không thể tái lập.",
                "Khóa test; tuning chỉ trên train/CV + val; chạy test đúng một lần.",
            ],
            [
                "Đạt một phần",
                "Metric có phù hợp?",
                "Macro F1 phù hợp dữ liệu lệch lớp; accuracy chỉ là bổ sung. Nhưng detection cần candidate recall, precision, false positives/image và mAP theo IoU. Confidence SVC chưa calibrated.",
                "Báo cáo per-class F1, PR curve, mAP50, candidate recall, FP/image và calibration.",
            ],
            [
                "Đạt một phần",
                "Có baseline?",
                "Có so sánh NMS với không NMS và khảo sát HOG, nhưng chưa có baseline khóa cho bài toán hoàn chỉnh. Metric cũ không tái lập được.",
                "Thêm majority/random, HOG+linear SVM, HOG+RBF và detector nhẹ trên cùng split.",
            ],
            [
                "Đạt một phần",
                "Training và inference cùng preprocessing?",
                "Cùng dùng API preprocess_task1, extract_rois và extract_hog_features về nguyên tắc. Tuy vậy artifact trung gian/notebook không lưu hash config, nên có thể drift giữa lúc tạo feature và inference.",
                "Lưu manifest config/hash/version cùng feature và model; thêm parity test.",
            ],
        ],
    )

    add_layer(
        document,
        "Tầng 3 — Software Engineering",
        "Nền tảng code tốt cho portfolio: lint sạch, 15 test pass, cấu hình và I/O rõ. Khả năng tái lập đầy đủ vẫn thất bại vì thiếu model/data/license và chưa có test tích hợp end-to-end thật.",
        [
            [
                "Đạt một phần",
                "Có test edge cases?",
                "Có test ảnh rỗng gián tiếp, kernel sai, crop vượt biên, NMS, box chạm biên, thư mục thiếu, Unicode I/O và model thiếu. Chưa test ảnh hỏng, label malformed, class name lệch, model không tương thích, CLI batch lỗi từng file và output schema.",
                "Bổ sung failure-path, malformed input, compatibility và golden integration tests.",
            ],
            [
                "Đạt",
                "Hard-code/secret/path máy cá nhân?",
                "Không phát hiện secret hoặc path cá nhân trong source/config. Path model là tương đối. Danh sách font hệ điều hành là fallback có chủ đích.",
                "Thêm secret scanning CI; không nhận model Joblib từ nguồn không tin cậy.",
            ],
            [
                "Đạt một phần",
                "Người khác clone repo có chạy được?",
                "Cài package, lint, test và detect-only có thể chạy. Full recognition không chạy vì không có hai model; training không chạy vì thiếu phần lớn dataset. README còn trỏ LICENSE không tồn tại.",
                "Cung cấp release model/checksum, downloader, LICENSE và smoke workflow sạch.",
            ],
            [
                "Không áp dụng",
                "API/UI có xử lý lỗi?",
                "Repository chỉ có Python API nội bộ và CLI; không có HTTP API hoặc UI. CLI có validation cơ bản nhưng chưa phải service contract.",
                "Nếu thêm service: schema validation, status code, giới hạn upload, timeout và request ID.",
            ],
            [
                "Đạt một phần",
                "Security/privacy?",
                "Không có secret và Dataset Card đã cảnh báo quyền dữ liệu. Rủi ro còn lại: joblib có thể thực thi mã khi deserialize; chưa có checksum/model provenance; chưa có chính sách ảnh người dùng.",
                "Chỉ tải model tin cậy, phát hành SHA-256, license/provenance và retention policy.",
            ],
            [
                "Đạt một phần",
                "README có kiến trúc + cách chạy?",
                "Có kiến trúc, cài đặt, CLI/API và docs. Nhưng badge/cây thư mục ghi 13 test thay vì 15; metric được mô tả như Train/Test dù thực tế là val; claim MIT nhưng thiếu LICENSE.",
                "Sửa claim, số test, nhãn metric; thêm limitations và reproducibility rõ ở README.",
            ],
        ],
    )

    add_layer(
        document,
        "Tầng 4 — Production / Business Value",
        "Có giá trị giáo dục và CV rõ, chưa có bằng chứng sản phẩm. Không có service, load test, SLA, telemetry hay benchmark phần cứng để đánh giá 100 người dùng hoặc edge deployment.",
        [
            [
                "Chưa đạt",
                "Nếu có 100 users còn chạy?",
                "Không thể khẳng định. Không có server/concurrency model/load test; smoke detect-only mất khoảng 6–8 giây/ảnh trong môi trường kiểm tra. SVC và pipeline CPU chưa có capacity plan.",
                "Định nghĩa service, benchmark p50/p95, memory, throughput; load test 1/10/100 concurrent users.",
            ],
            [
                "Đạt một phần",
                "Demo chứng minh use case thực tế?",
                "Demo detect-only chạy trên ảnh mẫu và chứng minh candidate pipeline. Nó không chứng minh phân loại 52 lớp, độ chính xác ngoài thực tế, video hoặc vận hành edge.",
                "Thêm video/GIF trên dữ liệu chưa dùng khi tuning, kèm ground truth và failure cases.",
            ],
            [
                "Đạt",
                "Có limitation rõ ràng?",
                "Dataset Card, Model Card và báo cáo đã nêu thiếu data/model, calibration, bias và use case không an toàn. README chính vẫn cần đưa limitation lên gần phần metric.",
                "Thêm mục Limitations ngay trong README và liên kết card.",
            ],
            [
                "Đạt một phần",
                "Có business value?",
                "Giá trị mạnh nhất là portfolio: giải thích pipeline CV cổ điển, CPU-only và kỹ năng software engineering. Chưa có user persona, KPI, cost/latency target hoặc so sánh giải pháp thay thế để chứng minh sản phẩm.",
                "Chọn một use case offline cụ thể, KPI đo được và baseline chi phí/chất lượng.",
            ],
        ],
    )

    document.add_heading("Ưu tiên sửa theo thứ tự", level=1)
    priorities = [
        "P0 — Tách test độc lập; không dùng test_files.txt để tuning; sửa StandardScaler leakage trong tune_svm().",
        "P0 — Hạ/loại các claim không chứng minh: Vieira 2024, production-grade, edge-ready, benchmark test.",
        "P0 — Bổ sung dataset/model có provenance hoặc quy trình tải tái lập; thêm LICENSE thật.",
        "P1 — Chạy benchmark detection + classification với metric phù hợp và confidence calibration.",
        "P1 — Thêm baseline, config/model/data hash và parity test training-inference.",
        "P2 — Hoàn thiện demo thực tế, failure gallery, performance benchmark và security notes.",
    ]
    for item in priorities:
        add_bullet(document, item)

    document.add_heading("Checklist chấp nhận trước khi đưa vào CV", level=1)
    checklist = [
        "[ ] Metric được tạo từ test split khóa và có script tái lập.",
        "[ ] Không còn preprocessing leakage trong cross-validation.",
        "[ ] Dataset/model/license/checksum có nguồn rõ ràng.",
        "[ ] README không còn claim mạnh hơn bằng chứng.",
        "[ ] Demo full recognition chạy từ clone sạch.",
        "[ ] Có baseline và failure cases.",
        "[ ] Có benchmark latency/memory trên phần cứng được nêu.",
        "[ ] 15 test, Ruff và CI đều xanh; thêm integration test thật.",
    ]
    for item in checklist:
        add_bullet(document, item)

    document.add_heading("Nguồn bằng chứng", level=1)
    document.add_paragraph(
        "Nguồn nội bộ: config.yaml; README.md; src/classifier.py, preprocessing.py, pipeline.py; notebooks/04_task4_roi.ipynb, 05_task5_hog.ipynb, 06_task6_svm.ipynb; tests/; Dataset Card; Model Card; báo cáo audit dữ liệu ngày 01/09/2026. Nguồn ngoài dùng để đối chiếu CLAHE: các bài báo chính thống tìm được mô tả cách chọn clip-limit khác hoặc chọn thực nghiệm; không tìm thấy nguồn xác nhận attribution 'Vieira et al., 2024' cho ngưỡng 50/100 và 4/2/1."
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_document()
