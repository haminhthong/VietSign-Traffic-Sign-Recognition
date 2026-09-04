# VietSign Vision 🚦

> **Explainable Vietnamese Traffic Sign Recognition System with Classical Computer Vision & 2-Tier SVM**

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

**VietSign Vision** là hệ thống nhận dạng và phân loại 52 lớp biển báo giao thông Việt Nam sử dụng hoàn toàn các kỹ thuật **Xử lý ảnh Truyền thống (Classical Computer Vision)** kết hợp với mô hình **Học máy Hai Tầng (2-Tier SVM)**. 

Dự án được thiết kế theo tiêu chí **Clean Code**, mô-đun hóa, chú thích tiếng Việt và chạy offline trên CPU. Đây là một **interpretable classical CV baseline**: người học có thể quan sát đầu ra từng tầng thay vì coi toàn bộ pipeline như một hộp đen.

---

## 📋 Mục Lục

- [Giới Thiệu & Điểm Sáng Dự Án](#-giới-thiệu--điểm-sáng-dự-án)
- [Sơ Đồ Kiến Trúc Pipeline](#-sơ-đồ-kiến-trúc-pipeline)
- [Kết Quả Đánh Giá Hiệu Năng](#-kết-quả-đánh-giá-hiệu-năng)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Cài Đặt](#-cài-đặt)
- [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
  - [1. Chạy dòng lệnh (CLI)](#1-chạy-dòng-lệnh-cli)
  - [2. Sử dụng qua Python API](#2-sử-dụng-qua-python-api)
  - [3. Kiểm tra toàn vẹn Dataset (Audit Tool)](#3-kiểm-tra-toàn-vẹn-dataset-audit-tool)
- [Kiểm Thử & Phát Triển](#-kiểm-thử--phát-triển)
- [Định Hướng Đưa Vào CV / Portfolio](#-định-hướng-đưa-vào-cv--portfolio)
- [Tài Liệu Chi Tiết](#-tài-liệu-chi-tiết)

---

## 🌟 Giới Thiệu & Điểm Sáng Dự Án

**VietSign Vision** được phát triển để minh họa một baseline Computer Vision cổ điển có thể quan sát từng bước và không yêu cầu GPU:

1. **Hiểu sâu bản chất xử lý ảnh**: Nắm vững các thuật toán cốt lõi như lọc nhiễu Median, cân bằng độ tương phản CLAHE tự thích nghi trên kênh L (không gian màu LAB), phân đoạn màu HSV, phát hiện vùng cực trị MSER, dò biên Canny và biến đổi Hough Circles.
2. **Kiến trúc SVM Hai Tầng (2-Tier SVM Classifier)**:
   - **Tầng 1 (Binary SVM)**: Phân biệt Biển báo vs Vùng nền (Background) để triệt tiêu các báo giả (False Positives).
   - **Tầng 2 (Multiclass SVM)**: Phân loại chi tiết 52 lớp biển báo giao thông Việt Nam.
3. **Trích xuất đặc trưng HOG ($1.764$ chiều)**: Mã hóa thông tin hướng gradient từ ảnh ROI đã chuẩn hóa $64 \times 64$ pixels.
4. **CPU-only offline**: Không phụ thuộc GPU; chưa tuyên bố edge-ready cho đến khi có benchmark trên phần cứng mục tiêu.
5. **Python package có kiểm thử**: CLI, type annotations, I/O Unicode và kiểm tra tự động trên nhiều phiên bản Python.

---

## 📐 Sơ Đồ Kiến Trúc Pipeline

```text
[ Ảnh BGR Gốc ]
       │
       ▼
 1. Tiền xử lý (Task 1) ───► Median Filter + Dynamic CLAHE (LAB - Kênh L)
       │
       ▼
 2. Tạo vùng ứng viên ────► Phân đoạn HSV (Đỏ/Xanh/Vàng) + MSER + Canny Convex Hull
       │
       ▼
 3. Xác minh hình học ────► Hough Circle (Tròn) + PolyDP (Tam giác / Tứ giác)
       │
       ▼
 4. Hợp nhất & Lọc ROI ────► NMS (IoU >= 0.4) + Crop & Warp Perspective -> ROI 64x64
       │
       ▼
 5. Trích xuất HOG ───────► HOG Feature Vector (1.764 chiều)
       │
       ▼
 6. SVM Hai Tầng ────────► Tầng 1: Biển/Nền (Sign/Bg) ──► Tầng 2: 52 Lớp Biển báo
       │
       ▼
[ Bounding Boxes & Nhãn Dán Tiếng Việt ]
```

---

## 📊 Chỉ Số Lịch Sử Chưa Tái Lập

Các giá trị dưới đây được giữ lại từ lần chạy notebook cũ. Tập `test_files.txt` khi đó đã được dùng làm validation để chọn tham số/ngưỡng, vì vậy **không được xem là test benchmark độc lập**. Repository hiện cũng thiếu phần lớn dataset và model nên chưa thể tái lập các số này.

| Chỉ số Đánh Giá (Metric) | Tầng 1: Binary SVM (Sign / Background) | Tầng 2: Multiclass SVM (52 Lớp) | Pipeline End-to-End |
|---|:---:|:---:|:---:|
| Accuracy lịch sử | 0.9680 | 0.8906 | 0.9516 |
| Macro F1 lịch sử | 0.9550 | 0.8025 | 0.7526 |

Không sử dụng các giá trị này trong CV cho đến khi chạy lại quy trình train/validation/test độc lập. Xem [Model Card](docs/MODEL_CARD.md) và [Dataset Card](docs/DATASET_CARD.md).

---

## 📁 Cấu Trúc Thư Mục

```text
TSR/
├── config.yaml                  # Tệp cấu hình tập trung cho các bước pipeline
├── pyproject.toml               # Cấu hình package & công cụ phát triển (Ruff, Pytest)
├── requirements.txt             # Thư viện phụ thuộc chính
├── requirements-dev.txt         # Thư viện dùng cho môi trường dev/test
├── src/                         # Mã nguồn cốt lõi (100% Chú thích Tiếng Việt & Type Hints)
│   ├── __init__.py              # Export package & phiên bản
│   ├── __main__.py              # Entry-point chạy `python -m src`
│   ├── audit.py                 # Công cụ kiểm tra tính toàn vẹn dataset
│   ├── classifier.py            # Huấn luyện, nạp/lưu & dự đoán SVM 2 tầng
│   ├── cli.py                   # Giao diện dòng lệnh chuyên nghiệp (CLI)
│   ├── data_loader.py           # Đọc/ghi ảnh Unicode an toàn trên Windows & YAML config
│   ├── feature_extraction.py    # Trích xuất đặc trưng HOG 1.764 chiều
│   ├── hough_detection.py       # Biến đổi Hough Circles phát hiện biển tròn
│   ├── pipeline.py              # Luồng xử lý end-to-end hoàn chỉnh
│   ├── polygon_detection.py     # Dò đa giác (Tam giác & Tứ giác/Chữ nhật)
│   ├── preprocessing.py         # Lọc Median + Dynamic CLAHE
│   ├── roi_extraction.py        # Cắt ROI, Warp biến dạng & lọc NMS
│   ├── segmentation.py          # Phân đoạn dải màu HSV (Đỏ, Xanh, Vàng, Phi sắc)
│   ├── task2_union.py           # Hợp nhất ứng viên từ HSV, MSER & Canny Hull
│   └── utils.py                 # Bounding box, tính IoU & hiển thị ảnh
├── docs/                        # Tài liệu chuyên sâu
│   ├── ARCHITECTURE.md          # Chi tiết kiến trúc toán học từng bước
│   ├── DATASET_CARD.md          # Thông tin chi tiết về dataset
│   ├── MODEL_CARD.md            # Thông số huấn luyện mô hình SVM
│   └── PORTFOLIO.md             # Hướng dẫn trình bày CV & bộ câu hỏi phỏng vấn
├── notebooks/                   # 8 Notebook Jupyter khảo sát và huấn luyện 00->06
├── tools/                       # Công cụ hỗ trợ tái lập & benchmark
│   ├── reproducible_split.py    # Script phân chia train/val/test & mã băm SHA-256
│   └── benchmark.py             # Công cụ benchmark độc lập & so sánh baseline
├── tests/                       # Bộ kiểm thử tự động (28 tests)
│   ├── test_audit.py            # Test kiểm tra tính toàn vẹn dataset
│   ├── test_core.py             # Test các hàm xử lý ảnh cốt lõi
│   ├── test_edge_cases.py       # Test đường dẫn xử lý lỗi & trường hợp biên
│   ├── test_parity.py           # Test tính nhất quán giữa training và inference
│   └── test_pipeline.py         # Test HOG, SVM & Pipeline end-to-end
├── data/                        # Dữ liệu ảnh raw, interim và processed
└── outputs/                     # Thư mục lưu models (.joblib) và kết quả dự đoán
```

---

## 🛠️ Cài Đặt

Khuyến nghị môi trường **Python 3.10 – 3.12**. Không yêu cầu GPU.

### Windows (PowerShell)

```powershell
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cập nhật pip và cài đặt thư viện
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. (Tùy chọn) Cài đặt package dạng editable
python -m pip install -e ".[notebooks]"
```

### Linux / macOS

```bash
# 1. Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài đặt phụ thuộc
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Chạy dòng lệnh (CLI)

#### Chế độ Demo nhanh (Không cần nạp mô hình SVM):
Trích xuất các vùng ứng viên (Candidate ROIs) và vẽ khung hiển thị:
```bash
python -m src.cli data/raw/images/0589.jpg --detect-only
```

#### Chế độ Nhận dạng Đầy đủ (Full Recognition Pipeline):
Nhận dạng và phân loại biển báo trên 1 tệp ảnh:
```bash
python -m src.cli data/raw/images/0589.jpg
```

Nhận dạng toàn bộ ảnh trong thư mục và xuất kết quả nhãn Tiếng Anh:
```bash
python -m src.cli data/raw/images --output outputs/predictions --classes data/raw/classes_en.txt
```

Kết quả xuất ra sẽ gồm ảnh đã vẽ Bounding Box + Tên biển báo + Độ tin cậy (Confidence) và tệp `predictions.json`.

---

### 2. Sử dụng qua Python API

```python
from src.pipeline import run_pipeline_on_image, draw_detections
from src.data_loader import save_image

# 1. Chạy pipeline nhận dạng end-to-end trên 1 tệp ảnh
enhanced_img, debug_mask, detections = run_pipeline_on_image("data/raw/images/0589.jpg")

# 2. In danh sách biển báo phát hiện được
for det in detections:
    print(f"Lớp biển báo: {det['predicted_class']} | Độ tin cậy: {det['confidence']:.2f} | BBox: {det['bounding_box']}")

# 3. Vẽ nhãn và lưu ảnh kết quả
visualized = draw_detections(enhanced_img, detections)
save_image("outputs/predictions/result.jpg", visualized)
```

---

### 3. Kiểm tra toàn vẹn Dataset (Audit Tool)

Chạy công cụ kiểm tra ảnh hỏng, nhãn trống, lệch class ID hoặc thiếu ảnh train/test split:
```bash
python -m src.audit --output outputs/dataset-audit.json
```

---

## 🧪 Kiểm Thử & Phát Triển

Dự án đi kèm bộ unit tests tự động phủ rộng các hàm cốt lõi, trích xuất HOG và dự đoán mô hình:

```bash
# Chạy bộ kiểm thử tiêu chuẩn bằng unittest
python -m unittest discover -s tests -v

# Hoặc chạy qua Pytest (nếu đã cài requirements-dev.txt)
pytest tests -v
```

---

## 💼 Định Hướng Đưa Vào CV / Portfolio

Dự án này là minh chứng tuyệt vời cho năng lực lập trình Python chuyên nghiệp và tư duy Computer Vision nền tảng. Bạn có thể sử dụng các đoạn mô tả mẫu sau cho Resume/CV của mình:

> **VietSign Vision — Vietnamese Traffic Sign Recognition Pipeline**
> - Thiết kế hệ thống nhận dạng 52 lớp biển báo giao thông Việt Nam sử dụng các kỹ thuật xử lý ảnh truyền thống (**CLAHE, HSV, MSER, Canny, Hough, Polygon**) kết hợp mô hình **2-Tier SVM**.
> - Trích xuất đặc trưng **HOG ($1.764$ chiều)** từ ảnh ROI $64 \times 64$ và thiết kế quy trình train/validation/test tách biệt để đánh giá trung thực.
> - Xây dựng Python package mô-đun có CLI, dataset audit, I/O Unicode, kiểm thử tự động và CI đa phiên bản Python.

Xem chi tiết hướng dẫn trình bày CV và câu hỏi phỏng vấn tại [docs/PORTFOLIO.md](docs/PORTFOLIO.md).

---

## 📜 Tài Liệu Chi Tiết

- 📐 [Kiến trúc Toán học & Chi tiết Pipeline](docs/ARCHITECTURE.md)
- 📊 [Thông tin Chi tiết về Dataset Card](docs/DATASET_CARD.md)
- 🤖 [Thông số Huấn luyện Model Card](docs/MODEL_CARD.md)
- 💼 [Hướng dẫn Trình bày CV & Phỏng vấn Portfolio](docs/PORTFOLIO.md)

---

**Tác giả**: *VietSign Vision Team*  
**Giấy phép**: [MIT License](LICENSE)
