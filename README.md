# VietSign Vision 🚦

> **Explainable Vietnamese Traffic Sign Recognition System with Multi-Cue Classical Computer Vision & Two-Stage SVM**

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

**VietSign Vision** là hệ thống nhận dạng và phân loại 52 lớp biển báo giao thông Việt Nam được xây dựng hoàn toàn bằng **Computer Vision Truyền Thống (Classical Computer Vision)** kết hợp mô hình **Học Máy Hai Tầng (2-Tier SVM Classifier)**.

Khác với các phương pháp học sâu nguyên khối (end-to-end black-box detectors như YOLO), VietSign Vision ưu tiên **tính diễn giải (explainability)**, cho phép quan sát, đo lường và hiệu chỉnh độc lập từng tầng: tiền xử lý ảnh $\to$ đa nguồn đề xuất vùng (multi-cue region proposals) $\to$ nắn thẳng hình học (affine/perspective rectification) $\to$ trích xuất đặc trưng HOG ($1.764$ chiều) $\to$ bộ lọc nền âm khó (hard-negative sign filter) $\to$ phân loại chi tiết 52 lớp biển báo. Hệ thống hướng đến suy luận tối ưu trên CPU, không đòi hỏi GPU.

---

## 📋 Mục Lục

- [1. Kiến Trúc Canonical 8 Giai Đoạn](#1-kiến-trúc-canonical-8-giai-đoạn)
- [2. Điểm Sáng Kỹ Thuật (Key Innovations)](#2-điểm-sáng-kỹ-thuật-key-innovations)
- [3. Giao Thức Chống Rò Rỉ Dữ Liệu (Leakage-Safe Split)](#3-giao-thức-chống-rò-rỉ-dữ-liệu-leakage-safe-split)
- [4. Đánh Giá & Benchmark Thực Tế](#4-đánh-giá--benchmark-thực-tế)
- [5. Cấu Trúc Thư Mục](#5-cấu-trúc-thư-mục)
- [6. Cài Đặt](#6-cài-đặt)
- [7. Hướng Dẫn Sử Dụng](#7-hướng-dẫn-sử-dụng)
- [8. Kiểm Thử & CI](#8-kiểm-thử--ci)
- [9. Định Hướng Đưa Vào CV / Portfolio](#9-định-hướng-đưa-vào-cv--portfolio)
- [10. Tài Liệu Chi Tiết](#10-tài-liệu-chi-tiết)

---

## 1. Kiến Trúc Canonical 8 Giai Đoạn

Toàn bộ hệ thống trực tuyến (online runtime) tuân thủ quy trình chuẩn hóa gồm **8 giai đoạn**:

```text
1. DATA INGESTION & PROVENANCE
   Ảnh đường phố thực tế Việt Nam + Bounding-box annotations + Class IDs
                 ↓
2. LEAKAGE-SAFE DATA PROTOCOL
   Mã băm SHA-256 (exact duplicate audit)
                 ↓
   Mã băm pHash (near-duplicate grouping, Hamming distance <= 8)
                 ↓
   Gom nhóm chuỗi video / route / sequence grouping
                 ↓
   Train / Validation / Locked Test (Invariant: cùng nhóm => duy nhất 1 split)
                 ↓
3. IMAGE ENHANCEMENT
   Ảnh BGR gốc
                 ↓
   Lọc nhiễu Median Filter (kernel 3x3)
                 ↓
   Không gian màu LAB → Cân bằng độ tương phản Dynamic CLAHE trên kênh L
                 ↓
4. MULTI-CUE REGION PROPOSAL ENGINE
   Ảnh đã tăng cường
                 ↓
   ┌───────────────────┬───────────────────┬───────────────────┐
   │ HSV Color Regions │ MSER Extremal Reg │ Canny Convex Hull │
   │ (Đỏ, Xanh, Vàng)  │ (Vùng đồng nhất)  │ (Biên cạnh & bao) │
   └───────────────────┴───────────────────┴───────────────────┘
                 ↓
          Candidate Union & Bằng chứng hình học (Shape Evidence)
          ├── Hough Circles (Biển tròn: Cấm / Hiệu lệnh)
          ├── Polygon approxPolyDP 3 đỉnh (Biển tam giác: Nguy hiểm)
          └── Polygon approxPolyDP 4 đỉnh (Biển chữ nhật: Chỉ dẫn)
                 ↓
          Proposal Quality Score (PQS) Evidence Fusion & NMS
                 ↓
5. ROI NORMALIZATION
   Vùng ROI ứng viên
                 ↓
   Kiểm tra tính hợp lệ hình học (min_w=12, min_h=12, aspect ratio 0.4..1.9)
                 ↓
   Nắn thẳng hình học (Warp Affine cho tam giác, Warp Perspective cho tứ giác)
                 ↓
   Chuẩn hóa kích thước 64 × 64 pixels (cv2.INTER_AREA)
                 ↓
   Trích xuất đặc trưng HOG 1.764 chiều (9 hướng, cell 8x8, block 2x2)
                 ↓
6. TWO-STAGE CLASSIFICATION
   HOG Feature Vector (1.764 chiều)
                 ↓
   Tầng 1: Binary SVM (Sign vs Background) với Hard-Negative Mining
                 ↓  (Chỉ cho phép ứng viên vượt ngưỡng p_sign >= 0.5 đi tiếp)
   Tầng 2: Multiclass SVM (52 Lớp Biển Báo Giao Thông Việt Nam)
                 ↓
7. DECISION & VALIDATION
   Model Score / Calibrated Probability + Bằng chứng hình thái
                 ↓
   Quyết định: Chấp nhận (Accept) / Loại bỏ (Reject) / Không rõ (Unknown)
                 ↓
8. FINAL OUTPUT & PROVENANCE
   Bounding Box (x, y, w, h) + Class ID / Tên biển Tiếng Việt + Model Score
   + Dấu vết nguồn gốc đề xuất (Proposal Provenance, vd: ["HSV", "MSER", "HOUGH_CIRCLE"])
   + Thời gian xử lý từng tầng (Stage Latency)
```

---

## 2. Điểm Sáng Kỹ Thuật (Key Innovations)

1. **Multi-Cue Region Proposal Engine**: Không phụ thuộc vào một kỹ thuật phân đoạn đơn lẻ. Kết hợp dải màu HSV (bắt biển rõ màu), MSER (bắt ký hiệu/ký tự bền vững với thay đổi độ sáng), Canny Hull (bắt biển chói sáng hoặc phai màu) cùng Hough Circle và Polygon Fitting.
2. **Proposal Quality Score (PQS) & Provenance Tracking**: Thay vì dùng heuristic sắp xếp cứng (`contour > shape`), hệ thống tính điểm tin cậy tổng hợp:
   $$\text{PQS} = w_{color} S_{HSV} + w_{mser} S_{MSER} + w_{edge} S_{Canny} + w_{shape} S_{Shape} + \text{MultiCueBonus} - \text{Penalty}_{AR}$$
   Đồng thời lưu giữ mảng nguồn gốc `proposal_sources: ["HSV", "MSER", "HOUGH_CIRCLE"]` xuyên suốt đến kết quả đầu ra.
3. **Warp Rectification trước khi trích xuất HOG**: Các biển báo nghiêng góc nhìn được nắn thẳng (Affine Transform cho tam giác, Perspective Transform cho tứ giác) về mặt phẳng chuẩn trước khi tính Gradient $64 \times 64$, tăng tính bất biến với biến dạng phối cảnh.
4. **Hard-Negative Mining cho Tier-1 SVM**: Thu thập trực tiếp các false proposals từ ảnh đường phố (biển quảng cáo đỏ, đèn giao thông, góc nhà, decal xe) có $IoU < 0.2$ để huấn luyện tầng nhị phân, loại trừ triệt để báo giả.
5. **Suy luận độc lập trên CPU**: Tối ưu cho môi trường edge/offline, toàn bộ quy trình chạy với độ trễ thấp và có thể giải thích chi tiết tại từng chặng.

---

## 3. Giao Thức Chống Rò Rỉ Dữ Liệu (Leakage-Safe Split)

> [!IMPORTANT]
> **Quy tắc phân chia dữ liệu (Data Split Invariant)**:
> Trong dữ liệu camera hành trình, các khung hình liên tiếp (`frame_0100`, `frame_0101`, `frame_0102`) có mức độ tương đồng cực cao. Nếu chia ngẫu nhiên từng ảnh (Random Shuffle Split), các frame liền kề sẽ rơi vào cả Train và Test, làm biến chất kết quả kiểm định.

Quy trình chuẩn hóa trong `tools/reproducible_split.py`:
1. **SHA-256 Audit**: Nhận diện ảnh trùng lặp tuyệt đối.
2. **pHash Near-Duplicate Grouping**: Dùng Perceptual Hash (Hamming distance $\le 8$) để gom các ảnh gần trùng góc quay.
3. **Sequence / Route Grouping**: Gom các frame có chung tiền tố video/chuyến đi (`video_XX`, `seq_XX`).
4. **Group-Level Split**: Phân chia ở cấp độ **nhóm** (cluster level) thay vì từng ảnh.
5. **Lineage Manifest**: Xuất `data/processed/manifest.json` ghi nhận đầy đủ `sha256`, `phash`, `group_id`, `sequence_id`, `class_ids`, `split`.

---

## 4. Đánh Giá & Benchmark Thực Tế

### Định Nghĩa Chuẩn Các Chỉ Số

1. **Candidate Proposal Engine**:
   - **`Proposal Recall@IoU0.5`**: Tỷ lệ biển báo nhãn thật (GT) có ít nhất 1 proposal khớp với $IoU \ge 0.5$.
   - **`Proposals per Image`**: Số lượng ROI ứng viên sinh ra trung bình mỗi ảnh.
   - **`FP Proposals per Image`**: Số lượng ROI nền sinh ra mỗi ảnh.
2. **End-to-End Recognition**:
   $$\text{E2E Correct} = (\text{IoU}(pred, GT) \ge 0.5) \land (class_{pred} == class_{GT})$$
3. **Stage Funnel**: Đo lường tỷ lệ sống sót của biển báo qua từng giai đoạn:
   $$\text{GT Signs} \longrightarrow \text{Candidate Proposals} \longrightarrow \text{ROI Filter} \longrightarrow \text{Tier 1 Sign Filter} \longrightarrow \text{Tier 2 Correct Class}$$

### Trạng Thái Dataset & Báo Cáo Benchmark

- **Bản phát hành mẫu trong repository**: Gồm 5 ảnh mẫu và 5 file nhãn tại `data/raw/` phục vụ **smoke testing và kiểm thử tự động (CI)**.
- **Tái lập chỉ số phân loại 52 lớp**: Cần liên kết bộ dữ liệu đầy đủ để huấn luyện lại file model SVM (`outputs/models/`).
- **Chỉ số lịch sử từ notebook khảo sát (Tham khảo)**:
  - *Tier-1 Binary SVM*: Accuracy $\approx 0.9680$, Macro-F1 $\approx 0.9550$.
  - *Tier-2 Multiclass SVM*: Accuracy $\approx 0.8906$, Macro-F1 $\approx 0.8025$.
  - *End-to-End Pipeline*: Accuracy $\approx 0.9516$, Macro-F1 $\approx 0.7526$.
  *(Khoảng cách giữa Accuracy và Macro-F1 phản ánh sự mất cân bằng giữa các lớp hiếm, là đối tượng ưu tiên cải thiện trong các phiên bản tiếp theo).*

---

## 5. Cấu Trúc Thư Mục

```text
TSR/
├── config.yaml                     # Cấu hình tham số tập trung (ngưỡng HSV, HOG, SVM)
├── pyproject.toml                  # Thiết lập package & công cụ phát triển (Ruff, Pytest)
├── requirements.txt                # Thư viện phụ thuộc chính (OpenCV, scikit-learn, imagehash)
├── src/                            # Mã nguồn cốt lõi (100% Type Hints & Chú thích Tiếng Việt)
│   ├── __init__.py                 # Export package & phiên bản
│   ├── __main__.py                 # Entry-point chạy python -m src
│   ├── audit.py                    # Công cụ kiểm tra tính toàn vẹn dataset
│   ├── classifier.py               # Huấn luyện, hiệu chỉnh xác suất, chẩn đoán nhầm lẫn SVM
│   ├── cli.py                      # Giao diện dòng lệnh CLI chuyên nghiệp
│   ├── data_loader.py              # Đọc/ghi ảnh Unicode an toàn trên Windows
│   ├── feature_extraction.py       # Trích xuất đặc trưng HOG 1.764 chiều
│   ├── hough_detection.py          # Biến đổi Hough Circles phát hiện biển tròn
│   ├── pipeline.py                 # Luồng điều phối 8 giai đoạn hoàn chỉnh
│   ├── polygon_detection.py        # Dò đa giác (Tam giác & Tứ giác/Chữ nhật)
│   ├── preprocessing.py            # Lọc Median + Dynamic CLAHE trên kênh L
│   ├── roi_extraction.py           # Warp biến dạng, Proposal Quality Score & NMS
│   ├── segmentation.py             # Phân đoạn dải màu HSV (Đỏ, Xanh, Vàng, Phi sắc)
│   ├── task2_union.py              # Hợp nhất ứng viên từ HSV, MSER & Canny Hull
│   └── utils.py                    # Đọc nhãn YOLO, tính IoU & hiển thị kết quả
├── tools/                          # Bộ công cụ phục vụ Data/MLOps & Benchmark
│   ├── reproducible_split.py       # Phân chia dữ liệu chống rò rỉ (sequence & pHash aware)
│   ├── benchmark.py                # Công cụ benchmark độc lập (Proposal Recall, E2E, Funnel)
│   └── hard_negative_mining.py     # Khai thác mẫu âm khó cho Tier-1 SVM
├── tests/                          # Bộ kiểm thử tự động toàn diện
│   ├── test_audit.py               # Kiểm tra tính toàn vẹn dataset
│   ├── test_core.py                # Kiểm thử các thuật toán xử lý ảnh cơ sở
│   ├── test_edge_cases.py          # Kiểm thử các trường hợp biên & dữ liệu lỗi
│   ├── test_parity.py              # Kiểm thử tính nhất quán giữa training và inference
│   ├── test_pipeline.py            # Kiểm thử HOG, SVM & Pipeline end-to-end
│   ├── test_split.py               # Kiểm thử chống rò rỉ dữ liệu chuỗi và deduplication
│   └── test_benchmark.py           # Kiểm thử động cơ benchmark và ghép cặp E2E
├── data/                           # Dữ liệu ảnh raw, interim và processed
└── outputs/                        # Thư mục lưu models (.joblib) và báo cáo JSON
```

---

## 6. Cài Đặt

Khuyến nghị môi trường **Python 3.10 – 3.12**.

```powershell
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt các thư viện phụ thuộc
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 7. Hướng Dẫn Sử Dụng

### 1. Chạy dòng lệnh (CLI)

```powershell
# Chế độ trích xuất ứng viên (không cần nạp model SVM)
python -m src.cli data/raw/images/0589.jpg --detect-only

# Chế độ nhận dạng đầy đủ trên 1 ảnh
python -m src.cli data/raw/images/0589.jpg

# Nhận dạng toàn bộ thư mục ảnh và xuất kết quả
python -m src.cli data/raw/images --output outputs/predictions --classes data/raw/classes_vie.txt
```

### 2. Phân chia dữ liệu chống rò rỉ chuỗi (Leakage-Safe Split)

```powershell
python tools/reproducible_split.py --data-dir data/raw/images --label-dir data/raw/labels --output-dir data/processed --phash-thresh 8
```

### 3. Chạy công cụ Benchmark độc lập

```powershell
python tools/benchmark.py --test-list data/processed/test_files.txt --output outputs/benchmark_results.json
```

### 4. Khai thác mẫu âm khó (Hard-Negative Mining)

```powershell
python tools/hard_negative_mining.py --train-list data/processed/train_files.txt --output outputs/hard_negatives.json
```

### 5. Kiểm tra tính toàn vẹn bộ dữ liệu (Dataset Audit)

```powershell
python -m src.audit --output outputs/dataset-audit.json
```

---

## 8. Kiểm Thử & CI

Dự án sở hữu bộ kiểm thử tự động phủ kín toàn bộ các khâu từ xử lý ảnh, trích xuất đặc trưng, huấn luyện SVM đến kiểm tra chống rò rỉ chuỗi và logic benchmark:

```powershell
python -m pytest tests -v
```

---

## 9. Định Hướng Đưa Vào CV / Portfolio

Dự án là minh chứng xuất sắc cho năng lực **Classical Computer Vision nền tảng** kết hợp tư duy **Data & MLOps chuyên nghiệp**:

> **VietSign Vision — Explainable Vietnamese Traffic Sign Recognition with Multi-Cue Classical CV & Two-Stage SVM**
> - Thiết kế hệ thống nhận dạng 52 lớp biển báo giao thông Việt Nam theo kiến trúc **8 giai đoạn chuẩn hóa**, minh bạch và giải thích được từng bước mà không cần GPU.
> - Xây dựng **Multi-Cue Region Proposal Engine** tích hợp phân đoạn HSV, MSER, Canny Convex Hull, Hough Circle và Polygon Approximation; chuẩn hóa góc nghiêng bằng **Affine/Perspective Warp Rectification**.
> - Triển khai cơ chế xếp hạng **Proposal Quality Score (PQS)** và truy vết nguồn gốc đề xuất (**Proposal Provenance**) đến từng kết quả đầu ra.
> - Thiết kế giao thức phân chia dữ liệu **Leakage-Safe Split** gom cụm theo chuỗi video và thuật toán **pHash Hamming Distance**, ngăn chặn 100% rò rỉ dữ liệu giữa các khung hình liền kề.
> - Huấn luyện mô hình **Two-Stage SVM Classifier** với quy trình **Hard-Negative Mining** sàng lọc vùng nền gây báo giả, đạt chuẩn mực MLOps khắt khe.

Xem chi tiết hướng dẫn trả lời phỏng vấn chuyên sâu tại [docs/PORTFOLIO.md](docs/PORTFOLIO.md).

---

## 10. Tài Liệu Chi Tiết

- 📐 [Kiến Trúc Toán Học & Thiết Kế Hệ Thống](docs/ARCHITECTURE.md)
- 📊 [Thông Tin Bộ Dữ Liệu (Dataset Card)](docs/DATASET_CARD.md)
- 🤖 [Thông Số Mô Hình (Model Card)](docs/MODEL_CARD.md)
- 💼 [Hướng Dẫn Portfolio & Phỏng Vấn Tuyển Dụng](docs/PORTFOLIO.md)

---

**Tác giả**: *VietSign Vision Team*  
**Giấy phép**: [MIT License](LICENSE)
