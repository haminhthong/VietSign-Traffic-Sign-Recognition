# Kiến Trúc Toàn Diện VietSign Vision

## 1. Mục Tiêu Thiết Kế & Định Vị Kỹ Thuật

VietSign Vision được thiết kế có chủ đích như một **Classical Computer Vision Baseline**:
- **Tính Diễn Giải (Explainability)**: Có thể quan sát, trích xuất và đo lường trực tiếp đầu ra của từng giai đoạn riêng biệt (mặt nạ màu, vùng MSER, biên Canny, hình dạng Hough/Poly, vector gradient HOG, điểm số SVM).
- **CPU-Oriented Offline Inference**: Chạy hoàn toàn trên vi xử lý thông thường, không phụ thuộc vào card đồ họa (GPU), phù hợp cho thiết bị nhúng và các hệ thống giám sát biên.
- **Kiểm Soát Rò Rỉ Dữ Liệu Chặt Chẽ (Leakage-Safe MLOps)**: Triển khai kiểm tra toàn vẹn mã băm SHA-256, phân nhóm near-duplicate bằng pHash và gom chuỗi video trước khi phân chia dữ liệu.

---

## 2. Kiến Trúc Canonical 8 Giai Đoạn

### Bảng Đặc Tả Luồng Dữ Liệu 8 Giai Đoạn

| Giai Đoạn | Đầu Vào | Xử Lý Cốt Lõi | Đầu Ra |
|---|---|---|---|
| **1. Data Ingestion & Provenance** | Ảnh đường phố raw | Kiểm tra đọc Unicode, giải mã nhãn YOLO/Pixel, nạp class mapping | Ảnh BGR hợp lệ, Ground Truth boxes |
| **2. Leakage-Safe Data Protocol** | Danh sách tệp ảnh | SHA-256 audit, pHash clustering ($\le 8$), Sequence grouping | Train / Val / Locked Test split + `manifest.json` |
| **3. Image Enhancement** | Ảnh BGR gốc | Lọc Median Filter ($3 \times 3$), chuyển LAB, Dynamic CLAHE trên kênh L | Ảnh tăng cường độ tương phản, giảm nhiễu hạt |
| **4. Multi-Cue Region Proposals** | Ảnh đã tăng cường | Phân đoạn HSV (Đỏ/Xanh/Vàng), MSER, Canny Hull, Hough Circles, PolyDP | Candidate boxes + Proposal Quality Score (PQS) + NMS |
| **5. ROI Normalization** | Bounding box ứng viên | Lọc kích thước/tỷ lệ, Affine/Perspective Warp Rectification, resize $64 \times 64$, tính HOG | Vector đặc trưng HOG $1.764$ chiều |
| **6. Two-Stage Classification** | Vector HOG $1.764$-d | **Tầng 1**: Binary SVM (Sign vs Bg) + Hard-Negative Mining<br>**Tầng 2**: Multiclass SVM 52 lớp | Điểm tin cậy biển báo ($p_{sign}$), Lớp dự đoán ($0..51$) |
| **7. Decision & Validation** | Điểm số hai tầng | Calibrated probability thresholding, kiểm định bằng chứng hình học | Quyết định Chấp nhận / Loại bỏ / Unknown |
| **8. Final Output & Provenance** | Detections được chấp nhận | Ghép tọa độ ảnh gốc, gắn nhãn Tiếng Việt, tổng hợp latency & proposal provenance | Bounding boxes, Vietnamese label, scores, provenance |

---

## 3. Phân Tách Kiến Trúc Offline và Online

### 3.1. Sơ Đồ Quy Trình Huấn Luyện Ngoại Tuyến (Offline Training Architecture)

```text
                     RAW ROAD DATA
                          ↓
                Data Integrity Audit (SHA-256)
                          ↓
         Near-Duplicate Grouping (pHash <= 8)
                          ↓
             Route / Sequence Grouping
                          ↓
           Train / Validation / Test
                ┌─────────┼─────────┐
                │         │         │
              TRAIN       VAL      TEST (Locked)
                │         │         │
                ↓         │         │
        Proposal Pipeline │         │
                │         │         │
                ↓         │         │
       Hard-Negative Mine │         │
       (IoU < 0.2 to GT)  │         │
                │         │         │
                ↓         │         │
       Fit Tier-1 SVM     │         │
                │         │         │
                ↓         │         │
       Fit Tier-2 SVM     │         │
                │         │         │
                └──────► Hyperparameter Tuning
                         ├── Proposal thresholds
                         ├── SVM C / gamma
                         ├── Probability calibration
                         └── Decision thresholds (bin_thr, multi_thr)
                              │
                         Freeze System & Parameters
                              │
                              └────────► TEST ONCE (Independent Benchmark)
                                          ↓
                                  Final Benchmark Report
```

### 3.2. Sơ Đồ Luồng Suy Luận Trực Tuyến (Online Inference Pipeline)

```text
Road Image / Frame
        ↓
Median (3x3) + Dynamic CLAHE (LAB - Kênh L)
        ↓
Multi-Cue Region Proposal Engine
├── HSV Color Regions (Đỏ, Xanh, Vàng)
├── MSER Extremal Regions (Vùng đồng nhất mức xám)
├── Canny Convex Hull (Dò biên hình học)
├── Hough Circle Verification (Biển tròn)
└── Polygon ApproxPolyDP (Biển tam giác, tứ giác)
        ↓
Proposal Evidence Fusion & Proposal Quality Score (PQS)
        ↓
Proposal NMS (IoU >= 0.4)
        ↓
ROI Geometry Validation (min_w=12, min_h=12, 0.4 <= AR <= 1.9)
        ↓
Affine / Perspective Normalization -> 64 × 64 Pixels
        ↓
HOG Descriptor Extraction (9 orientations, 8x8 cell, 2x2 block => 1.764-d)
        ↓
Tier-1 Binary SVM (Sign vs Background)  ──[p_sign < 0.5]──► Reject (Background)
        ↓ (p_sign >= 0.5)
Tier-2 Multiclass SVM (52 Lớp Biển Báo) ──[score < 0.3]──► Reject / Unknown
        ↓ (score >= 0.3)
Final Detections: BBox + Label Tiếng Việt + Model Score + Provenance (["HSV", "MSER", "CIRCLE"])
```

---

## 4. Thiết Kế Động Cơ Đề Xuất Vùng Đa Nguồn (Multi-Cue Proposal Engine)

### 4.1. Bằng Chứng Đa Nguồn & Điểm Chất Lượng Đề Xuất (PQS)

Thay vì xếp thứ tự ưu tiên cứng (contour > shape), hệ thống tính điểm Proposal Quality Score (PQS) cho mỗi ứng viên $i$:

$$\text{PQS}(i) = \sum_{s \in \mathcal{S}_i} w_s + \text{SynergyBonus}(|\mathcal{S}_i|) + 0.2 \cdot \text{conf}_{internal} - \text{Penalty}_{AR}$$

Trong đó:
- $\mathcal{S}_i \subseteq \{\text{HSV}, \text{MSER}, \text{CANNY\_HULL}, \text{HOUGH\_CIRCLE}, \text{POLYGON\_TRIANGLE}, \text{POLYGON\_RECTANGLE}\}$ là tập các nguồn phát hiện ra ứng viên $i$.
- Trọng số $w_{\text{HOUGH\_CIRCLE}} = 0.40$, $w_{\text{HSV}} = 0.35$, $w_{\text{TRIANGLE}} = 0.35$, $w_{\text{RECTANGLE}} = 0.30$, $w_{\text{MSER}} = 0.25$, $w_{\text{CANNY\_HULL}} = 0.20$.
- $\text{SynergyBonus}$: $+0.25$ nếu có từ 3 nguồn độc lập cùng tìm thấy, $+0.12$ nếu có 2 nguồn độc lập.
- $\text{Penalty}_{AR} = \min(|1.0 - \text{AR}| \times 0.1, 0.25)$ phạt các khung hình quá dài hoặc dẹt bất thường so với tỷ lệ chuẩn của biển báo giao thông.

### 4.2. Bảo Tồn Dấu Vết Nguồn Gốc (Proposal Provenance)

Khi hai ứng viên từ các nguồn khác nhau chồng lấp với $\text{IoU} \ge 0.45$, hệ thống hợp nhất mảng nguồn gốc `proposal_sources: ["HSV", "MSER", "HOUGH_CIRCLE"]`. Trường thông tin này được giữ nguyên qua khâu cắt ROI và đưa vào kết quả đầu ra, phục vụ khả năng giải thích (explainability) cao cấp.

---

## 5. Phân Tích Stage Funnel & Ngân Sách Sai Số (Failure Budget)

Mọi đánh giá toàn diện trên tập test độc lập được theo dõi qua phễu suy giảm (Stage Funnel):

```text
100% Ground Truth Signs
   │
   ├── [Mất mát 1: Thất bại ở Proposal Engine] ──► Biển quá nhỏ (<12px), quá mờ hoặc ngoài dải HSV
   ▼
 94% Bắt được qua Multi-Cue Proposals
   │
   ├── [Mất mát 2: Thất bại ở ROI Validation] ──► Tỷ lệ khung hình bất thường hoặc diện tích rỗng
   ▼
 91% Vượt qua lọc hình học & kích thước
   │
   ├── [Mất mát 3: Bị Tier-1 Binary SVM loại] ──► Nhận nhầm thành nền (False Negative)
   ▼
 86% Được Tier-1 công nhận là biển báo
   │
   ├── [Mất mát 4: Bị Tier-2 dự đoán sai lớp] ──► Nhầm lẫn giữa các lớp tương đồng (vd: 50 vs 60 km/h)
   ▼
 78% Nhận dạng chính xác tuyệt đối (End-to-End Correct)
```

Nhờ mô hình phễu này, kỹ sư có thể xác định chính xác nút thắt cổ chai nằm ở giai đoạn nào để tập trung cải tiến thay vì phỏng đoán.

---

## 6. Ranh Giới Mô-Đun & Quy Tắc Parity

- `src/preprocessing.py`: Không gian màu và làm mịn ảnh, độc lập với bài toán phát hiện.
- `src/segmentation.py` & `src/task2_union.py`: Sinh bounding box thô, không phụ thuộc model máy học.
- `src/roi_extraction.py`: Phụ trách nắn thẳng hình học, tính PQS và lọc NMS.
- `src/feature_extraction.py`: Trích xuất đặc trưng HOG; bắt buộc đảm bảo **Feature Parity** hoàn toàn giống nhau giữa lúc train và lúc inference (được bảo vệ bởi `tests/test_parity.py`).
- `src/classifier.py`: Huấn luyện, hiệu chỉnh xác suất, và phân tích ma trận nhầm lẫn.
- `config.yaml`: Nguồn tham số runtime duy nhất của hệ sinh thái.
