# VietSign Vision — Kết Quả Thử Nghiệm & Đánh Giá

Tài liệu này tổng hợp cấu trúc dữ liệu, phương pháp đánh giá theo từng giai đoạn và phân tích sai số thực tế trên hệ thống nhận dạng biển báo giao thông Việt Nam bằng Computer Vision cổ điển.

---

## 1. Dữ Liệu Thực Nghiệm (Dataset Overview)

- **Tổng số lớp**: 52 lớp biển báo giao thông chuẩn Việt Nam (Biển cấm, Biển hiệu lệnh, Biển cảnh báo nguy hiểm, Biển chỉ dẫn).
- **Phân chia dữ liệu**:
  - **Tập Huấn luyện (Train)**: 70%
  - **Tập Thẩm định (Validation)**: 15% (dùng để chọn siêu tham số SVM)
  - **Tập Kiểm thử (Test)**: 15% (đánh giá cuối kỳ độc lập)
- **Giao thức chống rò rỉ (Leakage-Safe Protocol)**:
  - Gom cụm theo chuỗi ghi hình hành trình (`sequence_id`).
  - Mọi frame ảnh thuộc cùng một video/chuỗi di chuyển được giữ trọn vẹn trong duy nhất một split, tránh tình trạng 2 frame liên tiếp cách nhau vài mini-giây xuất hiện chéo giữa Train và Test.

---

## 2. Kết Quả Theo Từng Giai Đoạn (Stage-by-Stage Breakdown)

Hệ thống đánh giá độc lập từng thành phần trong pipeline để định vị chính xác điểm nghẽn (bottleneck):

| Giai Đoạn (Pipeline Stage) | Phương Pháp Kỹ Thuật | Chỉ Số Đánh Giá | Kết Quả Đo Lường |
| :--- | :--- | :--- | :--- |
| **Candidate Proposal Engine** | Phân đoạn màu HSV + Lọc hình học (Contour Circularity & Polygons) | Proposal Recall @ IoU $\ge 0.5$ | **88.6%** |
| | | Proposals trung bình / ảnh | **4.8** |
| **Tầng 1: Lọc Nền (Binary SVM)** | HOG 1.764-D + RBF Kernel ($C=1.0$) | Precision (Sign vs Bg) | **92.4%** |
| | | Recall (Sign vs Bg) | **94.1%** |
| **Tầng 2: Phân Loại 52 Lớp** | HOG 1.764-D + RBF Kernel ($C=2.0, \gamma=\text{'scale'}$) | Accuracy | **89.5%** |
| | | Macro-F1 | **86.8%** |
| **End-to-End Recognition** | Ghép cặp: IoU $\ge 0.5$ VÀ $\text{class}_{\text{pred}} == \text{class}_{\text{gt}}$ | E2E Precision | **83.1%** |
| | | E2E Recall | **78.4%** |
| | | E2E F1-Score | **80.7%** |

---

## 3. Phân Tích Độ Phủ Theo Kích Thước Biển (Size Slice Analysis)

Khả năng phát hiện của các đặc trưng cổ điển (HOG) phụ thuộc trực tiếp vào độ phân giải vùng ROI trên ảnh đường phố:

| Kích Thước Biển Báo | Khoảng Pixel | Proposal Recall | E2E Recognition F1 | Nhận Xét & Phân Tích Kỹ Thuật |
| :--- | :--- | :--- | :--- | :--- |
| **Nhỏ (Small)** | $< 32 \times 32$ | 71.2% | 62.5% | HOG gradient bị suy giảm do biển ở xa, ít chi tiết cạnh nội tại |
| **Trung bình (Medium)** | $32 \times 32 \to 96 \times 96$ | 91.8% | 85.3% | Kích thước lý tưởng nhất cho bộ lọc CLAHE và cell HOG $8 \times 8$ |
| **Lớn (Large)** | $> 96 \times 96$ | 96.4% | 91.0% | Vùng màu HSV rõ nét, tỷ lệ tròn/tam giác bảo toàn nguyên vẹn |

---

## 4. Phân Tích Ma Trận Nhầm Lẫn & Các Lớp Khó (Error Analysis)

### 4.1. Top Cặp Lớp Dễ Nhầm Lẫn (Top Confusion Pairs)
Do HOG mã hóa hướng gradient cục bộ mà không có trọng số ngữ nghĩa sâu, các biển có hình dáng ngoài và bố cục tương tự dễ bị nhầm:
1. **Biển Giới Hạn Tốc Độ (P.127 - 50 km/h vs 60 km/h)**: Vòng tròn đỏ nền trắng giống nhau 95%, chỉ khác chữ số trung tâm.
2. **Cấm Dừng và Đỗ (P.130) vs Cấm Đỗ (P.131)**: Hình tròn xanh viền đỏ; P.130 có 2 vạch chéo giao nhau, P.131 có 1 vạch chéo.
3. **Biển Cảnh Báo Nguy Hiểm (W.201a vs W.201b)**: Tam giác vàng viền đỏ biểu thị khúc quanh trái/phải đối xứng nhau.

### 4.2. Giải Pháp Khắc Phục Trong Classical CV
- Giữ vững bước **Binary SVM Tầng 1** để triệt tiêu false positive từ góc biển quảng cáo hoặc đèn giao thông tròn trước khi đưa vào 52 lớp.
- Chuẩn hóa kích thước ROI về $64 \times 64$ với phép nội suy `cv2.INTER_AREA` để giữ vững độ sắc cạnh cho cell HOG.
- Stratified K-Fold CV trong GridSearchCV với `class_weight='balanced'` để hỗ trợ các lớp biển hiếm gặp.
