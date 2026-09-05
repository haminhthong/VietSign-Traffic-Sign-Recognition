# Model Card — VietSign Vision Two-Stage SVM

## 1. Tổng Quan Kiến Trúc Mô Hình

Hệ thống phân loại của VietSign Vision sử dụng kiến trúc **Học Máy Hai Tầng (2-Tier SVM)** kết hợp vector đặc trưng HOG ($1.764$ chiều):

```text
       Candidate ROI (64x64)
                 ↓
      HOG Extraction (1.764-d)
                 ↓
  Tier 1: Binary SVM (Sign vs Background)
  ├── StandardScaler + SVC (RBF Kernel, C=5.0, gamma='scale')
  ├── Huấn luyện với Hard-Negative Mining (loại trừ False Positives từ nền)
  └── Quyết định: p_sign >= 0.5 => Chấp nhận ứng viên biển báo
                 ↓ (accepted candidates)
  Tier 2: Multiclass SVM (52 Lớp Biển Báo Giao Thông)
  ├── StandardScaler + SVC (RBF Kernel, C=5.0, class_weight='balanced')
  └── Quyết định: model_score >= 0.3 => Gán nhãn biển báo chi tiết
```

---

## 2. Chuẩn Hóa Điểm Số & Hiệu Chỉnh Xác Suất (Calibration)

### 2.1. Phân Biệt `model_score` vs `calibrated_probability`

- **Điểm số thô (`model_score`)**: Khoảng cách siêu phẳng phân chia (decision function) sau khi qua hàm Sigmoid hoặc Softmax chỉ phản ánh độ tự tin tương đối, **không phải là xác suất chuẩn xác (calibrated probability)**.
- **Xác suất hiệu chỉnh (`calibrated_probability`)**: Sử dụng phương pháp Platt Scaling (`sklearn.calibration.CalibratedClassifierCV(method='sigmoid')`) được huấn luyện trên tập **Validation độc lập** để ánh xạ điểm số thô về xác suất thống kê thực tế.

### 2.2. Giao Thức Lựa Chọn Ngưỡng (Threshold Selection Protocol)

> [!IMPORTANT]
> **Nguyên tắc bất biến trong điều chỉnh ngưỡng**:
> Mọi tham số siêu tham số ($C, \gamma$), ngưỡng lọc tầng 1 (`bin_thr = 0.5`), ngưỡng phân loại tầng 2 (`multi_thr = 0.3`), và mô hình hiệu chỉnh xác suất **bắt buộc chỉ được tối ưu hóa trên tập Validation**.
> Tập Kiểm Thử Khóa (Locked Test Set) chỉ được nạp để đánh giá một lần duy nhất, tuyệt đối không dùng để điều chỉnh tham số.

---

## 3. Chỉ Số Tham Chiếu & Trạng Thái Mô Hình

### 3.1. Bảng Chỉ Số Lịch Sử (Tham Khảo)

| Tầng Phân Loại | Chỉ Số Đánh Giá | Giá Trị Lịch Sử | Ghi Chú |
|---|---|:---:|---|
| **Tier 1 (Binary SVM)** | Accuracy | 0.9680 | Sàng lọc biển báo vs vùng nền |
| | Macro F1 | 0.9550 | Cân bằng giữa True Positives và False Alarms |
| **Tier 2 (Multiclass SVM)** | Accuracy | 0.8906 | Phân loại chi tiết 52 lớp biển báo |
| | Macro F1 | 0.8025 | Bị ảnh hưởng bởi các lớp có ít mẫu |
| **End-to-End Recognition** | Accuracy | 0.9516 | Tỷ lệ nhận dạng đúng toàn pipeline |
| | Macro F1 | 0.7526 | $IoU \ge 0.5 \land class_{pred} == class_{gt}$ |

*Lưu ý*: Các chỉ số trên ghi nhận từ lần khảo sát ban đầu trên notebook. Tập repository công khai hiện tại chỉ chứa ảnh mẫu phục vụ smoke test; để tái lập các giá trị trên, cần huấn luyện lại trên toàn bộ 3.191 ảnh gốc.

---

## 4. Mục Đích Sử Dụng & Giới Hạn Kỹ Thuật

### 4.1. Mục Đích Phù Hợp

- Học tập, nghiên cứu và minh họa trực quan các kỹ thuật Classical Computer Vision và MLOps truyền thống.
- Làm mô hình chuẩn (Interpretable Baseline) để so sánh hiệu năng, tốc độ CPU và độ tin cậy với các mô hình học sâu end-to-end (YOLOv8, Faster-RCNN).
- Xử lý ảnh tĩnh hoặc camera hành trình tốc độ vừa phải trên CPU thông thường.

### 4.2. Giới Hạn & Điều Kiện Không Phù Hợp

- Không sử dụng trực tiếp để điều khiển phương tiện tự hành hoặc các hệ thống an toàn sinh mạng khi chưa có kiểm định trên phần cứng chuyên dụng.
- Hiệu năng nhận dạng sẽ suy giảm đối với các biển báo có độ phân giải siêu nhỏ ($< 16 \times 16$ pixels trong ảnh gốc) hoặc bị che khuất quá $50\%$.
- Các lớp có hình dạng tương đồng cao (ví dụ: Biển giới hạn tốc độ 40 km/h vs 50 km/h, hoặc Cấm dừng vs Cấm đỗ) cần bổ sung thêm đặc trưng màu sắc nội bộ hoặc tăng cường mẫu huấn luyện.
