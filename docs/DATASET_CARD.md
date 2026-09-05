# Dataset Card — VietSign Vision Dataset

## 1. Phạm Vi Dữ Liệu

Bộ dữ liệu hướng tới bài toán phát hiện và nhận dạng **52 lớp biển báo giao thông đường bộ Việt Nam** theo Quy chuẩn Kỹ thuật Quốc gia về Báo hiệu Đường bộ (QCVN 41:2019/BGTVT).
- Định dạng ảnh: JPEG / PNG độ phân giải đường phố thực tế.
- Định dạng nhãn: Tệp văn bản `.txt` hỗ trợ cả tọa độ chuẩn hóa YOLO (`class_id x_center y_center width height` $\in [0, 1]$) và tọa độ pixel tuyệt đối (`x y width height`).
- Bảng danh mục lớp: Khai báo 52 lớp tại `data/raw/classes.txt` (ID), `classes_vie.txt` (Tiếng Việt) và `classes_en.txt` (Tiếng Anh).

---

## 2. Trạng Thái Hiện Tại Trong Repository

- **Dữ liệu đính kèm**: 5 ảnh JPG mẫu thực tế và 5 tệp nhãn tương ứng tại `data/raw/images` và `data/raw/labels`.
- **Mục đích của tập mẫu**: Phục vụ **smoke testing**, kiểm thử tự động quy trình xử lý ảnh, kiểm tra tính toàn vẹn (parity test) và demo CLI.
- **Tập dữ liệu đầy đủ**: Danh sách split lịch sử tham chiếu 2.552 ảnh train và 639 ảnh test. Do giới hạn dung lượng lưu trữ repository và quyền riêng tư, toàn bộ ảnh gốc không được đính kèm trực tiếp trong commit code.
- **Khuyến nghị**: Để huấn luyện lại mô hình SVM 2 tầng hoặc tái lập chính xác các chỉ số F1 lịch sử, người sử dụng cần liên kết tập dữ liệu đầy đủ vào `data/raw/`.

---

## 3. Giao Thức Chống Rò Rỉ Dữ Liệu (Leakage-Safe Protocol)

Để bảo đảm tính độc lập tuyệt đối giữa các tập Train, Validation và Test, quy trình phân chia dữ liệu áp dụng nguyên tắc bất biến (Invariant):

1. **SHA-256 Exact Audit**: Quét mã băm toàn bộ tệp, loại bỏ hoặc gom cụm các ảnh trùng khớp 100%.
2. **pHash Near-Duplicate Grouping**: Sử dụng thuật toán Perceptual Hash với ngưỡng khoảng cách Hamming $\le 8$ để gom các khung hình gần giống nhau do xe dừng đèn đỏ hoặc di chuyển chậm.
3. **Sequence / Route Grouping**: Gom cụm các ảnh có chung tiền tố chuỗi quay video hành trình (`video_XX`, `seq_XX`).
4. **Group-Level Partitioning**: Toàn bộ các ảnh thuộc cùng một cụm (cluster) bắt buộc được gán vào **duy nhất một tập split** (`train`, `val`, hoặc `test`), triệt tiêu 100% rò rỉ thông tin sang tập kiểm thử khóa.
5. **Data Lineage Manifest**: Mọi thông tin phân chia được lưu vết minh bạch tại `data/processed/manifest.json`.

---

## 4. Các Lát Cắt Đánh Giá Rủi Ro (Evaluation Slices)

Khi đánh giá trên tập dữ liệu hoàn chỉnh, hệ thống bắt buộc báo cáo chi tiết theo các lát cắt:

1. **Kích thước biển báo trong ảnh (Size Slices)**:
   - Small: diện tích $< 32 \times 32 = 1.024$ pixels.
   - Medium: diện tích từ $1.024$ đến $9.216$ pixels ($32\times 32 - 96\times 96$).
   - Large: diện tích $> 9.216$ pixels ($> 96 \times 96$).
2. **Điều kiện ánh sáng & thời tiết**:
   - Ban ngày (Daylight) vs Ban đêm / Chập tối (Night / Low-light).
   - Nắng gắt (Chói sáng, suy giảm màu) vs Mưa / Ẩm ướt.
3. **Môi trường & Góc nhìn**:
   - Đô thị đông đúc (nhiều biển quảng cáo gây nhiễu) vs Đường quốc lộ / Nông thôn.
   - Góc nhìn trực diện (Frontal) vs Góc nghiêng phối cảnh (Oblique).
   - Không bị che khuất (Clear) vs Bị che khuất một phần (Partial occlusion bởi cành cây, xe cộ).
4. **Phân phối lớp & Độ mất cân bằng**:
   - Báo cáo số lượng mẫu (support) và F1 cho từng lớp trong 52 lớp.
   - Theo dõi đặc biệt nhóm 10 lớp có số lượng mẫu ít nhất (worst classes).

---

## 5. Kiểm Tra Tính Toàn Vẹn Tự Động

Dự án cung cấp công cụ tự động kiểm tra lỗi dataset:

```bash
python -m src.audit --output outputs/dataset-audit.json
```

Báo cáo JSON bao gồm: số lượng ảnh/nhãn, phân phối từng class_id, ảnh hỏng, nhãn rỗng, nhãn không có ảnh tương ứng (orphan labels), class_id ngoài danh mục và kiểm tra chồng chéo split.
