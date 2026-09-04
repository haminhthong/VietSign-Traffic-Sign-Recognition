# VietSign Vision - Hướng Dẫn Trình Bày Trong CV & Portfolio

Tài liệu này hướng dẫn cách trình bày dự án **VietSign Vision** vào Resume/CV, GitHub Profile, và chuẩn bị bộ câu hỏi phỏng vấn chuyên sâu dành cho vị trí **Computer Vision Engineer / AI Software Engineer / Data Scientist**.

---

## 1. Các Dòng Mô Tả Ấn Tượng Trong CV (CV Bullet Points)

### Mẫu 1: Nhấn mạnh Kỹ thuật Computer Vision & ML (Dành cho AI / CV Engineer)
- **VietSign Vision (Vietnamese Traffic Sign Recognition System)** | *OpenCV, Scikit-Learn, Python, Pytest*
  - Xây dựng pipeline phát hiện và phân loại 52 lớp biển báo giao thông Việt Nam sử dụng các kỹ thuật xử lý ảnh truyền thống (**Median Filter, Dynamic CLAHE, HSV Segmentation, MSER, Canny Hull, Hough Circles**) kết hợp học máy **2-Tier SVM**.
  - Thiết kế trích xuất đặc trưng **HOG ($1.764$ dimensions)** từ ảnh chuẩn hóa $64 \times 64$, kết hợp lọc trùng lấp **Non-Maximum Suppression (NMS)** đạt chỉ số Macro F1 **0.955** ở tầng Binary SVM và **0.803** ở tầng Multiclass SVM.
  - Tối ưu hóa mô hình siêu nhẹ chạy hoàn toàn trên CPU (No GPU required), phát triển giao diện dòng lệnh (CLI), bộ công cụ Dataset Auditor, và bộ unit tests tự động đạt 100% test pass.

### Mẫu 2: Nhấn mạnh Tư duy Hệ thống & Sản phẩm (Dành cho Software Engineer / Embedded AI)
- **VietSign Vision - Machine Learning & Signal Processing Pipeline** | *Python, Scikit-Learn, OpenCV, YAML, GitHub Actions*
  - Thiết kế kiến trúc phần mềm dạng modular với cấu hình tập trung (`config.yaml`), hỗ trợ đường dẫn Unicode tiếng Việt trên Windows bằng `cv2.imdecode/imencode`.
  - Phát triển tính năng phân loại 2 tầng giảm thiểu tối đa báo giả (False Positives) từ các vùng nền phức tạp.
  - Xây dựng bộ công cụ kiểm thử toàn vẹn dữ liệu (Audit Tool) để phát hiện ảnh hỏng, nhãn trống, lệch nhãn YOLO, và tích hợp GitHub Actions CI tự động kiểm tra trên Python 3.10–3.12.

---

## 2. Điểm Trả Lời Phỏng Vấn Chuyên Sâu (Technical Interview Q&A)

### Q1: Vì sao bạn chọn Computer Vision Truyền thống (Classical CV) & 2-Tier SVM thay vì Deep Learning (YOLO / CNN)?
- **Trả lời**:
  1. **Khả năng quan sát (Interpretability)**: Xử lý ảnh truyền thống cho phép kiểm tra ngưỡng HSV, độ tròn Hough và biên Canny tại từng bước. Đây là pipeline dễ quan sát, không phải tuyên bố đã triển khai một phương pháp XAI riêng.
  2. **Yêu cầu Tài nguyên (Resource Efficiency)**: Hệ thống hoạt động siêu nhẹ trên CPU, không cần GPU đắt tiền, phù hợp triển khai trên thiết bị nhúng hạn chế tài nguyên (Raspberry Pi / Camera hành trình).
  3. **Chứng minh tư duy nền tảng (Fundamentals)**: Giúp minh chứng sự hiểu biết sâu sắc về bản chất tín hiệu ảnh, không gian màu, độ dốc gradient trước khi sử dụng các mô hình học sâu.

### Q2: Tại sao lại chuyển đổi không gian màu sang LAB để chạy CLAHE?
- **Trả lời**: Trong không gian BGR/RGB, độ sáng (Luminance) và màu sắc (Chrominance) bị hòa trộn vào nhau. Nếu tăng tương phản trực tiếp trên RGB sẽ làm lệch màu gốc của ảnh. Trong không gian LAB, kênh **L (Lightness)** tách biệt hoàn toàn với kênh **A** và **B** (màu sắc). Việc áp dụng **Dynamic CLAHE** trên kênh L giúp tăng cường độ sáng/tương phản mà giữ nguyên tính trung thực của màu sắc biển báo.

### Q3: Kiến trúc SVM Hai Tầng (2-Tier SVM) giải quyết vấn đề gì?
- **Trả lời**: Bước trích xuất vùng ứng viên (Task 2 & 3) sinh ra rất nhiều vùng nền (Background) bị nhận nhầm do nhiễu môi trường. 
  - **Tầng 1 (Binary SVM)** hoạt động như một bộ lọc nhiễu nhanh, loại bỏ các vùng nền không phải biển báo.
  - **Tầng 2 (Multiclass SVM)** chỉ tập trung phân loại chi tiết 52 lớp trên các vùng ROI đã được xác nhận là biển báo. Điều này giúp cân bằng tập dữ liệu và giảm tỷ lệ báo giả (False Positive Rate).

---

## 3. Checklist Hoàn Thiện Khi Đưa Lên GitHub Portfolio

- [x] Đã chuẩn hóa toàn bộ mã nguồn `src/` đạt tiêu chuẩn **Clean Code** và chú thích Tiếng Việt Google-style.
- [x] Đã thêm **Type Annotations** đầy đủ cho 100% các hàm và module.
- [x] Đã bổ sung kiểm thử cho utility, pipeline, classifier, dataset audit và split leakage.
- [x] Cập nhật tệp `README.md` chuyên nghiệp với sơ đồ kiến trúc, hướng dẫn CLI/API, và kết quả đánh giá.
- [ ] Thêm ảnh/GIF demo nhận dạng thực tế vào thư mục `docs/assets/`.
- [ ] Gắn giấy phép mã nguồn phù hợp (ví dụ: MIT License) trong tệp `LICENSE`.
