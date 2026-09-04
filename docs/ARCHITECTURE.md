# Kiến trúc VietSign Vision

## Mục tiêu thiết kế

VietSign Vision ưu tiên ba thuộc tính: dễ giải thích, chạy trên CPU và cho phép đánh giá độc lập từng tầng. Đây là lựa chọn có chủ đích để minh họa năng lực xử lý ảnh, thiết kế đặc trưng và Machine Learning cổ điển thay vì chỉ gọi một mô hình end-to-end.

## Luồng dữ liệu

| Tầng | Đầu vào | Xử lý | Đầu ra |
|---|---|---|---|
| Tiền xử lý | Ảnh BGR | Median blur, CLAHE động trong LAB | Ảnh tăng cường |
| Candidate generation | Ảnh tăng cường | HSV, MSER, Canny và convex hull | Bounding box thô |
| Shape verification | Ảnh tăng cường | Hough Circle, polygon approximation | Candidate có hình dạng |
| ROI | Tất cả candidate | NMS, lọc tỷ lệ/kích thước, crop/warp | ROI chuẩn hóa |
| Feature | ROI 64×64 | HOG, 9 hướng, block 2×2 | Vector 1.764 chiều |
| Binary classifier | Vector HOG | StandardScaler + SVC | Xác suất biển/nền |
| Multiclass classifier | Candidate là biển | StandardScaler + SVC | ID một trong 52 lớp |

## Quyết định kỹ thuật

### CLAHE động

`clipLimit` thay đổi theo độ lệch chuẩn kênh L bằng một heuristic nội bộ. Quy tắc này chưa phải công thức học thuật đã được xác minh và phải được so sánh với các giá trị CLAHE cố định trên validation trước khi kết luận có lợi.

### Hợp nhất nhiều nguồn candidate

HSV có lợi thế khi màu rõ; MSER hỗ trợ vùng ổn định; Canny/Hough/polygon giúp khi màu suy giảm nhưng biên còn tốt. NMS loại các vùng trùng lặp trước khi trích xuất đặc trưng.

### SVM hai tầng

Tách bài toán biển/nền khỏi phân loại 52 lớp giúp tầng đa lớp không phải học trực tiếp toàn bộ biến thiên của nền. Đổi lại, false negative ở tầng đầu không thể được khôi phục ở tầng sau.

## Ranh giới mô-đun

- `pipeline.py` điều phối, không chứa thuật toán huấn luyện.
- Các mô-đun detection không phụ thuộc model SVM.
- `classifier.py` nhận ma trận đặc trưng, không phụ thuộc OpenCV ROI.
- `config.yaml` là nguồn tham số runtime duy nhất.
- Notebook dùng API trong `src/`, tránh sao chép logic sản phẩm.

## Khả năng tái lập

Seed mặc định là 42. Tuy nhiên, để tái lập hoàn toàn cần bổ sung dataset đầy đủ, hash dữ liệu, phiên bản model và manifest môi trường. Các chỉ số cũ trong cấu hình phải được xem là tham khảo cho đến khi chạy lại trên tập test khóa.

## Rủi ro và hướng xử lý

| Rủi ro | Ảnh hưởng | Hướng cải thiện |
|---|---|---|
| Biển nhỏ hoặc che khuất | Bỏ sót candidate | Image pyramid hoặc detector học sâu |
| Ánh sáng/màu phai | HSV kém ổn định | Color constancy, augmentation |
| Candidate quá nhiều | Tăng latency/báo giả | Tuning theo precision-recall, hard-negative mining |
| Confidence chưa hiệu chỉnh | Ngưỡng khó diễn giải | Probability calibration |
| Lệch phân phối dữ liệu | Chỉ số offline không phản ánh thực tế | Test theo điều kiện, camera và địa phương |
