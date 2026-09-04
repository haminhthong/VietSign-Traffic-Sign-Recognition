# Model Card — VietSign Vision SVM

## Mô hình

Hệ thống sử dụng hai `StandardScaler + SVC`:

1. SVM nhị phân nhận HOG và quyết định biển báo/nền.
2. SVM đa lớp nhận những ROI vượt ngưỡng tầng một và dự đoán một trong 52 lớp.

Model được lưu dưới dạng Joblib tại `outputs/models/`. File model không được đính kèm trong bản repository.

## Mục đích phù hợp

- học tập và minh họa Computer Vision cổ điển;
- baseline có thể giải thích để so sánh với detector học sâu;
- thử nghiệm offline trên ảnh tĩnh.

## Không phù hợp

- điều khiển phương tiện hoặc quyết định an toàn;
- nhận dạng thời gian thực khi chưa benchmark phần cứng mục tiêu;
- dữ liệu ngoài phân phối chưa được đánh giá;
- suy luận rằng confidence SVC là xác suất đã hiệu chỉnh.

## Chỉ số tham chiếu

| Tầng | Accuracy | Macro F1 |
|---|---:|---:|
| Binary | 0,9680 | 0,9550 |
| Multiclass | 0,8906 | 0,8025 |
| End-to-end | 0,9516 | 0,7526 |

Các giá trị được lấy từ `config.yaml` của lần huấn luyện trước. Chưa thể tái lập từ bản repository vì thiếu dataset đầy đủ và model gốc.

## Yêu cầu đánh giá trước khi phát hành

- khóa tập test và kiểm tra rò rỉ dữ liệu;
- báo cáo precision/recall/F1 từng lớp và confusion matrix;
- đánh giá theo kích thước, ánh sáng, che khuất và góc nhìn;
- đo latency p50/p95 và bộ nhớ trên CPU mục tiêu;
- hiệu chỉnh confidence và chọn ngưỡng trên validation, không chọn trên test;
- lưu checksum model, cấu hình và phiên bản thư viện.
