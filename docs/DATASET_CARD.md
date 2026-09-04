# Dataset Card

## Phạm vi

Dự án kỳ vọng ảnh đường phố và nhãn bounding box cho 52 lớp biển báo giao thông Việt Nam. Nhãn dùng ID lớp và tọa độ YOLO chuẩn hóa hoặc tọa độ pixel dạng `x y width height`.

## Trạng thái bản repository

- 5 ảnh JPG mẫu và 5 tệp nhãn được đính kèm.
- `classes.txt`, `classes_en.txt`, `classes_vie.txt` khai báo 52 lớp.
- Danh sách split tham chiếu 2.552 ảnh train và 639 ảnh test.
- Phần lớn ảnh trong danh sách split không có trong bản repository hiện tại.

Vì vậy repository đủ cho smoke test phần xử lý ảnh nhưng không đủ để huấn luyện hoặc tái lập benchmark.

## Kiểm tra dữ liệu

```bash
python -m src.audit --output outputs/dataset-audit.json
```

Báo cáo gồm số ảnh/nhãn, phân phối lớp, ảnh hỏng, nhãn thiếu/rỗng, nhãn không có ảnh, ID lớp ngoài phạm vi và mức độ đầy đủ của split.

## Nguy cơ thiên lệch

Cần thống kê và đánh giá riêng theo:

- ngày/đêm và điều kiện thời tiết;
- đô thị/nông thôn và khu vực địa lý;
- kích thước biển trong ảnh;
- góc nhìn, che khuất và độ mờ;
- thiết bị/camera;
- tần suất từng lớp.

Không nên kết luận khả năng tổng quát hóa nếu tập test có cùng tuyến đường, video nguồn hoặc ảnh gần trùng với tập train.

## Trước khi phát hành

Cần bổ sung nguồn dữ liệu, tác giả/chủ sở hữu, giấy phép, quy trình thu thập, quy tắc annotation và chính sách loại bỏ dữ liệu nhạy cảm. Không phân phối lại dataset nếu chưa xác minh quyền sử dụng.
