# VietSign Vision

![CI](https://github.com/haminhthong/Vietsign-Traffic-Sign-Recognition/actions/workflows/quality.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3--1.10-F7931E?logo=scikit-learn&logoColor=white)
![Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64?logo=ruff&logoColor=111111)
![Pytest](https://img.shields.io/badge/tests-Pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-2EA44F)

VietSign Vision là hệ thống phát hiện và nhận dạng biển báo giao thông Việt Nam bằng Computer Vision truyền thống, HOG và SVM hai tầng. Hệ thống ưu tiên khả năng giải thích, chạy CPU và theo dõi được nguồn gốc của từng proposal.

## Bài toán & phạm vi ứng dụng (Problem & Scope)

### Bài toán

Với một ảnh đường phố BGR, hệ thống cần:

1. Tăng cường ảnh trong điều kiện nhiễu hoặc tương phản thấp.
2. Tạo các vùng ứng viên có khả năng là biển báo.
3. Xác minh hình học bằng hình tròn, tam giác và tứ giác.
4. Chuẩn hóa ROI, trích xuất HOG 1.764 chiều.
5. Lọc nền bằng SVM nhị phân, sau đó phân loại biển báo bằng SVM đa lớp.
6. Trả về bounding box, class ID, độ tin cậy và nguồn proposal.

### Phạm vi

- 52 class theo các file data/raw/classes*.txt.
- Ảnh đầu vào: .jpg, .jpeg, .png; đọc bằng np.fromfile + cv2.imdecode để hỗ trợ đường dẫn Unicode trên Windows.
- Inference không cần GPU; mô hình là các file Joblib do notebook Task 6 tạo ra.
- Có hai chế độ CLI: --detect-only không cần model và recognition cần cả hai model SVM.
- Repository hiện chỉ đóng gói 5 ảnh mẫu và 5 file nhãn để smoke test. Hai danh sách split trong data/raw/split_dataset/ tham chiếu bộ dữ liệu đầy đủ, vì vậy audit sẽ báo số mục split chưa có trong checkout.
- Đây là pipeline nghiên cứu/giảng dạy có thể giải thích; chưa cam kết độ trễ thời gian thực hoặc độ chính xác production trên mọi điều kiện đường phố.

## Luồng logic, luồng dữ liệu và pipeline kỹ thuật duy nhất

Sơ đồ dưới đây là luồng chuẩn chi phối cách mã nguồn, config.yaml, notebook, công cụ đánh giá và các báo cáo JSON liên kết với nhau.

```mermaid
flowchart TD
    CFG[config.yaml] --> PARAMS[load_pipeline_config<br/>Chuẩn hóa tham số]
    RAW[Ảnh mẫu hoặc full dataset<br/>data/raw/images] --> LOAD[load_image<br/>BGR ndarray]
    LABELS[YOLO/text labels<br/>data/raw/labels] --> AUDIT[src.audit<br/>Kiểm tra nhãn và ảnh]
    RAW --> AUDIT
    RAW --> SPLIT[tools/reproducible_split.py<br/>SHA-256 + pHash + sequence]
    LABELS --> SPLIT
    SPLIT --> MANIFEST[data/processed/manifest.json<br/>train/val/test lists]
    MANIFEST --> TRAIN[Notebook 00-06<br/>huấn luyện/đánh giá]
    TRAIN --> MODELS[outputs/models/<br/>svm_binary.joblib<br/>svm_multiclass.joblib]
    INPUT[CLI input<br/>một ảnh hoặc thư mục ảnh] --> LOAD
    PARAMS --> PRE[Task 1<br/>Median 3x3 + Dynamic CLAHE trên LAB-L]
    LOAD --> PRE
    PRE --> UNION[Task 2 candidate union<br/>HSV preset 1 + preset 2 nếu bật<br/>MSER + Canny convex hull]
    PARAMS --> UNION
    PRE --> SHAPE[Task 3 shape evidence<br/>Hough circle + polygon triangle/rectangle]
    PARAMS --> SHAPE
    UNION --> MERGE[Merge candidates<br/>IoU dedup + Proposal Quality Score]
    SHAPE --> MERGE
    MERGE --> ROI[Task 4 ROI<br/>validate aspect/size + warp nếu có vertices<br/>NMS]
    PARAMS --> ROI
    ROI --> HOG[Task 5<br/>resize 64x64 + HOG 9 bins<br/>8x8 cell, 2x2 block]
    PARAMS --> HOG
    HOG --> BIN[Task 6 Tier 1<br/>Binary SVM sign/background<br/>p(sign) >= bin_thr]
    MODELS --> BIN
    BIN --> MULTI[Tier 2<br/>Multiclass SVM 52 class<br/>confidence >= multi_thr]
    MODELS --> MULTI
    MULTI --> FINAL[Final NMS<br/>bounding_box + class + confidence<br/>proposal_sources + proposal_score]
    ROI --> DEBUG[Debug funnel<br/>union_components + rejected_rois]
    FINAL --> PRED[outputs/predictions/<br/>ảnh vẽ + predictions.json]
    DEBUG --> PRED
    MANIFEST --> BENCH[tools/benchmark.py<br/>proposal recall + latency + E2E nếu có model]
    MODELS --> BENCH
    RAW --> BENCH
    LABELS --> BENCH
    BENCH --> REPORT[outputs/benchmark_results.json]
    BIN --> HARD[tools/hard_negative_mining.py<br/>IoU < 0.2 với GT]
    ROI --> HARD
    HARD --> HARDREPORT[outputs/hard_negatives.json]
```

### Chi tiết từng tầng

1. **Nạp và tiền xử lý**: src.data_loader.load_image trả về ảnh BGR. preprocess_task1 áp dụng Median Filter rồi đổi sang LAB, tính độ lệch chuẩn kênh L để chọn CLAHE clipLimit 4.0, 2.0 hoặc 1.0.
2. **Candidate union**: src.task2_union.build_union_boxes chạy HSV, MSER và Canny convex hull. Khi task2.union_hsv_sets là true, HSV chạy cả preset được chọn và preset còn lại. Các box được hợp nhất bằng NMS IoU.
3. **Bằng chứng hình học**: src.hough_detection.detect_circles tạo ứng viên hình tròn; src.polygon_detection.detect_polygons tạo tam giác/tứ giác từ Canny và approxPolyDP. Các nguồn được lưu trong proposal_sources.
4. **PQS và NMS**: merge_candidates gộp proposal trùng nhau, cộng dồn nguồn bằng chứng và tính Proposal Quality Score. apply_nms ưu tiên PQS, model score và hình học.
5. **ROI/HOG**: ROI có vertices được affine/perspective warp; ROI còn lại được crop theo box. ROI phải đạt kích thước và aspect ratio trong task4, sau đó HOG chuẩn hóa về 64×64.
6. **SVM hai tầng**: Tier 1 loại background theo bin_thr; Tier 2 trả class ID và confidence theo multi_thr. Nếu không có model, chỉ dùng --detect-only.
7. **Đầu ra**: CLI lưu ảnh đã vẽ và predictions.json. Khi chạy recognition, JSON ghi cả số ROI bị loại; khi chạy detect-only, JSON giữ proposal source/score.

## Luồng dữ liệu và định dạng

### Dữ liệu đầu vào

~~~text
data/raw/
├── images/*.jpg                 # ảnh BGR mẫu
├── labels/*.txt                 # class_id cx cy w h hoặc x y w h
├── classes.txt                  # 52 tên lớp chuẩn nội bộ
├── classes_vie.txt              # tên lớp tiếng Việt dùng để vẽ nhãn
├── classes_en.txt               # tên lớp tiếng Anh
└── split_dataset/
    ├── train_files.txt
    └── test_files.txt
~~~

src.utils.read_label_boxes chấp nhận tọa độ YOLO chuẩn hóa [0, 1] hoặc tọa độ pixel. Box được kẹp vào biên ảnh trước khi dùng cho audit và benchmark.

### Dữ liệu trung gian và đầu ra

- data/processed/: manifest và split list sinh bởi tools/reproducible_split.py; thư mục bị ignore để tránh commit dataset sinh ra.
- outputs/models/: hai file Joblib bắt buộc cho recognition; bị ignore vì có thể rất lớn.
- outputs/predictions/: ảnh kết quả và predictions.json.
- outputs/benchmark_results.json: proposal metrics, size slices, latency và E2E metrics nếu model tồn tại.
- outputs/hard_negatives.json: metadata proposal nền có IoU dưới ngưỡng với ground truth.

Một detection recognition có dạng:

~~~json
{
  "bounding_box": [x, y, width, height],
  "predicted_class": 14,
  "model_score": 0.91,
  "confidence": 0.91,
  "bin_model_score": 0.98,
  "bin_confidence": 0.98,
  "proposal_sources": ["HSV", "MSER", "HOUGH_CIRCLE"],
  "proposal_score": 1.42
}
~~~

## Cấu trúc thư mục dự án (Project Structure)

~~~text
.
├── .github/workflows/quality.yml   # CI: compile, Ruff, Pytest, CLI smoke test
├── config.yaml                     # tham số runtime và split
├── pyproject.toml                  # package, dependency và tool configuration
├── requirements.txt                # môi trường đầy đủ cho notebook/tooling
├── requirements-dev.txt            # requirements.txt + pytest + Ruff
├── src/
│   ├── cli.py                      # CLI detect-only/recognition
│   ├── pipeline.py                 # orchestration và debug funnel
│   ├── data_loader.py              # config, ảnh Unicode, file listing
│   ├── preprocessing.py            # Median + CLAHE
│   ├── segmentation.py             # HSV masks
│   ├── task2_union.py              # HSV/MSER/Canny candidate union
│   ├── hough_detection.py          # Hough circle
│   ├── polygon_detection.py        # triangle/rectangle
│   ├── roi_extraction.py           # crop, warp, PQS, NMS
│   ├── feature_extraction.py       # HOG
│   ├── classifier.py                # train/tune/evaluate/load/save SVM
│   ├── audit.py                    # dataset audit CLI
│   └── utils.py                    # labels, IoU, matching, visualization
├── tools/
│   ├── reproducible_split.py        # leakage-safe split + manifest
│   ├── benchmark.py                 # proposal/funnel/E2E benchmark
│   ├── hard_negative_mining.py      # hard negative pool
│   └── build_*.py                   # tạo các báo cáo DOCX hiện có
├── tests/                           # 40 unit/integration tests
├── notebooks/                       # khảo sát và chạy Task 1-6
├── data/raw/                        # sample data và class/split metadata
├── docs/                            # báo cáo DOCX, không còn Markdown trùng README
└── outputs/                         # báo cáo runtime; model/prediction lớn bị ignore
~~~

## Hướng dẫn cài đặt

Khuyến nghị Python 3.10, 3.11 hoặc 3.12.

### Windows PowerShell

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
~~~

.[dev] là đủ cho runtime, test và lint. Extra notebooks bổ sung pandas, matplotlib, imagehash, Jupyter và các thư viện phục vụ notebook/split near-duplicate. Có thể dùng python -m pip install -r requirements-dev.txt thay thế nếu muốn cài theo file requirements.

## Hướng dẫn chạy thử nghiệm

### 1. Smoke test không cần model

~~~powershell
python -m src.cli data/raw/images/0589.jpg --detect-only --output outputs/cleanup-smoke
~~~

Kết quả gồm ảnh đã vẽ proposal và outputs/cleanup-smoke/predictions.json. Với thư mục ảnh:

~~~powershell
python -m src.cli data/raw/images --detect-only --max-results 50
~~~

### 2. Recognition đầy đủ

Trước tiên cần có:

~~~text
outputs/models/svm_binary.joblib
outputs/models/svm_multiclass.joblib
~~~

Hai model và scaler được notebook Task 6 lưu bằng src.classifier.save_model. Sau đó chạy:

~~~powershell
python -m src.cli data/raw/images/0589.jpg --output outputs/predictions --classes data/raw/classes_vie.txt
~~~

Nếu thiếu một model, CLI dừng với thông báo rõ model Tier 1/Tier 2 cần được tạo từ notebook notebooks/06_task6_svm.ipynb.

### 3. Audit dữ liệu

~~~powershell
python -m src.audit --output outputs/dataset-audit.json
~~~

Audit kiểm tra ảnh hỏng, nhãn thiếu/rỗng, orphan label, class ID ngoài danh sách, dòng nhãn sai định dạng và giao nhau giữa các split. Với checkout mẫu, trường splits.*.missing cao là có chủ đích vì split list tham chiếu full dataset chưa được đóng gói.

### 4. Tạo split chống leakage

~~~powershell
python tools/reproducible_split.py --data-dir data/raw/images --label-dir data/raw/labels --output-dir data/processed --phash-thresh 8 --train-ratio 0.7 --val-ratio 0.15 --seed 42
~~~

Tool gom exact duplicate bằng SHA-256, near-duplicate bằng pHash khi imagehash có mặt, và các frame cùng sequence trước khi chia ở cấp group.

### 5. Benchmark và hard-negative mining

~~~powershell
python tools/benchmark.py --test-list data/processed/test_files.txt --output outputs/benchmark_results.json
python tools/hard_negative_mining.py --train-list data/processed/train_files.txt --output outputs/hard_negatives.json
~~~

Benchmark luôn đo proposal metrics và latency. E2E metrics chỉ xuất hiện khi cả hai model SVM tồn tại. Hard-negative mining giữ proposal có IoU < 0.2 với mọi ground-truth box và có thể chấm thêm p(sign) nếu Tier 1 đã được nạp.

### 6. Notebook

Thứ tự notebook phản ánh pipeline: 00_dataset_exploration → 01_task1_preprocessing → 02_task2_candidate_boxes → 03_task3_shape_verification → 04_task4_roi → 05_task5_hog → 06_task6_svm. run_pipeline.ipynb là notebook chạy tổng hợp.

## Cấu hình runtime

config.yaml là nguồn tham số runtime duy nhất của pipeline online; các tool split/benchmark/mining nhận đường dẫn và tỷ lệ qua CLI, với default trùng cấu hình dự án:

- task1: kernel Median và tile grid CLAHE.
- task2: ngưỡng HSV preset 1, preset mặc định, fill holes, achromatic option và cờ hợp nhất hai preset.
- task2_union: aspect ratio/extent của HSV/MSER, Canny và IoU NMS candidate.
- task3_shape: Hough circle, triangle và rectangle.
- task4: kích thước ROI tối thiểu, aspect ratio, resize HOG và NMS cuối.
- task5: orientations, pixels per cell và cells per block của HOG.
- task6: ngưỡng Tier 1/Tier 2 và đường dẫn hai model.

Đường dẫn tương đối trong task6 luôn được quy về project root. CLI và ba tool cũng quy về project root cho các đường dẫn output mặc định, nên có thể gọi từ thư mục làm việc khác.

## Kiểm thử, lint và CI

Chạy local:

~~~powershell
python -m compileall -q src tests tools
ruff check src tests tools
python -m pytest -q
python -m src.cli --help
~~~

Workflow .github/workflows/quality.yml chạy trên Python 3.10/3.11/3.12 và thực hiện đúng bốn kiểm tra: compile src/tests/tools, Ruff cho mã nguồn + tool + test, test tự động và CLI help smoke test. Không có bước nào yêu cầu model hoặc full dataset, nên CI có thể chạy trên checkout mẫu.

## Chỉ số và giới hạn diễn giải

- Proposal Recall@IoU0.5 được tính bằng ghép box một-một, không gọi là mAP.
- E2E correct yêu cầu đồng thời IoU ≥ 0.5 và đúng class ID.
- Accuracy/F1 lịch sử trong các báo cáo cũ không được xem là chỉ số tái lập của checkout này vì model và full dataset không nằm trong repository.
- Khi chỉ có dữ liệu mẫu, hãy dùng --detect-only, audit và unit test để kiểm tra luồng kỹ thuật; muốn báo cáo classification 52 class phải cung cấp full dataset, tạo split, chạy notebook huấn luyện và sinh hai model Joblib.

## License

MIT. Xem [LICENSE](LICENSE).
