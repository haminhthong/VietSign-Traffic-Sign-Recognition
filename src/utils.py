"""Module Tiện Ích Đánh Giá & Hiển Thị (Utilities & Visualization).

Cung cấp các hàm tính chỉ số IoU (Intersection over Union), đọc file nhãn nhị phân/YOLO,
lọc chứa nhau (containment), ghép cặp bounding box để tính Precision/Recall, và hiển thị ảnh.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import cv2
import numpy as np

from src.data_loader import load_image


def show_images(
    images: List[np.ndarray], titles: List[str], figsize: Tuple[int, int] = (18, 5)
) -> None:
    """Hiển thị nhiều hình ảnh song song kèm tiêu đề bằng Matplotlib.

    Args:
        images (List[np.ndarray]): Danh sách các ảnh BGR.
        titles (List[str]): Danh sách tiêu đề tương ứng.
        figsize (Tuple[int, int]): Kích thước khung hình vẽ Matplotlib.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_boxes(
    img_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ danh sách các hộp (x, y, w, h) lên ảnh BGR.

    Args:
        img_bgr (np.ndarray): Ảnh BGR gốc.
        boxes (List[Tuple[int, int, int, int]]): Danh sách hộp dạng (x, y, w, h).
        color (Tuple[int, int, int]): Màu khung (BGR). Mặc định xanh lá.
        thickness (int): Độ dày đường vẽ.

    Returns:
        np.ndarray: Ảnh mới đã vẽ khung.
    """
    vis = img_bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    return vis


def read_label_boxes(
    label_path: Union[str, Path], image_shape: Tuple[int, ...]
) -> List[Dict[str, Any]]:
    """Đọc ground-truth bounding box từ tệp nhãn (Hỗ trợ chuẩn YOLO chuẩn hóa [0..1] hoặc Pixel tuyệt đối).

    Args:
        label_path (Union[str, Path]): Đường dẫn tệp nhãn text.
        image_shape (Tuple[int, ...]): Kích thước ảnh (H, W, C).

    Returns:
        List[Dict[str, Any]]: Danh sách các từ điển thông tin hộp GT:
            {"x1", "y1", "x2", "y2", "w", "h", "area", "class"}.
    """
    p = Path(label_path)
    img_h, img_w = image_shape[:2]
    if not p.is_file():
        return []

    boxes: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.replace(",", " ").split()
        try:
            nums = [float(x) for x in parts]
        except ValueError:
            continue

        cls_id = None
        if len(nums) >= 5:
            cls_id = int(nums[0])
            a, b, c, d = nums[1:5]
        elif len(nums) == 4:
            a, b, c, d = nums
        else:
            continue

        # Nếu là tọa độ chuẩn hóa YOLO [0.0 .. 1.0]
        if max(a, b, c, d) <= 1.0:
            cx, cy = a * img_w, b * img_h
            w, h = c * img_w, d * img_h
            x1, y1 = cx - w / 2.0, cy - h / 2.0
        else:
            x1, y1, w, h = a, b, c, d

        x2, y2 = x1 + w, y1 + h
        x1_i = int(np.clip(round(x1), 0, img_w - 1))
        y1_i = int(np.clip(round(y1), 0, img_h - 1))
        x2_i = int(np.clip(round(x2), 0, img_w))
        y2_i = int(np.clip(round(y2), 0, img_h))

        if x2_i <= x1_i or y2_i <= y1_i:
            continue

        boxes.append(
            {
                "x1": x1_i,
                "y1": y1_i,
                "x2": x2_i,
                "y2": y2_i,
                "w": x2_i - x1_i,
                "h": y2_i - y1_i,
                "area": (x2_i - x1_i) * (y2_i - y1_i),
                "class": cls_id,
            }
        )
    return boxes


def box_xywh_to_xyxy(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Chuyển một hộp từ ``(x, y, w, h)`` sang ``(x1, y1, x2, y2)``."""
    x, y, width, height = box
    return x, y, x + width, y + height


def boxes_xywh_to_xyxy(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Chuyển danh sách hộp sang dạng hai góc đối diện."""
    return [box_xywh_to_xyxy(box) for box in boxes]


def compute_iou(boxA: Tuple[Union[int, float], ...], boxB: Tuple[Union[int, float], ...]) -> float:
    """Tính chỉ số IoU (Intersection over Union) giữa 2 hộp dạng (x1, y1, x2, y2).

    Args:
        boxA: Hộp A dạng (x1, y1, x2, y2).
        boxB: Hộp B dạng (x1, y1, x2, y2).

    Returns:
        float: Chỉ số IoU trong khoảng [0.0, 1.0].
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return float(inter / union) if union > 0 else 0.0


def match_boxes(
    pred_boxes: List[Tuple[int, int, int, int]],
    gt_boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.3,
) -> Tuple[int, int, int]:
    """Ghép cặp các hộp dự đoán (Pred) với hộp thực tế (GT) theo chỉ số IoU giảm dần.

    Args:
        pred_boxes (List[Tuple[int, int, int, int]]): Danh sách hộp dự đoán (x1, y1, x2, y2).
        gt_boxes (List[Tuple[int, int, int, int]]): Danh sách hộp GT (x1, y1, x2, y2).
        iou_threshold (float): Ngưỡng IoU coi là ghép thành công. Mặc định 0.3.

    Returns:
        Tuple[int, int, int]: (Số hộp ghép thành công, Tổng số hộp GT, Tổng số hộp Pred).
    """
    pairs = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = compute_iou(gt, pred)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)

    matched_gt, matched_pred = set(), set()
    for iou, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)

    return len(matched_gt), len(gt_boxes), len(pred_boxes)


def evaluate_boxes(
    eval_paths: List[Path],
    label_dir: Path,
    box_fn: Callable[[np.ndarray], List[Tuple[int, int, int, int]]],
    iou_threshold: float = 0.3,
) -> Dict[str, float]:
    """Đánh giá hiệu năng phát hiện vùng ứng viên trên tập danh sách ảnh kiểm thử.

    Args:
        eval_paths (List[Path]): Danh sách các đường dẫn ảnh kiểm thử.
        label_dir (Path): Thư mục chứa nhãn tương ứng.
        box_fn (Callable): Hàm dự đoán trả về danh sách hộp (x, y, w, h) từ ảnh BGR.
        iou_threshold (float): Ngưỡng IoU đánh giá ghép cặp. Mặc định 0.3.

    Returns:
        Dict[str, float]: Từ điển kết quả gồm Recall, Precision, Avg box/ảnh, Số ảnh có GT.
    """
    total_matched = total_gt = total_pred = 0
    n_images_with_gt = 0

    for path in eval_paths:
        img = load_image(path)
        if img is None:
            continue
        gt_boxes = read_label_boxes(label_dir / f"{path.stem}.txt", img.shape)
        if not gt_boxes:
            continue
        n_images_with_gt += 1

        pred_boxes = boxes_xywh_to_xyxy(box_fn(img))
        gt_boxes_xyxy = [(g["x1"], g["y1"], g["x2"], g["y2"]) for g in gt_boxes]

        n_matched, n_gt, n_pred = match_boxes(pred_boxes, gt_boxes_xyxy, iou_threshold)
        total_matched += n_matched
        total_gt += n_gt
        total_pred += n_pred

    return {
        "recall": float(total_matched / total_gt) if total_gt else 0.0,
        "precision": float(total_matched / total_pred) if total_pred else 0.0,
        "avg_pred": float(total_pred / n_images_with_gt) if n_images_with_gt else 0.0,
        "n_images": float(n_images_with_gt),
    }


def print_eval_table(results: Dict[str, Dict[str, float]], label_width: int = 22) -> None:
    """In bảng tổng hợp kết quả đánh giá các phương án phát hiện ứng viên.

    Args:
        results (Dict[str, Dict[str, float]]): Kết quả từ evaluate_boxes.
        label_width (int): Độ rộng cột tên phương án.
    """
    print(
        f"{'Phương án':<{label_width}} | {'Recall (IoU>=0.3)':<20} | {'Precision':<12} | {'Avg box/ảnh':<14} | {'Số ảnh có GT'}"
    )
    print("-" * (label_width + 75))
    for label, r in results.items():
        print(
            f"{label:<{label_width}} | {r['recall']:<20.4f} | {r['precision']:<12.4f} | {r['avg_pred']:<14.2f} | {int(r['n_images'])}"
        )
