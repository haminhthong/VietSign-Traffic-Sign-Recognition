"""Module Nạp & Quản Lý Dữ Liệu (Data Loader & Utilities).

Cung cấp các tiện ích đọc/ghi ảnh an toàn với đường dẫn tiếng Việt Unicode trên Windows (`cv2.imdecode`/`cv2.imencode`),
tải và tuần tự hóa tệp cấu hình `config.yaml`, và duyệt thư mục dữ liệu.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], Path, Path]:
    """Tải tệp cấu hình `config.yaml` từ gốc dự án.

    Args:
        project_root (Optional[Union[str, Path]]): Đường dẫn gốc dự án. Nếu None sẽ tự xác định.

    Returns:
        Tuple[Dict[str, Any], Path, Path]: (Đối tượng dict cấu hình, Đường dẫn gốc, Đường dẫn tệp config).

    Raises:
        FileNotFoundError: Nếu tệp config.yaml không tồn tại.
        ValueError: Nếu tệp config không đúng định dạng dictionary.
    """
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    config_path = root / "config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp cấu hình YAML tại: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Tệp config.yaml phải chứa một cấu trúc đối tượng (dict) ở cấp cao nhất: {config_path}"
        )

    return cfg, root, config_path


def get_image_dirs(cfg: Dict[str, Any], project_root: Path) -> Tuple[Path, Path]:
    """Lấy thư mục chứa ảnh raw và nhãn tương ứng từ cấu hình.

    Args:
        cfg (Dict[str, Any]): Cấu hình từ config.yaml.
        project_root (Path): Đường dẫn gốc.

    Returns:
        Tuple[Path, Path]: (Thư mục ảnh images, Thư mục nhãn labels).
    """
    image_dir = project_root / cfg["paths"]["data_raw"] / "images"
    label_dir = project_root / cfg["paths"]["data_raw"] / "labels"
    return image_dir, label_dir


def list_image_paths(
    image_dir: Union[str, Path], valid_ext: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
) -> List[Path]:
    """Liệt kê tất cả các tệp ảnh hợp lệ trong thư mục theo thứ tự bảng chữ cái.

    Args:
        image_dir (Union[str, Path]): Thư mục ảnh.
        valid_ext (Tuple[str, ...]): Định dạng mở rộng hợp lệ. Mặc định (.jpg, .jpeg, .png).

    Returns:
        List[Path]: Danh sách các đường dẫn tệp ảnh.
    """
    p = Path(image_dir)
    if not p.is_dir():
        return []
    return sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_ext])


def load_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Đọc ảnh BGR an toàn, hỗ trợ hoàn hảo đường dẫn tiếng Việt có dấu/Unicode trên Windows.

    Sử dụng `np.fromfile` kết hợp với `cv2.imdecode` để tránh lỗi đường dẫn chứa ký tự tiếng Việt hoặc khoảng trắng.

    Args:
        path (Union[str, Path]): Đường dẫn tệp ảnh.

    Returns:
        Optional[np.ndarray]: Ma trận ảnh BGR OpenCV (hoặc None nếu không đọc được).
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        encoded = np.fromfile(p, dtype=np.uint8)
        if encoded.size == 0:
            return None
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def save_image(path: Union[str, Path], image: np.ndarray) -> bool:
    """Ghi ảnh an toàn với đường dẫn tiếng Việt Unicode trên Windows (`cv2.imencode`).

    Args:
        path (Union[str, Path]): Đường dẫn tệp xuất.
        image (np.ndarray): Ma trận ảnh OpenCV BGR.

    Returns:
        bool: True nếu ghi ảnh thành công, False nếu thất bại.
    """
    p = Path(path)
    if image is None or image.size == 0 or not p.suffix:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(p.suffix, image)
    if not success:
        return False
    try:
        encoded.tofile(p)
        return True
    except OSError:
        return False


def save_config(cfg: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Lưu từ điển cấu hình về tệp `config.yaml`.

    Args:
        cfg (Dict[str, Any]): Từ điển cấu hình.
        config_path (Union[str, Path]): Đường dẫn tệp lưu.
    """

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, (tuple, list)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        return obj

    p = Path(config_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            _sanitize(cfg),
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
