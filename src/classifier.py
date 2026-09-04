"""Module Phân Loại SVM Hai Tầng (Task 6: 2-Tier SVM Classifier).

Quản lý việc huấn luyện, tinh chỉnh tham số (GridSearchCV), dự đoán xác suất an toàn (predict_proba_safe),
đánh giá chỉ số (Accuracy, Macro-F1) và lưu/nạp mô hình SVM hai tầng (Binary SVM & Multiclass SVM).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _validate_training_data(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kiểm tra tính hợp lệ của dữ liệu huấn luyện X và y.

    Args:
        X (np.ndarray): Ma trận đặc trưng 2D.
        y (np.ndarray): Vector nhãn 1D.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (X_arr, y_arr, unique_classes, class_counts).

    Raises:
        ValueError: Nếu kích thước dữ liệu không hợp lệ hoặc số lớp < 2.
    """
    X_arr, y_arr = np.asarray(X), np.asarray(y)
    if X_arr.ndim != 2 or y_arr.ndim != 1 or len(X_arr) != len(y_arr) or len(y_arr) == 0:
        raise ValueError("X phải là ma trận 2 chiều và y là vector cùng số mẫu hợp lệ")

    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Cần ít nhất 2 lớp khác nhau để huấn luyện bộ phân loại SVM")

    return X_arr, y_arr, classes, counts


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: Union[str, float] = "scale",
    class_weight: Optional[str] = "balanced",
    probability: bool = True,
    random_state: int = 42,
) -> Tuple[SVC, StandardScaler]:
    """Huấn luyện mô hình Support Vector Machine (SVM) với bộ chuẩn hóa StandardScaler.

    Args:
        X_train (np.ndarray): Dữ liệu đặc trưng huấn luyện (ví dụ HOG 1.764 chiều).
        y_train (np.ndarray): Nhãn tương ứng.
        kernel (str): Loại kernel SVM ('rbf', 'linear', 'poly'). Mặc định 'rbf'.
        C (float): Tham số điều hòa (Regularization). Mặc định 1.0.
        gamma (Union[str, float]): Hệ số kernel gamma. Mặc định 'scale'.
        class_weight (Optional[str]): Trọng số lớp ('balanced' hoặc None).
        probability (bool): Cho phép tính toán xác suất (probability estimation).
        random_state (int): Hạt giống ngẫu nhiên.

    Returns:
        Tuple[SVC, StandardScaler]: (Mô hình SVC đã train, Scaler đã fit).
    """
    X_train, y_train, _, _ = _validate_training_data(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        decision_function_shape="ovr",
        probability=probability,
        random_state=random_state,
    )
    clf.fit(X_train_scaled, y_train)
    return clf, scaler


def tune_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    scoring: str = "f1_macro",
    cv: int = 3,
    n_jobs: int = -1,
    class_weight: Optional[str] = "balanced",
    probability: bool = True,
    random_state: int = 42,
) -> Tuple[SVC, StandardScaler, Dict[str, Any], float]:
    """Tự động tinh chỉnh siêu tham số (Hyperparameter Tuning) cho SVM bằng GridSearchCV.

    Args:
        X_train (np.ndarray): Đặc trưng huấn luyện.
        y_train (np.ndarray): Nhãn huấn luyện.
        param_grid (Optional[Dict[str, List[Any]]]): Lưới tham số cần tìm kiếm C và gamma.
        scoring (str): Chỉ số tối ưu. Mặc định 'f1_macro'.
        cv (int): Số nếp gấp Cross-validation. Mặc định 3.
        n_jobs (int): Số luồng xử lý song song (-1 cho toàn bộ CPU).
        class_weight (Optional[str]): Cân bằng trọng số lớp.
        probability (bool): Tính xác suất đầu ra.
        random_state (int): Seed ngẫu nhiên.

    Returns:
        Tuple[SVC, StandardScaler, Dict[str, Any], float]: (Model tốt nhất, Scaler, Bộ tham số tốt nhất, Điểm F1 tốt nhất).
    """
    if param_grid is None:
        param_grid = {
            "C": [0.5, 1.0, 2.0, 5.0, 10.0],
            "gamma": ["scale", 0.001, 0.01],
        }

    X_train, y_train, _, counts = _validate_training_data(X_train, y_train)
    min_class = int(counts.min())
    if min_class < 2:
        raise ValueError("Mỗi lớp cần ít nhất 2 mẫu để thực hiện kiểm thử chéo Cross-Validation")

    n_splits = max(2, min(int(cv), min_class))

    # Đặt scaler bên trong Pipeline để mỗi fold chỉ học mean/std từ training fold.
    # Fit scaler trước khi chia fold sẽ làm rò rỉ thống kê của validation fold.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    class_weight=class_weight,
                    decision_function_shape="ovr",
                    probability=probability,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline_grid = {
        (key if key.startswith("svc__") else f"svc__{key}"): value
        for key, value in param_grid.items()
    }
    gs = GridSearchCV(
        pipeline,
        param_grid=pipeline_grid,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        n_jobs=n_jobs,
        refit=True,
    )
    gs.fit(X_train, y_train)
    best_pipeline = gs.best_estimator_
    best_params = {key.removeprefix("svc__"): value for key, value in gs.best_params_.items()}
    return (
        best_pipeline.named_steps["svc"],
        best_pipeline.named_steps["scaler"],
        best_params,
        float(gs.best_score_),
    )


def evaluate_svm(clf: SVC, scaler: StandardScaler, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Đánh giá hiệu năng mô hình SVM trên tập dữ liệu kiểm thử.

    Args:
        clf (SVC): Mô hình SVM đã huấn luyện.
        scaler (StandardScaler): Bộ chuẩn hóa đã fit.
        X (np.ndarray): Ma trận đặc trưng kiểm thử.
        y (np.ndarray): Nhãn thực tế.

    Returns:
        Dict[str, Any]: Từ điển chứa các chỉ số Accuracy, Macro F1, Báo cáo phân loại và Confusion Matrix.
    """
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "y_pred": y_pred,
    }


def calibrate_classifier(
    clf: SVC, scaler: StandardScaler, X_val: np.ndarray, y_val: np.ndarray, method: str = "sigmoid"
) -> CalibratedClassifierCV:
    """Hiệu chỉnh điểm tin cậy (Probability Calibration) trên tập Validation độc lập.

    Giúp chuyển đổi khoảng cách siêu phẳng / raw probability thành xác suất thực sự calibrated.

    Args:
        clf (SVC): Mô hình SVM đã fit.
        scaler (StandardScaler): Bộ chuẩn hóa tương ứng.
        X_val (np.ndarray): Đặc trưng tập Validation.
        y_val (np.ndarray): Nhãn tập Validation.
        method (str): Phương pháp calibration ('sigmoid' hoặc 'isotonic'). Mặc định 'sigmoid'.

    Returns:
        CalibratedClassifierCV: Bộ phân loại đã calibrated trên tập validation.
    """
    X_val_scaled = scaler.transform(X_val)
    calibrated_clf = CalibratedClassifierCV(estimator=clf, method=method, cv="prefit")
    calibrated_clf.fit(X_val_scaled, y_val)
    return calibrated_clf


def predict_proba_safe(
    clf: SVC, scaler: StandardScaler, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Dự đoán nhãn và điểm tin cậy (Confidence score / Probability) một cách an toàn.

    Tự động hỗ trợ cả mô hình bật `probability=True` và mô hình dùng hàm khoảng cách `decision_function`.

    Args:
        clf (SVC): Mô hình SVM.
        scaler (StandardScaler): Bộ chuẩn hóa.
        X (np.ndarray): Ma trận đặc trưng 2D.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Mảng nhãn dự đoán, Mảng điểm tin cậy [0.0..1.0]).
    """
    if X is None or len(X) == 0:
        return np.array([]), np.array([])

    X_scaled = scaler.transform(X)

    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X_scaled)
            idx = proba.argmax(axis=1)
            labels = clf.classes_[idx]
            conf = proba[np.arange(len(idx)), idx]
            return labels, conf
        except (AttributeError, ValueError):
            # SVC không bật probability sẽ dùng decision_function bên dưới.
            pass

    labels = clf.predict(X_scaled)
    # Dự phòng: Tính Sigmoid / Softmax từ khoảng cách siêu phẳng decision_function
    if hasattr(clf, "decision_function"):
        df = clf.decision_function(X_scaled)
        if df.ndim == 1:
            conf = 1.0 / (1.0 + np.exp(-np.abs(df)))
        else:
            e = np.exp(df - df.max(axis=1, keepdims=True))
            conf = (e / e.sum(axis=1, keepdims=True)).max(axis=1)
        return labels, conf

    return labels, np.ones(len(labels), dtype=float)


def save_model(clf: SVC, scaler: StandardScaler, path: Union[str, Path]) -> None:
    """Lưu mô hình SVM và Scaler vào tệp .joblib với nén dữ liệu.

    Args:
        clf (SVC): Mô hình đã fit.
        scaler (StandardScaler): Scaler tương ứng.
        path (Union[str, Path]): Đường dẫn tệp đích.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler}, p, compress=3)


def load_model(path: Union[str, Path]) -> Tuple[SVC, StandardScaler]:
    """Nạp mô hình SVM và Scaler từ tệp .joblib.

    Args:
        path (Union[str, Path]): Đường dẫn tệp .joblib.

    Returns:
        Tuple[SVC, StandardScaler]: (Mô hình SVC, Scaler).

    Raises:
        ValueError: Nếu tệp mô hình không hợp lệ hoặc thiếu dữ liệu.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp mô hình tại: {p}")

    data = joblib.load(p)
    if not isinstance(data, dict) or "model" not in data or "scaler" not in data:
        raise ValueError(f"Tệp mô hình không đúng định dạng joblib chứa 'model' và 'scaler': {p}")

    return data["model"], data["scaler"]
