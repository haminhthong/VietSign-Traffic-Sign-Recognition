"""VietSign Vision package.

Các module nặng như OpenCV và scikit-learn chỉ được nạp khi API tương ứng
được sử dụng. Điều này giữ cho ``import src`` và lệnh CLI ``--help`` nhẹ hơn.
"""

from importlib import import_module
from typing import Any

__version__ = "1.0.0"
__author__ = "VietSign Vision Team"

_LAZY_EXPORTS = {
    "load_model": ("src.classifier", "load_model"),
    "save_model": ("src.classifier", "save_model"),
    "load_config": ("src.data_loader", "load_config"),
    "load_image": ("src.data_loader", "load_image"),
    "save_image": ("src.data_loader", "save_image"),
    "load_pipeline_config": ("src.pipeline", "load_pipeline_config"),
    "load_pipeline_models": ("src.pipeline", "load_pipeline_models"),
    "process_image_to_rois": ("src.pipeline", "process_image_to_rois"),
    "run_pipeline_on_image": ("src.pipeline", "run_pipeline_on_image"),
}


def __getattr__(name: str) -> Any:
    """Nạp lazy các symbol công khai để tránh import toàn bộ pipeline."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute


def __dir__() -> list[str]:
    """Hiển thị cả các symbol lazy khi dùng ``dir(src)``."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "run_pipeline_on_image",
    "process_image_to_rois",
    "load_pipeline_config",
    "load_pipeline_models",
    "load_image",
    "save_image",
    "load_config",
    "load_model",
    "save_model",
]
