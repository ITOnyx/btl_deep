"""Hiển thị lỗi tìm kiếm thống nhất cho các tab: lỗi đầu vào -> cảnh báo, lỗi khác -> báo lỗi + ghi log."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image, UnidentifiedImageError

from app.engine_service import ImageValidationError, SearchInputError, SearchRuntimeError

logger = logging.getLogger(__name__)


def show_search_error(exc: Exception) -> None:
    if isinstance(exc, (SearchInputError, ImageValidationError)):
        st.warning(str(exc))
    elif isinstance(exc, SearchRuntimeError):
        st.error(str(exc))
    else:
        # Lỗi không lường trước: ghi traceback vào log server, chỉ hiện thông báo gọn cho người dùng
        logger.exception("Lỗi tìm kiếm không mong đợi")
        st.error(f"Lỗi tìm kiếm: {type(exc).__name__}: {exc}")


def load_sample_image(path: Path) -> Optional[Image.Image]:
    """Mở ảnh mẫu từ gallery; ảnh thiếu/hỏng -> cảnh báo và trả None thay vì làm vỡ trang."""
    if not path.exists():
        st.warning(f"Không tìm thấy ảnh mẫu: {path.name}")
        return None
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        logger.warning("Ảnh mẫu hỏng %s: %s", path, e)
        st.warning(f"Ảnh mẫu {path.name} bị hỏng hoặc không đọc được.")
        return None
