"""Unit tests to verify app package structure and syntax."""
from __future__ import annotations

import importlib


def test_app_imports() -> None:
    app_main = importlib.import_module("app.main")
    assert hasattr(app_main, "main")

    sidebar = importlib.import_module("app.components.sidebar")
    assert hasattr(sidebar, "render_sidebar")

    pcard = importlib.import_module("app.components.product_card")
    assert hasattr(pcard, "render_product_grid")

    t1 = importlib.import_module("app.tabs.text_to_image")
    assert hasattr(t1, "render_tab_text_to_image")

    t2 = importlib.import_module("app.tabs.image_to_image")
    assert hasattr(t2, "render_tab_image_to_image")

    t3 = importlib.import_module("app.tabs.cir_catalog")
    assert hasattr(t3, "render_tab_cir_catalog")

    t4 = importlib.import_module("app.tabs.cir_upload")
    assert hasattr(t4, "render_tab_cir_upload")
