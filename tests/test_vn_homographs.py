"""Hồi quy cho khớp đồng dạng tiếng Việt (vn_text / taxonomy / parser_v3).

Khớp nới theo ngữ cảnh ('vì da' -> 'ví da') chỉ được dùng khi không có từ sản phẩm nào khớp chặt,
và không bao giờ nới một từ màu thành một chữ của cụm sản phẩm.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.serve.parser_v3 import VietnameseQueryParserV3
from src.serve.taxonomy import match_query_taxonomy
from src.serve.vn_text import span_compatible


@pytest.fixture(scope="module")
def parser():
    return VietnameseQueryParserV3()


def test_span_compatible_never_relaxes_colour():
    assert not span_compatible("túi đeo hồng", "túi đeo hông", allow_context=True)
    assert not span_compatible("đỏ ngủ", "đồ ngủ", allow_context=True)
    assert span_compatible("vì da", "ví da", allow_context=True)
    assert not span_compatible("vì da", "ví da")


@pytest.mark.parametrize("query, expected_type", [
    ("túi đeo hồng", []),
    ("áo đỏ ngủ", []),
    ("quần đỏ lót", []),
    ("đồ đỏ bơi", []),
    ("áo chống nắng vì da nhạy cảm", ["Tshirts"]),
    ("áo đẹp tông trắng", []),
    ("quần đẹp tông xám", []),
    ("giày đẹp kẹp", []),
    # vẫn giữ các trường hợp nới có chủ đích
    ("vì da nam", ["Wallets"]),
    ("đẹp tông nam", ["Flip Flops"]),
    ("túi đeo hông", ["Waist Pouch"]),
])
def test_parser_v3_homographs(parser, query, expected_type):
    assert parser.parse(query)["articleType"] == expected_type


@pytest.mark.parametrize("query, art", [
    ("áo chống nắng vì da nhạy cảm", None),
    ("áo đẹp tông trắng", None),
    ("đồ đỏ bơi", None),
    ("vì da nam", "Wallets"),
    ("đẹp tông nam", "Flip Flops"),
])
def test_taxonomy_prefers_strict_match(query, art):
    assert match_query_taxonomy(query)["articleType"] == art


def test_red_colour_kept_with_homograph_context(parser):
    assert parser.parse("túi đeo hồng")["baseColour"] == ["Pink"]


def test_hnsw_search_does_not_mutate_query():
    from src.index.faiss_index import FaissHNSWIndex

    rng = np.random.default_rng(42)
    idx = FaissHNSWIndex(dim=8)
    idx.add(rng.standard_normal((20, 8)).astype(np.float32))
    q = (rng.standard_normal((1, 8)) * 5).astype(np.float32)
    before = q.copy()
    idx.search(q, top_k=3)
    assert np.array_equal(q, before)


@pytest.mark.parametrize("query, gender", [
    ("giày hàng việt nam", None),
    ("áo miền nam", None),
    ("ao mien nam", None),
    ("giày nam việt nam", "Men"),
    ("giày nam", "Men"),
])
def test_place_names_are_not_men(parser, query, gender):
    assert parser.parse(query)["gender"] == gender


@pytest.mark.parametrize("query, colours", [
    ("áo khoác đen trắng", ["Black", "White"]),
    ("áo trắng đen", ["White", "Black"]),
    ("áo xanh đậm đen", ["Blue", "Black"]),
    ("áo thun cam kết chính hãng", []),
    ("túi xách trang trí", []),
    ("áo màu cam kết hợp trắng", ["Orange", "White"]),
    ("trang sức bạc", ["Silver"]),
])
def test_colour_order_and_non_colour_phrases(parser, query, colours):
    assert parser.parse(query)["baseColour"] == colours


@pytest.mark.parametrize("query, colours, art", [
    ("áo cam", ["Orange"], []),
    ("ao cam", ["Orange"], []),
    ("quần jean xanh", ["Blue"], ["Jeans"]),
    ("quần bò xanh", ["Blue"], ["Jeans"]),
    ("áo xanh", ["Blue"], []),          # trước đây khớp mờ đọc thành 'áo cánh' (Tops)
    ("giày be", ["Beige"], []),
    ("áo bé trai", [], []),             # 'bé' không phải màu be
    ("áo cảm ứng", [], []),             # 'cảm' không phải cam
    ("kem chống nắng", [], []),         # 'kem' nằm trong tên sản phẩm
    ("quần kaki nam màu be dáng ôm", ["Beige", "Khaki"], ["Trousers"]),  # 'kaki' thuộc tên sản phẩm
])
def test_standalone_colour_after_product(parser, query, colours, art):
    r = parser.parse(query)
    assert r["baseColour"] == colours
    assert r["articleType"] == art


@pytest.mark.parametrize("query, art", [
    ("giống vậy nhưng màu đỏ", []),
    ("giong vay nhung mau do", []),
    ("như vậy nhưng cho nữ", []),
    ("váy giống vậy màu đen", ["Dresses"]),
    ("váy nhung đỏ", ["Dresses"]),
    ("vay nhung do", ["Dresses"]),
    ("vay hoa nu", ["Dresses"]),
])
def test_giong_vay_is_not_dress(parser, query, art):
    assert parser.parse(query)["articleType"] == art


@pytest.mark.parametrize("query, group", [
    ("giong vay nhung mau do", None),
    ("giống vậy nhưng màu đỏ", None),
    ("tụi mình cần đồ", None),
    ("váy giống vậy", "Đầm & Trang phục truyền thống"),
])
def test_taxonomy_giong_vay(query, group):
    assert match_query_taxonomy(query)["group"] == group


def test_rewriter_keeps_giong_vay():
    from src.serve.query_rewriter import restore_diacritics_for_query
    assert restore_diacritics_for_query("giong vay nhung mau do").startswith("giống vậy")
    assert restore_diacritics_for_query("vay hoa nu") == "váy hoa nữ"
