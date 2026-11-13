"""Unit tests cho Bước 3b/3c/3e (parser_v2, query_rewriter, BM25, truy nguyên catalog)."""
import pathlib
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.canonical_eval import evaluate_t2i_canonical
from src.serve.parser_v2 import (
    ARTICLE_TYPE_LEXICON_V2,
    COLOUR_LEXICON_V2,
    GENDER_LEXICON_V2,
    VietnameseQueryParserV2,
)
from src.serve.query_rewriter import BM25GalleryScorer, restore_diacritics_for_query


def test_parser_v2_catalog_traceability():
    """Mọi nhãn trong parser_v2 đều truy nguyên trực tiếp về catalog_rebuild.parquet."""
    df = pd.read_parquet(PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet")
    cat_ats = set(df["articleType"].dropna().unique())
    cat_gds = set(df["gender"].dropna().unique())
    cat_cls = set(df["baseColour"].dropna().unique())

    for label in ARTICLE_TYPE_LEXICON_V2:
        assert label in cat_ats, f"articleType '{label}' không có trong catalog"
    for label in GENDER_LEXICON_V2:
        assert label in cat_gds, f"gender '{label}' không có trong catalog"
    for label in COLOUR_LEXICON_V2:
        assert label in cat_cls, f"baseColour '{label}' không có trong catalog"


def test_parser_v2_homograph_false_positive_guard():
    """Kiểm tra chống dương tính giả trên bảng homograph 2c.1c."""
    p = VietnameseQueryParserV2()

    # Không được kích hoạt màu từ đồng âm
    assert "Purple" not in p.parse("tìm mua áo thun nam")["baseColour"]
    assert "Purple" not in p.parse("tim mua ao thun nam")["baseColour"]
    assert "Purple" not in p.parse("cần tìm áo sơ mi")["baseColour"]
    assert "Red" not in p.parse("bộ đồ ngủ nữ")["baseColour"]
    assert "Red" not in p.parse("túi trống đựng đồ")["baseColour"]
    assert "Beige" not in p.parse("áo thun bé trai")["baseColour"]
    assert p.parse("áo thun bé trai")["gender"] == "Boys"
    assert "Orange" not in p.parse("ví cầm tay nữ")["baseColour"]
    assert "Beige" not in p.parse("giày búp bê nữ")["baseColour"]
    assert "White" not in p.parse("đồng hồ thời trang nam")["baseColour"]
    assert p.parse("áo sơ mi năm mới")["gender"] is None

    # Phải bắt đúng khi là màu thật (có dấu hoặc có ngữ cảnh)
    assert "Purple" in p.parse("áo thun nam màu tím")["baseColour"]
    assert "Purple" in p.parse("ao thun nam mau tim")["baseColour"]
    assert "Purple" in p.parse("áo thun nam tím")["baseColour"]
    assert "Purple" in p.parse("ao thun nam tim")["baseColour"]
    assert "Steel" in p.parse("đồng hồ nam thép không gỉ")["baseColour"]
    assert "Olive" in p.parse("áo khoác nam màu xanh ô liu")["baseColour"]
    assert "Navy Blue" in p.parse("quần tây nam xanh navy")["baseColour"]


def test_query_rewriter_and_bm25_canonical():
    """Kiểm tra khôi phục dấu 3c và BM25 + zscore trong canonical_eval."""
    rew = restore_diacritics_for_query("tim mua ao thun nam mau den")
    assert "tìm mua" in rew
    assert "áo thun" in rew
    assert "màu đen" in rew

    # Câu đã có dấu giữ nguyên
    acc = "áo thun nam màu tím"
    assert restore_diacritics_for_query(acc) == acc

    id_map = [
        {"id": 1, "articleType": "Tshirts", "gender": "Men", "baseColour": "Black", "articleType_vi": "áo thun", "baseColour_vi": "đen"},
        {"id": 2, "articleType": "Jeans", "gender": "Men", "baseColour": "Blue", "articleType_vi": "quần jean", "baseColour_vi": "xanh dương"},
    ]
    bm25 = BM25GalleryScorer(id_map)
    s = bm25.score_query("ao thun den")
    assert s.shape == (2,)
    assert s[0] > s[1]

    class _E:
        def __init__(self):
            self.embeddings = np.array([[0.8, 0.6], [0.8, 0.6]], dtype=np.float32)
            self.id_map = id_map

        def _apply_mean_centering(self, v):
            return v

    res = evaluate_t2i_canonical(
        engine=_E(),
        queries=[{"id": "q1", "query": "ao thun den", "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]}}],
        query_vecs=np.array([[0.8, 0.6]], dtype=np.float32),
        parsed_queries=[{"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]}],
        gender_policy_cfg={"policies": {"hybrid": {"strict_categories": ["Tshirts"]}}},
        mode="system",
        candidate_pool=2,
        eval_top_k=2,
        gamma=0.3,
        zscore_cosine=True,
        bm25_scores=s.reshape(1, -1),
        bm25_weight=0.2,
    )
    assert res["policies"]["hybrid"]["summary"]["recall@1"] == 1.0
