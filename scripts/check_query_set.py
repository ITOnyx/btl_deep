"""
Công cụ kiểm tra chung cho các tập truy vấn xác nhận (nâng cấp từ scripts/check_b_human.py).
KHÔNG nạp hay chạy model.

Hỗ trợ kiểm tra:
- n_relevant trên index/val_gallery/ hoặc index/test_gallery/ (chính sách hybrid)
- dia_class tự phát hiện (none / full / partial) và cảnh báo lệch khai báo
- Trùng / gần trùng Jaccard >= 0.8 với DEV / A / B-synth / C / B-human cũ và nội bộ
- Token nằm ngoài lexicon v1 và lexicon v2.1 (CHỈ BÁO CÁO, không vá lexicon)
- Trường `source` (nguồn gốc câu truy vấn)
- Kiểm tra ngưỡng theo preset (`b_human_v4`, `heldout_v2`, `generic`)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_b_human import (
    build_v1_lexicon_tokens,
    detect_dia_class_auto,
    jaccard_sim,
    tokenize_norm,
)
from src.eval.metrics import is_item_relevant, load_gender_gt_policy
from src.serve.parser_v2 import (
    ARTICLE_TYPE_LEXICON_V2,
    COLOUR_LEXICON_V2,
    GENDER_LEXICON_V2,
)


def build_v2_lexicon_tokens() -> Set[str]:
    """Tập token chuẩn hóa thuộc lexicon v2.1 (chỉ đọc từ parser_v2)."""
    tokens: Set[str] = set()
    for lex in (ARTICLE_TYPE_LEXICON_V2, GENDER_LEXICON_V2, COLOUR_LEXICON_V2):
        for kw_list in lex.values():
            for kw in kw_list:
                tokens.update(tokenize_norm(kw))
    return tokens


def make_cluster_id(spec: Dict[str, Any]) -> str:
    ats = "|".join(sorted(spec.get("articleType") or []))
    gd = str(spec.get("gender") or "None")
    cls = "|".join(sorted(spec.get("baseColour") or []))
    return f"{ats}__{gd}__{cls}"


def load_all_reference_datasets(
    root: pathlib.Path,
    exclude_path: Optional[pathlib.Path] = None,
) -> Dict[str, List[Tuple[str, Set[str]]]]:
    """Nạp các tập khóa để kiểm tra trùng Jaccard."""
    refs: Dict[str, List[Tuple[str, Set[str]]]] = {}
    candidates = [
        ("DEV", "eval/handwritten_queries.json", None),
        ("A", "eval/val_queries_a_synthetic.json", None),
        ("B-synth", "eval/val_queries_b_synth.json", None),
        ("C", "eval/val_queries_c_error_injection.json", "queries"),
        ("B-human", "eval/val_queries_b_human.json", None),
    ]
    ex_resolved = exclude_path.resolve() if exclude_path else None
    for name, rel_path, key in candidates:
        p = root / rel_path
        if not p.exists():
            continue
        if ex_resolved is not None and p.resolve() == ex_resolved:
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data[key] if key else data
        refs[name] = [
            (str(x.get("id", i)), tokenize_norm(x.get("query", "")))
            for i, x in enumerate(items)
        ]
    return refs


def check_query_set(
    query_path: pathlib.Path,
    gallery_id_map: List[Dict[str, Any]],
    strict_categories: Set[str],
    ref_datasets: Dict[str, List[Tuple[str, Set[str]]]],
    min_queries: int = 60,
    min_relevant: int = 5,
    jaccard_threshold: float = 0.8,
    preset: str = "generic",
) -> Dict[str, Any]:
    raw_data = json.loads(query_path.read_text(encoding="utf-8"))
    queries = raw_data["queries"] if isinstance(raw_data, dict) and "queries" in raw_data else raw_data

    v1_tokens = build_v1_lexicon_tokens()
    v2_tokens = build_v2_lexicon_tokens()

    rows: List[Dict[str, Any]] = []
    warnings_low_rel = 0
    warnings_dup = 0
    warnings_internal_dup = 0
    warnings_dia_mismatch = 0
    warnings_missing_source = 0

    seen_tokens: List[Tuple[str, Set[str]]] = []
    cluster_ids: Set[str] = set()
    source_counter: Counter = Counter()
    dia_counter: Counter = Counter()
    n_outside_v1 = 0
    n_outside_v2 = 0
    n_typo = 0
    n_usage_season = 0

    for idx, q in enumerate(queries):
        qid = str(q.get("id", f"row_{idx}"))
        text = str(q.get("query", ""))
        spec = q.get("spec") or {"articleType": [], "gender": None, "baseColour": []}
        source = q.get("source")
        if not source:
            warnings_missing_source += 1
            source_counter["<missing>"] += 1
        else:
            source_counter[str(source)] += 1

        n_rel = sum(
            1 for m in gallery_id_map if is_item_relevant(spec, m, "hybrid", strict_categories)
        )
        if n_rel < min_relevant:
            warnings_low_rel += 1

        cid = make_cluster_id(spec)
        cluster_ids.add(cid)

        dia_auto = detect_dia_class_auto(text)
        dia_counter[dia_auto] += 1
        dia_declared = q.get("dia_class")
        dia_mismatch = (dia_declared is not None) and (dia_declared != dia_auto)
        if dia_mismatch:
            warnings_dia_mismatch += 1

        q_toks = tokenize_norm(text)

        near_dups: List[Dict[str, Any]] = []
        for ds_name, ref_items in ref_datasets.items():
            for ref_id, ref_toks in ref_items:
                sim = jaccard_sim(q_toks, ref_toks)
                if sim >= jaccard_threshold:
                    near_dups.append({"dataset": ds_name, "id": ref_id, "jaccard": round(sim, 4)})
        if near_dups:
            warnings_dup += 1

        internal_dups: List[Dict[str, Any]] = []
        for prev_id, prev_toks in seen_tokens:
            sim = jaccard_sim(q_toks, prev_toks)
            if sim >= jaccard_threshold:
                internal_dups.append({"id": prev_id, "jaccard": round(sim, 4)})
        if internal_dups:
            warnings_internal_dup += 1
        seen_tokens.append((qid, q_toks))

        oov_v1 = sorted(q_toks - v1_tokens)
        oov_v2 = sorted(q_toks - v2_tokens)
        if oov_v1:
            n_outside_v1 += 1
        if oov_v2:
            n_outside_v2 += 1

        typo_type = q.get("typo_type")
        if typo_type:
            n_typo += 1

        has_usage_season = bool(
            q.get("usage")
            or q.get("season")
            or (isinstance(spec, dict) and (spec.get("usage") or spec.get("season")))
        )
        if has_usage_season:
            n_usage_season += 1

        rows.append({
            "id": qid,
            "query": text,
            "source": source,
            "spec_cluster_id": cid,
            "n_relevant": n_rel,
            "low_relevant_warning": n_rel < min_relevant,
            "dia_class_declared": dia_declared,
            "dia_class_auto": dia_auto,
            "dia_class_mismatch_warning": dia_mismatch,
            "near_duplicates": near_dups,
            "internal_duplicates": internal_dups,
            "tokens_outside_v1_lexicon": oov_v1,
            "tokens_outside_v2_lexicon": oov_v2,
            "typo_type": typo_type,
            "has_usage_season": has_usage_season,
        })

    n_total = len(queries)
    n_no_dia = dia_counter.get("none", 0)
    pct_outside_v2 = (n_outside_v2 / n_total) if n_total else 0.0

    preset_checks: Dict[str, Any] = {}
    if preset == "b_human_v4":
        preset_checks = {
            "min_queries_60": n_total >= 60,
            "min_clusters_30": len(cluster_ids) >= 30,
            "all_n_relevant_ge5": warnings_low_rel == 0,
            "no_ref_duplicates": warnings_dup == 0,
            "no_internal_duplicates": warnings_internal_dup == 0,
            "min_outside_lexicon_v2_20": n_outside_v2 >= 20,
            "min_no_diacritics_20": n_no_dia >= 20,
            "exact_typo_10": n_typo == 10,
            "all_have_source": warnings_missing_source == 0,
        }
        preset_checks["passed"] = all(preset_checks.values())
    elif preset == "heldout_v2":
        preset_checks = {
            "min_queries_100": n_total >= 100,
            "all_n_relevant_ge5": warnings_low_rel == 0,
            "no_ref_duplicates": warnings_dup == 0,
            "no_internal_duplicates": warnings_internal_dup == 0,
            "min_no_diacritics_40": n_no_dia >= 40,
            "min_pct_outside_lexicon_v2_30": pct_outside_v2 >= 0.30,
            "min_usage_season_40": n_usage_season >= 40,
            "all_have_source": warnings_missing_source == 0,
        }
        preset_checks["passed"] = all(preset_checks.values())

    return {
        "file": str(query_path),
        "preset": preset,
        "n_queries": n_total,
        "n_clusters": len(cluster_ids),
        "meets_min_queries": n_total >= min_queries,
        "min_queries_required": min_queries,
        "warnings_low_relevant_lt5": warnings_low_rel,
        "warnings_near_duplicate_ge08": warnings_dup,
        "warnings_internal_duplicate_ge08": warnings_internal_dup,
        "warnings_dia_class_mismatch": warnings_dia_mismatch,
        "warnings_missing_source": warnings_missing_source,
        "dia_class_counts": dict(dia_counter),
        "source_counts": dict(source_counter),
        "n_outside_v1_lexicon": n_outside_v1,
        "n_outside_v2_lexicon": n_outside_v2,
        "pct_outside_v2_lexicon": round(pct_outside_v2, 4),
        "n_typo": n_typo,
        "n_with_usage_season": n_usage_season,
        "preset_checks": preset_checks,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra chung các tập truy vấn (không chạy model).")
    parser.add_argument("path", help="Đường dẫn tới file JSON tập truy vấn")
    parser.add_argument(
        "--gallery",
        choices=["val", "test"],
        default="val",
        help="Gallery đối chiếu: val (index/val_gallery) hoặc test (index/test_gallery)",
    )
    parser.add_argument(
        "--preset",
        choices=["generic", "b_human_v4", "heldout_v2"],
        default="generic",
        help="Bộ tiêu chí kiểm tra áp dụng",
    )
    parser.add_argument("--min-queries", type=int, default=60)
    parser.add_argument("--min-relevant", type=int, default=5)
    parser.add_argument("--out", default=None, help="Ghi báo cáo JSON ra file nếu muốn")
    args = parser.parse_args()

    q_path = pathlib.Path(args.path)
    gallery_dir = "index/val_gallery" if args.gallery == "val" else "index/test_gallery"
    id_map = json.loads((PROJECT_ROOT / gallery_dir / "id_map.json").read_text(encoding="utf-8"))
    strict_cats = set(
        load_gender_gt_policy(PROJECT_ROOT / "eval/gender_gt_policy.json")["policies"]["hybrid"]["strict_categories"]
    )
    refs = load_all_reference_datasets(PROJECT_ROOT, exclude_path=q_path)

    min_q = 100 if args.preset == "heldout_v2" else args.min_queries
    report = check_query_set(
        query_path=q_path,
        gallery_id_map=id_map,
        strict_categories=strict_cats,
        ref_datasets=refs,
        min_queries=min_q,
        min_relevant=args.min_relevant,
        preset=args.preset,
    )

    print(f"File: {report['file']} | Gallery: {gallery_dir} | Preset: {args.preset}")
    print(f"Số câu: {report['n_queries']} | Số cụm: {report['n_clusters']}")
    print(f"Cảnh báo n_relevant < {args.min_relevant}: {report['warnings_low_relevant_lt5']}")
    print(f"Cảnh báo trùng tập khóa (Jaccard >= 0.8): {report['warnings_near_duplicate_ge08']}")
    print(f"Cảnh báo trùng nội bộ (Jaccard >= 0.8): {report['warnings_internal_duplicate_ge08']}")
    print(f"Cảnh báo lệch dia_class: {report['warnings_dia_class_mismatch']}")
    print(f"Cảnh báo thiếu source: {report['warnings_missing_source']}")
    print(f"Phân bố dia_class: {report['dia_class_counts']}")
    print(f"Phân bố source: {report['source_counts']}")
    print(
        f"Ngoài lexicon v2.1: {report['n_outside_v2_lexicon']}/{report['n_queries']} "
        f"({report['pct_outside_v2_lexicon']*100:.1f}%)"
    )
    print(f"Số câu typo: {report['n_typo']} | Có usage/season: {report['n_with_usage_season']}")
    if report["preset_checks"]:
        print(f"Preset checks: {report['preset_checks']}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if report["preset_checks"]:
        return 0 if report["preset_checks"]["passed"] else 1
    return 0 if report["meets_min_queries"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
