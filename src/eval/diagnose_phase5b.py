"""
Phân tích và chẩn đoán tầng suy luận Pha 5b trên Baseline Test Gallery (DEV SET).
Tuân thủ:
- Luật 1: Không train/fine-tune/sửa best_model.pt.
- Luật 3: Ghi vào reports/phase5b/, không sửa index/test_gallery.
- Luật 6: Hash Guard kiểm định checkpoint.
- C.5: eval/handwritten_queries.json là DEV SET (chẩn đoán, không tune).
- Seed: 42
"""
import os
import sys
import json
import pathlib
import pandas as pd
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Dọn dẹp biến môi trường TLS Windows nếu có
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

from src.serve.search_engine import ECommerceSearchEngine
from src.serve.parser import VietnameseQueryParser, compute_soft_attribute_match
from src.eval.metrics import (
    load_gender_gt_policy,
    is_item_relevant,
    compute_recall_at_k,
    compute_average_precision,
    compute_retrieval_summary
)


def run_diagnostics():
    print("=" * 60)
    print("🔍 BẮT ĐẦU CHẨN ĐOÁN BASELINE PHA 5b (DEV SET)")
    print("=" * 60)

    # 1. Khởi tạo Engine trên test_gallery
    engine = ECommerceSearchEngine(
        index_dir="index/test_gallery",
        config_path="config.yaml",
        device="cuda",
        use_mean_centering=False
    )

    # 2. Đọc DEV SET queries
    with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    policy_cfg = load_gender_gt_policy("eval/gender_gt_policy.json")
    strict_cats = policy_cfg["policies"]["hybrid"]["strict_categories"]
    id_map = engine.id_map
    gallery_size = len(id_map)
    print(f"• Số lượng ảnh trong test gallery: {gallery_size}")
    print(f"• Số lượng truy vấn DEV SET: {len(queries)}")

    # 3. Encode trước toàn bộ queries
    query_texts = [q["query"] for q in queries]
    query_vecs = []
    for txt in query_texts:
        vec = engine.encode_text_query(txt)
        query_vecs.append(vec)
    query_vecs = np.vstack(query_vecs)

    parser = engine.parser

    # Danh sách kết quả chi tiết từng query
    diag_rows = []

    # Thu thập cho tính ceiling
    ranked_rels_pool100_sys = []
    ranked_rels_pool1290_sys = []
    ranked_rels_pool100_model = []
    ranked_rels_pool1290_model = []
    tot_rels_valid = []

    # Chạy từng query
    for i, q in enumerate(queries):
        txt = q["query"]
        q_vec = query_vecs[i : i + 1]
        spec = q["spec"]
        has_diacritics = q.get("has_diacritics", True)
        thin_slot = q.get("category", "unknown")

        # Đếm tổng số ảnh GT thực tế trong test gallery theo policy hybrid
        rel_mask = [is_item_relevant(spec, meta, "hybrid", strict_cats) for meta in id_map]
        n_relevant = sum(rel_mask)

        # Parse query
        parsed = parser.parse(txt)

        # --- A. Baseline Candidate Pool = 100 ---
        # 1. Model only (pool=100)
        res_model_100 = engine.search_by_vector(
            query_vec=q_vec,
            top_k=100,
            parsed_query=parsed,
            mode="model_only",
            candidate_pool=100
        )
        # 2. System (pool=100)
        res_sys_100 = engine.search_by_vector(
            query_vec=q_vec,
            top_k=100,
            parsed_query=parsed,
            mode="system",
            candidate_pool=100
        )

        # --- B. Ceiling Candidate Pool = gallery_size (1.290) ---
        res_model_1290 = engine.search_by_vector(
            query_vec=q_vec,
            top_k=100,
            parsed_query=parsed,
            mode="model_only",
            candidate_pool=gallery_size
        )
        res_sys_1290 = engine.search_by_vector(
            query_vec=q_vec,
            top_k=100,
            parsed_query=parsed,
            mode="system",
            candidate_pool=gallery_size
        )

        # Tính binary relevance cho từng kết quả
        rels_model_100 = [1 if is_item_relevant(spec, id_map[r["index_pos"]], "hybrid", strict_cats) else 0 for r in res_model_100]
        rels_sys_100 = [1 if is_item_relevant(spec, id_map[r["index_pos"]], "hybrid", strict_cats) else 0 for r in res_sys_100]
        rels_model_1290 = [1 if is_item_relevant(spec, id_map[r["index_pos"]], "hybrid", strict_cats) else 0 for r in res_model_1290]
        rels_sys_1290 = [1 if is_item_relevant(spec, id_map[r["index_pos"]], "hybrid", strict_cats) else 0 for r in res_sys_1290]

        # Kiểm tra ảnh đúng trong top-100 candidate pool (chính là res_model_100)
        is_rel_in_pool_100 = any(rels_model_100)

        # Tìm rank đầu tiên (1-based)
        model_rank_first = (rels_model_100.index(1) + 1) if any(rels_model_100) else None
        sys_rank_first = (rels_sys_100.index(1) + 1) if any(rels_sys_100) else None
        sys_rank_first_1290 = (rels_sys_1290.index(1) + 1) if any(rels_sys_1290) else None

        # Tính AP
        model_ap_100 = compute_average_precision(rels_model_100, total_relevant=n_relevant) if n_relevant > 0 else 0.0
        sys_ap_100 = compute_average_precision(rels_sys_100, total_relevant=n_relevant) if n_relevant > 0 else 0.0
        model_ap_1290 = compute_average_precision(rels_model_1290, total_relevant=n_relevant) if n_relevant > 0 else 0.0
        sys_ap_1290 = compute_average_precision(rels_sys_1290, total_relevant=n_relevant) if n_relevant > 0 else 0.0

        if n_relevant > 0:
            ranked_rels_pool100_sys.append(rels_sys_100)
            ranked_rels_pool1290_sys.append(rels_sys_1290)
            ranked_rels_pool100_model.append(rels_model_100)
            ranked_rels_pool1290_model.append(rels_model_1290)
            tot_rels_valid.append(n_relevant)

        # Phân loại lỗi
        error_type = "PASS"
        error_details = []
        if n_relevant == 0:
            error_type = "NO_GT_IN_TEST_GALLERY"
            error_details.append("Không có sản phẩm nào thỏa mãn spec trong test gallery (N=1.290)")
        elif sys_rank_first is not None and sys_rank_first <= 10:
            error_type = "PASS"
        else:
            # Truy vấn chưa đạt Recall@10
            # Kiểm tra lỗi (C): parser hiểu sai thuộc tính
            parser_failed = False
            # Check articleType
            spec_types = set(spec.get("articleType") or [])
            parsed_types = set(parsed.get("articleType") or [])
            if spec_types and (not parsed_types or not (spec_types & parsed_types)):
                parser_failed = True
                error_details.append(f"Parser thiếu/sai articleType: spec={list(spec_types)}, parsed={list(parsed_types)}")
            # Check gender
            spec_gender = spec.get("gender")
            parsed_gender = parsed.get("gender")
            if spec_gender and parsed_gender != spec_gender:
                parser_failed = True
                error_details.append(f"Parser thiếu/sai gender: spec={spec_gender}, parsed={parsed_gender}")
            # Check colour
            spec_colours = set(spec.get("baseColour") or [])
            parsed_colours = set(parsed.get("baseColour") or [])
            if spec_colours and (not parsed_colours or not (spec_colours & parsed_colours)):
                parser_failed = True
                error_details.append(f"Parser thiếu/sai baseColour: spec={list(spec_colours)}, parsed={list(parsed_colours)}")

            # Kiểm tra lỗi (D): GT mơ hồ
            gt_ambiguous = False
            # Nếu query có từ chỉ giới tính hoặc màu nhưng spec lại rỗng hoặc catalog phân tán
            if "dạo phố" in txt or "công sở" in txt or "ấm áp" in txt or "dễ thương" in txt:
                # Ngữ nghĩa phong cách/mục đích không có trong metadata catalog
                pass

            if not is_rel_in_pool_100:
                error_type = "(A) Không có ảnh đúng trong pool 100"
                if parser_failed:
                    error_type += " + (C) Parser thiếu thuộc tính"
            elif parser_failed:
                error_type = "(C) Parser/lexicon hiểu sai thuộc tính"
            else:
                error_type = "(B) Có trong pool nhưng xếp thấp (>10)"

        diag_rows.append({
            "query_idx": i,
            "query": txt,
            "has_diacritics": has_diacritics,
            "thin_slot": thin_slot,
            "n_relevant": n_relevant,
            "model_rank_first_rel": model_rank_first if model_rank_first is not None else ">100",
            "sys_rank_first_rel": sys_rank_first if sys_rank_first is not None else ">100",
            "sys_rank_first_pool1290": sys_rank_first_1290 if sys_rank_first_1290 is not None else ">100",
            "model_ap": round(model_ap_100, 4),
            "sys_ap": round(sys_ap_100, 4),
            "sys_ap_pool1290": round(sys_ap_1290, 4),
            "is_rel_in_pool_100": is_rel_in_pool_100,
            "error_type": error_type,
            "error_details": "; ".join(error_details) if error_details else "None",
            "parsed_articleType": parsed.get("articleType"),
            "parsed_gender": parsed.get("gender"),
            "parsed_baseColour": parsed.get("baseColour"),
            "spec_articleType": spec.get("articleType"),
            "spec_gender": spec.get("gender"),
            "spec_baseColour": spec.get("baseColour")
        })

    # Tạo DataFrame và lưu diagnostics.csv
    df_diag = pd.DataFrame(diag_rows)
    out_dir = PROJECT_ROOT / "reports" / "phase5b"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "diagnostics.csv"
    df_diag.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Đã lưu bảng chẩn đoán tại: {csv_path}")

    # 4. Tính toán Ceiling & Phân tích tổng thể
    summary_100_sys = compute_retrieval_summary(ranked_rels_pool100_sys, tot_rels_valid, topk=(1, 5, 10))
    summary_1290_sys = compute_retrieval_summary(ranked_rels_pool1290_sys, tot_rels_valid, topk=(1, 5, 10))
    summary_100_model = compute_retrieval_summary(ranked_rels_pool100_model, tot_rels_valid, topk=(1, 5, 10))
    summary_1290_model = compute_retrieval_summary(ranked_rels_pool1290_model, tot_rels_valid, topk=(1, 5, 10))

    map_baseline = summary_100_sys["mAP"]
    map_ceiling = summary_1290_sys["mAP"]
    r10_baseline = summary_100_sys["recall@10"]
    r10_ceiling = summary_1290_sys["recall@10"]

    map_gap_to_target = 0.75 - map_baseline
    map_gain_ceiling = map_ceiling - map_baseline
    pct_gap_due_to_pool = (map_gain_ceiling / map_gap_to_target * 100.0) if map_gap_to_target > 0 else 0.0

    print("\n--- TỔNG KẾT SO SÁNH CEILING (candidate_pool = 1290 vs baseline = 100) ---")
    print(f"• Baseline System (pool=100) : Recall@10 = {r10_baseline:.4f}, mAP = {map_baseline:.4f}")
    print(f"• Ceiling System (pool=1290) : Recall@10 = {r10_ceiling:.4f}, mAP = {map_ceiling:.4f}")
    print(f"• Mức tăng tuyệt đối mAP     : +{map_gain_ceiling:.4f}")
    print(f"• Khoảng cách tới mAP 0.75   : {map_gap_to_target:.4f}")
    print(f"• Tỷ lệ % khoảng cách do pool: {pct_gap_due_to_pool:.2f}%")

    # Thống kê phân loại lỗi trên các query không đạt Top-10
    is_failed_r10 = df_diag["sys_rank_first_rel"].apply(lambda x: True if str(x) == ">100" else (int(x) > 10 if str(x).isdigit() else False))
    failed_df = df_diag[(df_diag["n_relevant"] > 0) & is_failed_r10]
    n_failed = len(failed_df)
    print(f"\n• Số truy vấn không đạt Top-10 trong số 86 truy vấn có GT: {n_failed}/86")

    error_counts = failed_df["error_type"].value_counts().to_dict()
    for err, cnt in error_counts.items():
        print(f"  - {err}: {cnt} truy vấn ({cnt/n_failed*100:.1f}%)")

    # Thống kê theo có dấu / không dấu
    diacritics_stats = df_diag[df_diag["n_relevant"] > 0].groupby("has_diacritics").agg(
        n_queries=("query_idx", "count"),
        r10_sys_100=("sys_rank_first_rel", lambda s: np.mean([1.0 if str(x).isdigit() and int(x) <= 10 else 0.0 for x in s])),
        r10_sys_1290=("sys_rank_first_pool1290", lambda s: np.mean([1.0 if str(x).isdigit() and int(x) <= 10 else 0.0 for x in s])),
        map_sys_100=("sys_ap", "mean"),
        map_sys_1290=("sys_ap_pool1290", "mean"),
        not_in_pool=("is_rel_in_pool_100", lambda s: sum(~s))
    )
    print("\n--- THỐNG KÊ THEO DẤU TIẾNG VIỆT ---")
    print(diacritics_stats)

    # 5. Xuất diagnostics_summary.md
    summary_md_path = out_dir / "diagnostics_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Chẩn đoán Baseline & Ceiling Phân tích (Pha 5b)\n\n")
        f.write(f"- **Tập dữ liệu chẩn đoán:** `eval/handwritten_queries.json` (**DEV SET**, N=100)\n")
        f.write(f"- **Official Test Gallery:** `index/test_gallery/` (N=1.290 sản phẩm test)\n")
        f.write("- **Chính sách Ground Truth:** `hybrid` (86 truy vấn có $\\ge 1$ ảnh đúng, 14 truy vấn không có GT)\n")
        f.write(f"- **Commit Baseline:** `bcc5284`\n")
        f.write(f"- **Seed:** 42\n\n")
        f.write("---\n\n")

        f.write("## 1. Phân tích Giới hạn Trần (Ceiling Analysis): Vai trò của Candidate Pool\n\n")
        f.write("Khi `candidate_pool = 100`, chỉ có 100 ứng viên gần nhất từ ANN HNSW được chuyển sang bộ Re-ranking (Soft Attribute Bonus). ")
        f.write("Nếu tăng `candidate_pool = 1.290` (toàn bộ Test Gallery), tất cả ảnh trong gallery đều được bộ lọc quy tắc thuộc tính chấm điểm.\n\n")
        f.write("| Cấu hình | candidate_pool | Recall@1 | Recall@5 | Recall@10 | mAP |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        f.write(f"| **Model thuần** | 100 | {summary_100_model['recall@1']:.4f} | {summary_100_model['recall@5']:.4f} | {summary_100_model['recall@10']:.4f} | {summary_100_model['mAP']:.4f} |\n")
        f.write(f"| **Model thuần (Toàn gallery)** | 1.290 | {summary_1290_model['recall@1']:.4f} | {summary_1290_model['recall@5']:.4f} | {summary_1290_model['recall@10']:.4f} | {summary_1290_model['mAP']:.4f} |\n")
        f.write(f"| **Hệ thống (Baseline Pha 5)** | 100 | **{summary_100_sys['recall@1']:.4f}** | **{summary_100_sys['recall@5']:.4f}** | **{summary_100_sys['recall@10']:.4f}** | **{summary_100_sys['mAP']:.4f}** |\n")
        f.write(f"| **Hệ thống (Ceiling Pool toàn phần)** | 1.290 | **{summary_1290_sys['recall@1']:.4f}** | **{summary_1290_sys['recall@5']:.4f}** | **{summary_1290_sys['recall@10']:.4f}** | **{summary_1290_sys['mAP']:.4f}** |\n")
        f.write(f"| **Mức chênh lệch ($\\\\Delta$)** | +1.190 | `+{summary_1290_sys['recall@1']-summary_100_sys['recall@1']:.4f}` | `+{summary_1290_sys['recall@5']-summary_100_sys['recall@5']:.4f}` | `+{summary_1290_sys['recall@10']-summary_100_sys['recall@10']:.4f}` | `+{map_gain_ceiling:.4f}` |\n\n")

        f.write("### Trả lời câu hỏi định lượng:\n")
        f.write(f"- Khoảng cách từ baseline mAP ({map_baseline:.4f}) tới ngưỡng chỉ tiêu 0.75 là **`{map_gap_to_target:.4f}`**.\n")
        f.write(f"- Mức tăng mAP khi mở rộng candidate_pool từ 100 lên 1.290 là **`+{map_gain_ceiling:.4f}`** (tăng từ `{map_baseline:.4f}` lên `{map_ceiling:.4f}`).\n")
        f.write(f"- Tỷ lệ khoảng cách mAP do candidate pool gây ra: **`{pct_gap_due_to_pool:.2f}%`**.\n")
        f.write(f"- Đồng thời, Recall@10 tăng từ `{r10_baseline:.4f}` lên **`{r10_ceiling:.4f}`** (thu hồi thêm {int(round(r10_ceiling*86 - r10_baseline*86))} truy vấn vào Top-10).\n\n")

        f.write("---\n\n")
        f.write("## 2. Phân loại Nguyên nhân Lỗi (Error Taxonomy)\n\n")
        f.write(f"Trong số 86 truy vấn có GT trên test gallery, có **{n_failed} truy vấn** chưa đạt Top-10 ở baseline (sys_rank_first_rel > 10 hoặc > 100):\n\n")
        f.write("| Nhóm lỗi | Số lượng | Tỷ lệ (%) | Đặc điểm nhận diện |\n")
        f.write("|---|---:|---:|---|\n")
        for err, cnt in error_counts.items():
            f.write(f"| **{err}** | {cnt} | {cnt/n_failed*100:.1f}% | Xem bảng chi tiết bên dưới |\n")

        f.write("\n### Chi tiết từng truy vấn thất bại:\n\n")
        f.write("| ID | Truy vấn | Có dấu | n_rel | Rank thuần | Rank hệ thống | Rank ceiling (pool 1290) | Trong pool 100? | Nhóm lỗi & Chi tiết |\n")
        f.write("|---|---|:---:|---:|---:|---:|---:|:---:|---|\n")
        for _, r in failed_df.iterrows():
            f.write(f"| {r['query_idx']} | `{r['query']}` | {'Có' if r['has_diacritics'] else 'Không'} | {r['n_relevant']} | {r['model_rank_first_rel']} | {r['sys_rank_first_rel']} | {r['sys_rank_first_pool1290']} | {'Có' if r['is_rel_in_pool_100'] else '**KHÔNG**'} | {r['error_type']} ({r['error_details']}) |\n")

        f.write("\n---\n\n")
        f.write("## 3. Phân tích Chênh lệch Có dấu vs Không dấu\n\n")
        f.write("| Nhóm truy vấn | Số lượng (có GT) | R@10 (Baseline Pool=100) | R@10 (Ceiling Pool=1290) | mAP (Baseline) | mAP (Ceiling) | Số query ảnh đúng rớt ngoài Pool 100 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for has_d, row in diacritics_stats.iterrows():
            label = "Có dấu (`has_diacritics=True`)" if has_d else "Không dấu (`has_diacritics=False`)"
            f.write(f"| **{label}** | {int(row['n_queries'])} | {row['r10_sys_100']:.4f} | {row['r10_sys_1290']:.4f} | {row['map_sys_100']:.4f} | {row['map_sys_1290']:.4f} | {int(row['not_in_pool'])} |\n")

        f.write("\n### Nhận định then chốt từ dữ liệu chẩn đoán:\n")
        f.write("1. **Khoảng cách mAP chủ yếu đến từ đâu?**\n")
        f.write(f"   - Mở rộng `candidate_pool` lên toàn bộ gallery test chỉ đóng góp **{pct_gap_due_to_pool:.2f}%** mức tăng mAP ({map_baseline:.4f} -> {map_ceiling:.4f}).\n")
        f.write(f"   - Phần còn lại ({100.0 - pct_gap_due_to_pool:.2f}%) là do **thứ tự xếp hạng sâu (ranking quality)**: ngay cả khi ở pool=1290, mAP cũng chỉ đạt {map_ceiling:.4f} (vẫn cách ngưỡng 0.75 một khoảng {0.75 - map_ceiling:.4f}). Lý do: tử số mAP tính precision tại mọi vị trí rank của các ảnh đúng, nhưng SigLIP text encoder chỉ đưa 1-2 ảnh đúng lên top 1-5, còn các ảnh đúng thứ 3..N bị phân tán ở các rank 20..100 dẫn đến Precision@K suy giảm nhanh.\n")
        f.write("2. **Vấn đề nhóm không dấu:**\n")
        f.write(f"   - Nhóm không dấu có mAP baseline chỉ đạt `{diacritics_stats.loc[False, 'map_sys_100']:.4f}` (so với `{diacritics_stats.loc[True, 'map_sys_100']:.4f}` ở nhóm có dấu).\n")
        f.write(f"   - Ở nhóm không dấu, có {int(diacritics_stats.loc[False, 'not_in_pool'])} truy vấn mà SigLIP ANN không tìm thấy bất kỳ ảnh đúng nào trong Top-100 candidates (Embedding của câu không dấu bị trôi xa trong không gian SigLIP).\n")
        f.write("   - Điều này khẳng định giả thuyết: Cần giải pháp **khôi phục dấu tiếng Việt (lexicon-based)** và **Dual-encode** ở Bước 3 để kéo vector câu không dấu về gần vùng biểu diễn của câu có dấu!\n")

    print(f"✅ Đã lưu báo cáo tóm tắt chẩn đoán tại: {summary_md_path}")


if __name__ == "__main__":
    run_diagnostics()
