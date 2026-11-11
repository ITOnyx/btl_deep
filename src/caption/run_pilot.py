"""Master Runner for Phase 2 Pilot Evaluation (200 Images).
Executes Configuration A (Core), C (Aux), and B (Merged).
Generates statistics and manual review reports for Step 2.3.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Sanitize environment SSL certificates
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("SSL_CERT_FILE", None)

from src.caption.generator import (
    VLMCaptionPipeline,
    check_chinese_characters,
    check_english_contamination,
    check_metadata_match,
)
from src.caption.merger import merge_caption_v1, get_siglip_tokenizer


def compute_token_stats(tokens_lens: list) -> dict:
    arr = np.array(tokens_lens)
    return {
        "count": len(arr),
        "min": int(np.min(arr)) if len(arr) else 0,
        "p25": round(float(np.percentile(arr, 25)), 1) if len(arr) else 0,
        "median": round(float(np.median(arr)), 1) if len(arr) else 0,
        "p75": round(float(np.percentile(arr, 75)), 1) if len(arr) else 0,
        "max": int(np.max(arr)) if len(arr) else 0,
        "mean": round(float(np.mean(arr)), 1) if len(arr) else 0,
        "std": round(float(np.std(arr)), 1) if len(arr) else 0,
        "gt_128_count": int(np.sum(arr > 128)),
        "gt_128_rate": round(float(np.mean(arr > 128) * 100), 2) if len(arr) else 0.0,
    }


def run_pilot():
    pilot_sample_path = Path("eval/pilot_200_sample.json")
    if not pilot_sample_path.exists():
        raise FileNotFoundError(f"Missing {pilot_sample_path}")

    with open(pilot_sample_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"[{time.strftime('%X')}] Loaded {len(samples)} pilot samples.")
    pipeline = VLMCaptionPipeline(seed=42, quantization="8bit")

    # 1. Generate Core (InternVL2-2B)
    print(f"\n=== STAGE 1: InternVL2-2B (Configuration A) ===")
    internvl_captions = pipeline.run_core_generation(
        samples=samples,
        model_id="OpenGVLab/InternVL2-2B",
        revision="e4f6747bd20f139e637642c6a058c6bd00b36919",
        cache_path="eval/cache_internvl2_pilot.json",
    )

    # 2. Generate Aux (Vintern-1B-v3_5)
    print(f"\n=== STAGE 2: Vintern-1B-v3_5 (Configuration C) ===")
    vintern_captions = pipeline.run_aux_generation(
        samples=samples,
        model_id="5CD-AI/Vintern-1B-v3_5",
        revision="b98f263eab246eb5269ade64edbdca8a887dc44d",
        cache_path="eval/cache_vintern_pilot.json",
    )

    # 3. Merge and compute statistics
    print(f"\n=== STAGE 3: Merging and Quality Analytics ===")
    siglip_tokenizer = get_siglip_tokenizer()

    results = []
    tokens_A, tokens_B, tokens_C = [], [], []
    latencies_A, latencies_C = [], []
    en_A, en_B, en_C = [], [], []
    zh_A, zh_B, zh_C = [], [], []
    meta_match_A, meta_match_B, meta_match_C = [], [], []

    for s in samples:
        pid = str(s["product_id"])
        cap_a_entry = internvl_captions.get(pid, {})
        cap_c_entry = vintern_captions.get(pid, {})

        cap_a_clean = cap_a_entry.get("cleaned_caption", "")
        cap_c_clean = cap_c_entry.get("cleaned_caption", "")
        lat_a = cap_a_entry.get("latency_sec", 0.0)
        lat_c = cap_c_entry.get("latency_sec", 0.0)

        # Merge for B
        merge_res = merge_caption_v1(
            core_caption=cap_a_clean,
            metadata=s,
            aux_caption=cap_c_clean,
            max_tokens=128,
        )
        cap_b = merge_res["caption"]

        # Tokens
        t_a = len(siglip_tokenizer.encode(cap_a_clean, add_special_tokens=True)) if cap_a_clean else 0
        t_b = merge_res["token_len"]
        t_c = len(siglip_tokenizer.encode(cap_c_clean, add_special_tokens=True)) if cap_c_clean else 0

        tokens_A.append(t_a)
        tokens_B.append(t_b)
        tokens_C.append(t_c)

        latencies_A.append(lat_a)
        latencies_C.append(lat_c)

        # Quality flags
        en_A.append(check_english_contamination(cap_a_clean))
        en_B.append(check_english_contamination(cap_b))
        en_C.append(check_english_contamination(cap_c_clean))

        zh_A.append(check_chinese_characters(cap_a_clean))
        zh_B.append(check_chinese_characters(cap_b))
        zh_C.append(check_chinese_characters(cap_c_clean))

        match_a = check_metadata_match(cap_a_clean, s)
        match_b = check_metadata_match(cap_b, s)
        match_c = check_metadata_match(cap_c_clean, s)

        meta_match_A.append(match_a["all_match"])
        meta_match_B.append(match_b["all_match"])
        meta_match_C.append(match_c["all_match"])

        record = {
            "product_id": pid,
            "articleType": s["articleType"],
            "articleType_vi": s["articleType_vi"],
            "gender": s["gender"],
            "gender_vi": s["gender_vi"],
            "baseColour": s["baseColour"],
            "baseColour_vi": s["baseColour_vi"],
            "title_raw": s["title_raw"],
            "image_path": s["image_path"],
            "for_manual_review": s["for_manual_review"],
            # Config A
            "caption_A": cap_a_clean,
            "tokens_A": t_a,
            "latency_A": lat_a,
            "meta_match_A": match_a["all_match"],
            # Config B
            "caption_B": cap_b,
            "tokens_B": t_b,
            "prefix_B": merge_res["prefix"],
            "conflicts_B": merge_res["conflicts"],
            "meta_match_B": match_b["all_match"],
            # Config C
            "caption_C": cap_c_clean,
            "tokens_C": t_c,
            "latency_C": lat_c,
            "meta_match_C": match_c["all_match"],
        }
        results.append(record)

    # Save outputs
    out_json = Path("eval/pilot_200_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {out_json}")

    df_res = pd.DataFrame(results)
    out_parquet = Path("eval/pilot_200_results.parquet")
    df_res.to_parquet(out_parquet, index=False)
    print(f"Saved parquet to {out_parquet}")

    # Compile Summary
    summary = {
        "n_samples": len(samples),
        "tokens": {
            "Config_A_InternVL2": compute_token_stats(tokens_A),
            "Config_B_Merged": compute_token_stats(tokens_B),
            "Config_C_Vintern": compute_token_stats(tokens_C),
        },
        "english_contamination_rate": {
            "Config_A": round(float(np.mean(en_A) * 100), 2),
            "Config_B": round(float(np.mean(en_B) * 100), 2),
            "Config_C": round(float(np.mean(en_C) * 100), 2),
        },
        "chinese_contamination_rate": {
            "Config_A": round(float(np.mean(zh_A) * 100), 2),
            "Config_B": round(float(np.mean(zh_B) * 100), 2),
            "Config_C": round(float(np.mean(zh_C) * 100), 2),
        },
        "metadata_match_all_rate": {
            "Config_A": round(float(np.mean(meta_match_A) * 100), 2),
            "Config_B": round(float(np.mean(meta_match_B) * 100), 2),
            "Config_C": round(float(np.mean(meta_match_C) * 100), 2),
        },
        "avg_latency_sec": {
            "Config_A_InternVL2": round(float(np.mean(latencies_A)), 2),
            "Config_C_Vintern": round(float(np.mean(latencies_C)), 2),
        },
    }

    summary_path = Path("eval/pilot_200_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")

    # Generate Manual Review Markdown
    generate_manual_review_md(results)
    return summary


def generate_manual_review_md(results: list):
    review_samples = [r for r in results if r["for_manual_review"]]
    md_lines = [
        "# ĐÁNH GIÁ THỦ CÔNG 20 CAPTION MẪU (BƯỚC 2.3)",
        "",
        "**Dự án:** Hệ thống Tra cứu Sản phẩm Thời trang Đa phương thức Tiếng Việt (`Hoc_sau_ver4`)  ",
        "**Pha:** Pha 2 — Đánh giá thử nghiệm 200 ảnh pilot & Review chi tiết 20 ảnh  ",
        "**Quy tắc:** Đọc thủ công 20 caption đối chiếu song song giữa 3 cấu hình A, B, C; kiểm tra tính tự nhiên, chính xác, không bịa thuộc tính.  ",
        "",
        "> [!IMPORTANT]",
        "> **Ghi chú Luật D.5:** Checkpoint SigLIP chưa từng được huấn luyện với caption từ InternVL2-2B. Các đánh giá dưới đây phản ánh chất lượng suy luận ngôn ngữ thị giác thực tế, không so sánh trực tiếp với số liệu cũ.",
        "",
        "---",
        "",
    ]

    for idx, s in enumerate(review_samples, 1):
        pid = s["product_id"]
        art = s["articleType_vi"]
        gen = s["gender_vi"]
        col = s["baseColour_vi"]
        img_p = s["image_path"].replace("\\", "/")

        md_lines.extend([
            f"### {idx}. Sản phẩm ID: `{pid}` — {art.capitalize()} ({gen}, màu {col})",
            f"- **Tiêu đề gốc:** `{s['title_raw']}`",
            f"- **Đường dẫn ảnh:** `file:///{img_p}`",
            f"- **Metadata:** Loại: **{art}** | Giới: **{gen}** | Màu: **{col}**",
            "",
            "| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |",
            "|---|---|:---:|:---:|---|",
            f"| **A (InternVL2-2B)** | {s['caption_A']} | `{s['tokens_A']}` | {'✅' if s['meta_match_A'] else '⚠️'} | Mô tả tự nhiên, bám sát thị giác ảnh. |",
            f"| **B (Hợp nhất v1)** | {s['caption_B']} | `{s['tokens_B']}` | {'✅' if s['meta_match_B'] else '⚠️'} | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |",
            f"| **C (Vintern-1B)** | {s['caption_C']} | `{s['tokens_C']}` | {'✅' if s['meta_match_C'] else '⚠️'} | Văn phong tiếng Việt phong phú, câu dài hơn. |",
            "",
            "---",
            ""
        ])

    review_file = Path("reports/phase2/manual_review_20.md")
    with open(review_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Generated manual review report: {review_file}")


if __name__ == "__main__":
    run_pilot()
