"""
Sinh bộ đánh giá ngoài miền từ nhãn của index/crawl_v2_gallery (KHÔNG dùng mô hình, KHÔNG xem kết quả cấu hình nào).
Seed 42. Ghi eval/crawl_v2_t2i_queries.json, eval/crawl_v2_cir_pairs.json và eval/crawl_v2_eval_set.sha256.

Text→Ảnh (2 loại):
  - "template": ghép tên loại + giới tính + màu tiếng Việt (từ app/i18n_vi), 1/3 gõ không dấu,
    ngẫu nhiên bỏ màu hoặc giới tính; spec = các thuộc tính có mặt trong câu.
  - "title": tên sản phẩm thật do thương hiệu viết (độc lập với dự án); spec = {articleType} của sản phẩm.
CIR: ảnh tham chiếu trong gallery + câu đổi màu hoặc đổi giới tính; target = loại gốc + thuộc tính mới
  (đổi giới tính giữ màu gốc, như eval/cir_eval_val.json); chỉ lấy cặp có >= 3 sản phẩm đích (trừ ảnh gốc).

Chạy MỘT lần sau khi dựng gallery; không sinh lại sau khi đã chạy đánh giá.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.i18n_vi import ARTICLE_VI, COLOR_VI, GENDER_VI  # noqa: E402
from src.serve.parser_v1 import remove_diacritics  # noqa: E402

GAL = ROOT / "index" / "crawl_v2_gallery" / "id_map.json"
OUT_T2I = ROOT / "eval" / "crawl_v2_t2i_queries.json"
OUT_CIR = ROOT / "eval" / "crawl_v2_cir_pairs.json"
OUT_SHA = ROOT / "eval" / "crawl_v2_eval_set.sha256"

N_TEMPLATE, N_TITLE, N_CIR = 150, 100, 120
MAX_PER_TYPE = 10
SKIP_COLOURS = {"Multi", "Metallic", "needs_review", "", None}
GENDER_WORD = {"Men": "nam", "Women": "nữ", "Boys": "bé trai", "Girls": "bé gái", "Unisex": "unisex"}


def vi_type(t: str) -> str:
    return ARTICLE_VI.get(t, t).lower()


def vi_col(c: str) -> str:
    return COLOR_VI.get(c, c).lower()


def main() -> None:
    rng = random.Random(42)
    items = json.load(open(GAL, encoding="utf-8"))
    for it in items:
        assert GENDER_VI.get(it["gender"]) or it["gender"] in GENDER_WORD, it["gender"]

    # ---------------- Text→Ảnh: template
    groups = Counter((it["articleType"], it["gender"], it["baseColour"]) for it in items
                     if it["baseColour"] not in SKIP_COLOURS and it["gender"] in GENDER_WORD)
    cand = sorted(g for g, n in groups.items() if n >= 5)
    rng.shuffle(cand)
    per_type, chosen = Counter(), []
    for g in cand:
        if per_type[g[0]] < MAX_PER_TYPE:
            chosen.append(g)
            per_type[g[0]] += 1
        if len(chosen) == N_TEMPLATE:
            break
    t2i = []
    for i, (t, gen, col) in enumerate(chosen):
        drop = rng.random()
        use_col, use_gen = drop >= 0.2, not (0.2 <= drop < 0.35) and gen != "Unisex"
        parts = [vi_type(t)]
        if use_gen:
            parts.append(GENDER_WORD[gen] if rng.random() < 0.6 else f"cho {GENDER_WORD[gen]}")
        if use_col:
            parts.append(f"màu {vi_col(col)}" if rng.random() < 0.7 else vi_col(col))
        text = " ".join(parts)
        unacc = rng.random() < 1 / 3
        if unacc:
            text = remove_diacritics(text)
        spec = {"articleType": [t]}
        if use_gen:
            spec["gender"] = gen
        if use_col:
            spec["baseColour"] = [col]
        t2i.append({"id": f"cv2_t{i:03d}", "kind": "template", "query": text, "has_diacritics": not unacc,
                    "spec": spec})

    # ---------------- Text→Ảnh: title (tên sản phẩm thật)
    pool = [it for it in items if remove_diacritics(it["productDisplayName"]) != it["productDisplayName"].lower()
            and 6 <= len(it["productDisplayName"]) <= 80]
    by_type = defaultdict(list)
    for it in pool:
        by_type[it["articleType"]].append(it)
    types = sorted(by_type)
    rng.shuffle(types)
    titles, k = [], 0
    while len(titles) < N_TITLE and any(by_type.values()):
        t = types[k % len(types)]
        k += 1
        if by_type[t]:
            it = by_type[t].pop(rng.randrange(len(by_type[t])))
            titles.append({"id": f"cv2_n{len(titles):03d}", "kind": "title", "query": it["productDisplayName"],
                           "has_diacritics": True, "spec": {"articleType": [it["articleType"]]},
                           "source_item": it["item_id"]})
    t2i += titles

    # ---------------- CIR
    by_tgc = defaultdict(list)
    for it in items:
        by_tgc[(it["articleType"], it["gender"], it["baseColour"])].append(it["item_id"])
    refs = [it for it in items if it["gender"] in ("Men", "Women") and it["baseColour"] not in SKIP_COLOURS]
    rng.shuffle(refs)
    cir, seen_tasks = [], Counter()
    for ref in refs:
        if len(cir) == N_CIR:
            break
        t, gen, col = ref["articleType"], ref["gender"], ref["baseColour"]
        want_gender = seen_tasks["gender"] < N_CIR // 3 and rng.random() < 0.4
        if want_gender:
            new_gen = "Women" if gen == "Men" else "Men"
            gt = [x for x in by_tgc[(t, new_gen, col)] if x != ref["item_id"]]
            if len(gt) < 3:
                continue
            mod = rng.choice([f"nhưng cho {GENDER_WORD[new_gen]}", f"mẫu dành cho {GENDER_WORD[new_gen]}",
                              f"phiên bản {GENDER_WORD[new_gen]}"])
            changed, target = "gender", {"articleType": t, "gender": new_gen, "baseColour": col}
        else:
            opts = sorted(c for (tt, gg, c), ids in by_tgc.items()
                          if tt == t and gg == gen and c != col and c not in SKIP_COLOURS and len(ids) >= 3)
            if not opts:
                continue
            new_col = rng.choice(opts)
            gt = list(by_tgc[(t, gen, new_col)])
            mod = rng.choice([f"nhưng màu {vi_col(new_col)}", f"đổi sang màu {vi_col(new_col)}",
                              f"giống vậy nhưng màu {vi_col(new_col)}"])
            changed, target = "baseColour", {"articleType": t, "gender": gen, "baseColour": new_col}
        if rng.random() < 0.25:
            mod = remove_diacritics(mod)
        seen_tasks[changed] += 1
        cir.append({"id": f"cv2_c{len(cir):03d}", "ref_product_id": ref["item_id"],
                    "ref_attrs": {"articleType": t, "gender": gen, "baseColour": col},
                    "changed_attr": changed, "modifier_vi": mod, "target_attrs": target,
                    "gt_product_ids": sorted(gt), "n_gt": len(gt)})

    meta = {"source_gallery": "index/crawl_v2_gallery", "seed": 42,
            "gallery_id_map_sha256": hashlib.sha256(GAL.read_bytes()).hexdigest()}
    json.dump({**meta, "queries": t2i}, open(OUT_T2I, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({**meta, "pairs": cir}, open(OUT_CIR, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    OUT_SHA.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in (OUT_T2I, OUT_CIR)),
                       encoding="utf-8")
    print(f"T2I: {len(t2i)} ({Counter(q['kind'] for q in t2i)}), unaccented={sum(not q['has_diacritics'] for q in t2i)}")
    print(f"CIR: {len(cir)} ({Counter(p['changed_attr'] for p in cir)})")
    print(OUT_SHA.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
