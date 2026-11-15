# -*- coding: utf-8 -*-
"""Chốt eval/test_queries_heldout_v2.json. KHÔNG nạp model, KHÔNG chạy embedding, KHÔNG sửa/thêm/bớt câu.
Chỉ ghi file cuối khi TẤT CẢ tiêu chí (worksheet mục 1) đạt; thiếu một tiêu chí => thoát mã 1, không ghi file.

Ví dụ (chạy trong repo):
  python scripts/finalize_heldout_v2.py \
    --draft eval/test_queries_heldout_v2.draft.json \
    --worksheet reports/phase5c/heldout_v2_worksheet.csv \
    --id-map index/test_gallery/id_map.json --styles <đường dẫn styles.csv> \
    --lexicon-fn src.parser.lexicon:is_in_lexicon_v21 \
    --gt-fn src.eval.gt:resolve_relevant_ids \
    --out eval/test_queries_heldout_v2.json

--lexicon-fn  hàm f(query:str)->bool, True nếu câu nằm TRONG lexicon v2.1 (BẮT BUỘC).
--gt-fn       hàm g(spec:dict, catalog:DataFrame)->list[id] theo GT hybrid CHÍNH THỨC của dự án,
              spec = {"articleType","gender"(None nếu bỏ qua),"baseColour"(None nếu bỏ qua)}.
              Không có => dùng khớp đơn giản theo cụm và gắn gt_source="simple_cluster_match" (KHÔNG khuyến nghị cho bản cuối).
"""
import argparse, hashlib, importlib, json, re, sys, unicodedata
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load_fn(spec):
    m, f = spec.split(":"); return getattr(importlib.import_module(m), f)

def share(s, key):
    for m in re.finditer(r"([^:;]+):(\d+)/(\d+)\((\d+)%\)", str(s)):
        if m.group(1).strip() == key: return int(m.group(2)) / int(m.group(3))
    return None

def strip_dia(s):
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True); ap.add_argument("--worksheet", required=True)
    ap.add_argument("--id-map", required=True); ap.add_argument("--styles", required=True)
    ap.add_argument("--lexicon-fn", required=True); ap.add_argument("--gt-fn")
    ap.add_argument("--out", required=True); ap.add_argument("--id-col", default="id")
    ap.add_argument("--min-queries", type=int, default=100); ap.add_argument("--min-none", type=int, default=40)
    ap.add_argument("--min-ooc", type=float, default=0.30); ap.add_argument("--min-usage-season", type=int, default=40)
    ap.add_argument("--min-thin-article-types-cov", type=float, default=1.0)
    ap.add_argument("--allow-nrel-mismatch", action="store_true")
    a = ap.parse_args()
    if Path(a.out).exists(): sys.exit(f"Từ chối ghi đè {a.out} (file đã tồn tại; bộ truy vấn đã khóa không được sửa).")

    Q = json.load(open(a.draft, encoding="utf-8")); W = pd.read_csv(a.worksheet).set_index("spec_cluster_id")
    raw = json.load(open(a.id_map, encoding="utf-8"))
    if isinstance(raw, dict): gal = set(map(str, raw.keys()))
    else: gal = set(str(x.get("id", x.get("item_id", x.get("product_id", x))) if isinstance(x, dict) else x) for x in raw)
    S = pd.read_csv(a.styles, on_bad_lines="skip"); S[a.id_col] = S[a.id_col].astype(str); S = S[S[a.id_col].isin(gal)].copy()
    errs, warns = [], []
    if len(S) != len(gal): errs.append(f"styles.csv chỉ phủ {len(S)}/{len(gal)} id của test_gallery")
    lex = load_fn(a.lexicon_fn); gt = load_fn(a.gt_fn) if a.gt_fn else None
    gt_source = "project_gt_fn" if gt else "simple_cluster_match"
    if not gt: warns.append("Dùng GT khớp đơn giản theo cụm, không phải hàm GT hybrid của dự án.")

    seen, out = set(), []
    for q in Q:
        qid, cid = q["query_id"], q["spec_cluster_id"]
        if cid not in W.index: errs.append(f"{qid}: cụm {cid} không có trong worksheet"); continue
        w = W.loc[cid]; at, g, c = cid.split("__")
        spec = {"articleType": at, "gender": None if g in ("None", "") else g, "baseColour": c or None}
        key = q["query"].strip().lower()
        if key in seen: errs.append(f"{qid}: câu trùng")
        seen.add(key)
        has_dia = any(ord(ch) > 127 for ch in q["query"])
        if q["dia_class_author"] == "none" and (has_dia or strip_dia(q["query"]) != q["query"]): errs.append(f"{qid}: nhãn none nhưng còn dấu")
        if q["dia_class_author"] == "full" and not has_dia: errs.append(f"{qid}: nhãn full nhưng không có dấu")
        if gt: ids = [str(i) for i in gt(spec, S)]
        else:
            m = S.articleType == at
            if spec["gender"]: m &= S.gender == spec["gender"]
            if spec["baseColour"]: m &= S.baseColour == spec["baseColour"]
            ids = S[m][a.id_col].tolist()
        if not set(ids) <= gal: errs.append(f"{qid}: relevant_ids chứa id ngoài test_gallery")
        if len(ids) < 5: errs.append(f"{qid}: n_relevant={len(ids)} < 5")
        if len(ids) != int(w["n_relevant"]):
            (warns if a.allow_nrel_mismatch else errs).append(f"{qid}: n_relevant={len(ids)} lệch worksheet={int(w['n_relevant'])} ({cid})")
        sec_ok, sec_cat = False, None
        if q["secondary"]:
            kind, val = q["secondary"]["kind"], q["secondary"]["value"]
            col = "usage_5_70pct" if kind == "usage" else "season_5_70pct"
            sh = share(w[col], val); sec_ok = sh is not None and 0.05 <= sh <= 0.70
            if not sec_ok: errs.append(f"{qid}: secondary {kind}={val} ngoài 5..70% theo worksheet")
            if kind in S.columns and ids:
                cs = S[S[a.id_col].isin(ids)][kind].astype(str).str.strip().str.lower()
                sec_cat = float((cs == val.lower()).mean())
                if not 0.05 <= sec_cat <= 0.70: errs.append(f"{qid}: secondary {kind}={val} theo catalog = {sec_cat:.0%} ngoài 5..70%")
        in_lex = bool(lex(q["query"]))
        r = dict(q); r.update(relevant_ids=ids, n_relevant=len(ids), in_lexicon=in_lex, out_of_lexicon=not in_lex,
                 dia_class="none" if not has_dia else q["dia_class_author"], is_thin_slot=bool(w["is_thin_slot"]),
                 slot_kind=str(w["slot_kind"])[0] + "_slot", has_usage_season_signal=bool(sec_ok and str(w["slot_kind"]).startswith("3")),
                 secondary_share_in_catalog=sec_cat, gt_source=gt_source)
        out.append(r)

    n = len(out); D = pd.DataFrame(out)
    none_n = int((D.dia_class == "none").sum()); ooc_n = int(D.out_of_lexicon.sum()); us_n = int(D.has_usage_season_signal.sum())
    thin = W[W.is_thin_slot]; cov_at = set(thin.loc[[c for c in D[D.is_thin_slot].spec_cluster_id.unique()], "articleType"])
    thin_at = set(thin.articleType); cov_ratio = len(cov_at) / max(1, len(thin_at))
    crit = {
        f"so_cau>={a.min_queries}": (n, n >= a.min_queries),
        f"khong_dau>={a.min_none}": (none_n, none_n >= a.min_none),
        f"ngoai_lexicon>={a.min_ooc:.0%}": (f"{ooc_n}/{n}={ooc_n/max(1,n):.1%}", n > 0 and ooc_n / n >= a.min_ooc),
        f"usage_season>={a.min_usage_season}": (us_n, us_n >= a.min_usage_season),
        "phu_articleType_slot_mong": (f"{len(cov_at)}/{len(thin_at)}", cov_ratio >= a.min_thin_article_types_cov),
        "khong_co_loi_du_lieu": (len(errs), not errs)}
    for k, (v, ok) in crit.items(): print(f"{'OK ' if ok else 'LOI'} {k}: {v}")
    thin_all = set(thin.index); miss = sorted(thin_all - set(D[D.is_thin_slot].spec_cluster_id))
    print(f"Cụm mỏng phủ: {len(thin_all)-len(miss)}/{len(thin_all)} (chưa phủ {len(miss)})")
    for w_ in warns: print("CẢNH BÁO:", w_)
    for e in errs: print("LỖI:", e)
    if not all(ok for _, ok in crit.values()):
        print("\nKHÔNG ghi file cuối: còn tiêu chí chưa đạt. Không sửa câu theo kết quả model; chỉ sửa theo tiêu chí catalog/worksheet."); sys.exit(1)

    meta = dict(spec="heldout_v2", n_queries=n, draft_sha256=sha256(a.draft), worksheet_sha256=sha256(a.worksheet),
                id_map_sha256=sha256(a.id_map), styles_sha256=sha256(a.styles), gt_source=gt_source,
                criteria={k: {"value": v, "ok": ok} for k, (v, ok) in crit.items()},
                thin_clusters_uncovered=miss, used_model=False, seed=42)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": meta, "queries": out}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    h = sha256(a.out); Path(a.out + ".sha256").write_text(f"{h}  {Path(a.out).name}\n")
    print(f"\nĐã ghi {a.out}\nSHA-256: {h}\nGhi vào AGENTS/báo cáo và KHÓA tệp này. Sửa sau khi chạy model => nhãn DEV SET (Luật C.5).")
main()
