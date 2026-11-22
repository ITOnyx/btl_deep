# -*- coding: utf-8 -*-
"""Kiểm đếm tiêu chí held-out v2 (worksheet mục 1) và (tuỳ chọn) suy relevant_ids từ catalog.
Chỉ dùng thống kê catalog. KHÔNG nạp model, KHÔNG chạy embedding.
Dùng:
  python validate_heldout_v2.py --queries test_queries_heldout_v2.draft.json --worksheet heldout_v2_worksheet.csv
  (tuỳ chọn) --id-map index/test_gallery/id_map.json --styles <styles.csv> --out eval/test_queries_heldout_v2.json
  (tuỳ chọn) --lexicon-fn pkg.module:ham   # ham(query:str)->bool, True nếu câu nằm TRONG lexicon v2.1
"""
import argparse, json, re, sys, importlib
import pandas as pd

def parse_share(s, key):
    for m in re.finditer(r"([^:;]+):(\d+)/(\d+)\((\d+)%\)", str(s)):
        if m.group(1).strip()==key: return int(m.group(2))/int(m.group(3))
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--queries",required=True); ap.add_argument("--worksheet",required=True)
    ap.add_argument("--id-map"); ap.add_argument("--styles"); ap.add_argument("--out")
    ap.add_argument("--lexicon-fn")
    a=ap.parse_args()
    Q=json.load(open(a.queries,encoding="utf-8")); W=pd.read_csv(a.worksheet)
    W=W.set_index("spec_cluster_id"); errs=[]; rows=[]
    lex=None
    if a.lexicon_fn:
        m,f=a.lexicon_fn.split(":"); lex=getattr(importlib.import_module(m),f)
    seen=set()
    for q in Q:
        cid=q["spec_cluster_id"]
        if cid not in W.index: errs.append(f"{q['query_id']}: cụm {cid} không có trong worksheet"); continue
        w=W.loc[cid]
        if q["query"].strip().lower() in seen: errs.append(f"{q['query_id']}: câu trùng")
        seen.add(q["query"].strip().lower())
        has_dia=any(ord(c)>127 for c in q["query"])
        if q["dia_class_author"]=="none" and has_dia: errs.append(f"{q['query_id']}: nhãn none nhưng còn dấu")
        if q["dia_class_author"]=="full" and not has_dia: errs.append(f"{q['query_id']}: nhãn full nhưng không có dấu")
        sec_ok=False
        if q["secondary"]:
            col="usage_5_70pct" if q["secondary"]["kind"]=="usage" else "season_5_70pct"
            sh=parse_share(w[col],q["secondary"]["value"])
            sec_ok = sh is not None and 0.05<=sh<=0.70
            if not sec_ok: errs.append(f"{q['query_id']}: secondary {q['secondary']} ngoài 5..70% ở {cid}")
        is3=str(w["slot_kind"]).startswith("3")
        in_lex=None if lex is None else bool(lex(q["query"]))
        q["in_lexicon"]=in_lex
        rows.append(dict(id=q["query_id"],cid=cid,n_rel=int(w["n_relevant"]),thin=bool(w["is_thin_slot"]),
            none=q["dia_class_author"]=="none",ooc_intent=q["ooc_intent_author"],ooc_real=None if in_lex is None else (not in_lex),
            us=bool(sec_ok and is3),slot=str(w["slot_kind"])[0]))
    D=pd.DataFrame(rows); n=len(D)
    print(f"Tổng câu: {n}  (yêu cầu >=100)  -> {'OK' if n>=100 else 'THIẾU'}")
    print(f"n_relevant>=5 cho mọi câu: {'OK' if (D.n_rel>=5).all() else 'LỖI'}")
    print(f"Không dấu: {D.none.sum()} (yêu cầu >=40) -> {'OK' if D.none.sum()>=40 else 'THIẾU'}")
    if D.ooc_real.notna().all():
        k=int(D.ooc_real.sum()); print(f"Ngoài lexicon v2.1 (TÍNH THẬT): {k}/{n}={k/n:.0%} (>=30%) -> {'OK' if k/n>=.3 else 'THIẾU'}")
    else:
        k=int(D.ooc_intent.sum()); print(f"Ngoài lexicon: CHƯA TÍNH THẬT (thiếu --lexicon-fn). Ý định người viết: {k}/{n}={k/n:.0%}. KHÔNG dùng số này làm bằng chứng.")
    print(f"Câu có tín hiệu usage/season (3-slot, 5..70%): {D.us.sum()} (>=40) -> {'OK' if D.us.sum()>=40 else 'THIẾU'}")
    thin_all=W[W.is_thin_slot]; cov=set(D[D.thin].cid)
    print(f"Phủ cụm slot mỏng: {len(cov)}/{len(thin_all)} cụm; articleType mỏng được phủ: "
          f"{thin_all.articleType.nunique() and len(set(thin_all.loc[list(cov),'articleType']))}/{thin_all.articleType.nunique()}")
    miss=sorted(set(thin_all.index)-cov); print(f"Cụm mỏng chưa phủ ({len(miss)}), 10 đầu: {miss[:10]}")
    print("Phân bố slot:",D.slot.value_counts().to_dict(),"| mỏng:",int(D.thin.sum()),"| không mỏng:",int((~D.thin).sum()))
    if errs: print("\nLỖI:"); [print(" -",e) for e in errs]
    if a.id_map and a.styles:
        ids=json.load(open(a.id_map,encoding="utf-8")); ids=list(ids.keys()) if isinstance(ids,dict) else list(ids)
        S=pd.read_csv(a.styles); S["id"]=S["id"].astype(str); S=S[S["id"].isin(set(map(str,ids)))]
        bad=0
        for q in Q:
            at,g,c=q["spec_cluster_id"].split("__"); m=S.articleType==at
            if g not in("None",""): m&=S.gender==g
            if c: m&=S.baseColour==c
            q["relevant_ids"]=S[m]["id"].tolist()
            if len(q["relevant_ids"])!=int(W.loc[q["spec_cluster_id"],"n_relevant"]): bad+=1; print("n_relevant lệch:",q["query_id"],len(q["relevant_ids"]))
        print(f"Suy GT từ catalog (ĐƠN GIẢN theo cụm; chưa áp chính sách hybrid Unisex của dự án): lệch {bad} câu")
        print("LƯU Ý: GT hybrid/strict/relaxed chính thức phải sinh bằng hàm GT sẵn có của dự án, không dùng bản đơn giản này.")
    if a.out: json.dump(Q,open(a.out,"w",encoding="utf-8"),ensure_ascii=False,indent=1); print("Ghi",a.out)
    sys.exit(1 if errs or n<100 else 0)
main()
