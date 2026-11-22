"""Chạy TẤT CẢ truy vấn DEV (handwritten, val_a, val_b, val_h) qua AppSearchService thật
(reference_gallery, cfg_v5_11) và đánh dấu từng truy vấn có kết quả lạc loại / lạc giới tính / lạc màu.
Không dùng held-out v2. Kết quả: all_queries.csv, flagged.txt, summary.json."""
import sys, json, os, csv, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; os.chdir(ROOT)
# CODE_ROOT: thư mục chứa app/ và src/ cần đo (mặc định: chính dự án). Dùng để so code HEAD với code hiện tại
# trên cùng dữ liệu/checkpoint/gallery.
CODE_ROOT = Path(os.environ.get("CODE_ROOT", ROOT)); sys.path.insert(0, str(CODE_ROOT))
OUT = Path(__file__).resolve().parent / os.environ.get("RUN_TAG", "current"); OUT.mkdir(exist_ok=True)
from app.engine_service import AppSearchService
from src.serve.parser_v3 import VietnameseQueryParserV3
from src.eval.metrics import is_item_relevant, load_gender_gt_policy, compute_retrieval_summary

GAL = os.environ.get("GAL", "reference_gallery"); CFG = os.environ.get("CFG", "cfg_v5_11")
import yaml, tempfile
_cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
_cfg["paths"]["checkpoint_file"] = str(ROOT / _cfg["paths"]["checkpoint_file"])
_tmp_cfg = Path(tempfile.gettempdir()) / "run4_config.yaml"; yaml.safe_dump(_cfg, open(_tmp_cfg, "w", encoding="utf-8"), allow_unicode=True)
svc = AppSearchService(index_dir=str(ROOT / "index" / GAL), config_path=str(_tmp_cfg))
import app.engine_service as _es; print("code from:", _es.__file__, flush=True)
P = VietnameseQueryParserV3()
policy = load_gender_gt_policy(); strict = set(policy["policies"]["hybrid"].get("strict_categories", []))
def L(p):
    d = json.load(open(ROOT / p, encoding="utf-8")); return d if isinstance(d, list) else d.get("queries") or d.get("pairs")
SETS = {"hand": "eval/handwritten_queries.json", "val_a": "eval/val_queries_a_synthetic.json",
        "val_b": "eval/val_queries_b_synth.json", "val_h": "eval/val_queries_b_human_v4.validated.json"}
for _ in range(10): svc.search_text_to_image("áo thun nam", CFG, 10)
rows, flagged, summ = [], [], {}
for name, path in SETS.items():
    qs = L(path); rels, tots, lat = [], [], []
    for q in qs:
        text = q.get("query") or q.get("text"); spec = q.get("spec") or {}
        res, _, ms = svc.search_text_to_image(text, CFG, 10); lat.append(ms)
        p = P.parse(text); types = [r.get("articleType") for r in res]
        flags = []
        if p["articleType"]:
            bad = [t for t in types if t not in p["articleType"]]
            if bad: flags.append(f"lac_loai({len(bad)}/10:{sorted(set(bad))[:3]})")
        elif not spec.get("articleType"): pass
        else: flags.append("khong_nhan_ra_loai")
        if p["gender"] in ("Men", "Women"):
            bg = [r.get("gender") for r in res if r.get("gender") not in (p["gender"], "Unisex")]
            if bg: flags.append(f"lac_gioi_tinh({len(bg)}/10)")
        if p["baseColour"]:
            bc = [r.get("baseColour") for r in res if r.get("baseColour") not in p["baseColour"]]
            if len(bc) > 5: flags.append(f"lac_mau({len(bc)}/10)")
        tot = sum(is_item_relevant(spec, m, "hybrid", strict) for m in svc.id_map) if spec else 0
        rel = [int(is_item_relevant(spec, r, "hybrid", strict)) for r in res] if spec else []
        if tot > 0: rels.append(rel); tots.append(tot)
        r10 = (sum(rel) / min(10, tot)) if tot > 0 else None
        if tot > 0 and r10 < 1: flags.append(f"R@10={r10:.2f}")
        rows.append([name, text, "|".join(p["articleType"]), p["gender"], "|".join(p["baseColour"]), ";".join(f"{r.get('articleType')}/{r.get('gender')}/{r.get('baseColour')}" for r in res[:3]), round(ms, 1), ";".join(flags)])
        if flags: flagged.append(f"[{name}] {text}  ->  " + " ; ".join(flags))
    s = compute_retrieval_summary(rels, tots) if rels else {}
    summ[name] = dict(n=len(qs), n_gt=len(rels), metrics=s, lat_p50=float(np.percentile(lat, 50)), lat_p95=float(np.percentile(lat, 95)))
    print(name, summ[name], flush=True)
with open(OUT / "all_queries.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["set", "query", "type", "gender", "colour", "top3(type/gender/colour)", "ms", "flags"]); w.writerows(rows)
open(OUT / "flagged.txt", "w", encoding="utf-8").write("\n".join(flagged))
json.dump(dict(gallery=GAL, cfg=CFG, summary=summ, n_flagged=len(flagged), n_total=len(rows)), open(OUT / "summary.json", "w", encoding="utf-8"), indent=1)
open(OUT / "DONE.txt", "w").write("done"); print("flagged", len(flagged), "/", len(rows))
