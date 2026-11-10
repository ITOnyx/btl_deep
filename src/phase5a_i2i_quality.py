"""
Phase 5a - Image->Image quality: serving.i2i_mode "fast" (1 view) vs "accurate" (Batch-2 TTA flip).

Protocol (config.eval.quality_device=cpu, quality_precision=float32):
  - queries  : all in_scope items with split == "test" (n=1290), image loaded from disk
  - gallery  : only split == "test" items (stored index_v2 embeddings, mean-centred as in serving)
  - A) paired self-match (target = same product): Recall@1/5/10, mAP (single target => AP=1/rank), MRR
  - B) attribute NN, self excluded (relevance = same articleType + compatible gender):
       P@10, Hit@10, AP@10 normalised by min(R,10)   [AGENTS.md: attribute relevance => no Recall@10]
  - C) top-10 overlap fast vs accurate (self excluded)
Resumable: raw query features cached to reports/phase4/_cache_i2i_feats.npz.
Output: reports/phase4/phase5a_i2i_quality.json
"""
import sys, json, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

from src.search_pipeline import VisualSearchEngine

CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
torch.manual_seed(CFG["eval"]["seed"]); np.random.seed(CFG["eval"]["seed"])
CACHE = ROOT / "reports/phase4/_cache_i2i_feats.npz"
OUT = ROOT / "reports/phase4/phase5a_i2i_quality.json"


def ap_at_k(retrieved, relevant, k=10):
    if not relevant:
        return None
    hits, s = 0, 0.0
    for r, rid in enumerate(retrieved[:k], start=1):
        if rid in relevant:
            hits += 1
            s += hits / r
    return s / min(len(relevant), k)


def main():
    eng = VisualSearchEngine(device="cpu")
    assert eng.device == "cpu" and eng.model is not None
    df = eng.df.reset_index(drop=True)
    test_mask = (df["split"] == "test").values
    tidx = np.where(test_mask)[0]
    n = len(tidx)
    print(f"test queries/gallery: {n}")
    ids = [str(df.iloc[i]["item_id"]) for i in tidx]

    # raw features: orig + flip, float32
    f_orig = np.zeros((n, 1152), np.float32); f_flip = np.zeros((n, 1152), np.float32)
    done = 0
    if CACHE.exists():
        z = np.load(CACHE)
        f_orig, f_flip, done = z["orig"], z["flip"], int(z["done"])
        print(f"resume from {done}")
    t_start = time.time()
    for k in range(done, n):
        p = ROOT / "data/fashion-dataset/images" / f"{ids[k]}.jpg"
        im = Image.open(p).convert("RGB")
        batch = torch.stack([eng.img_transform(im), eng.img_transform(ImageOps.mirror(im))])
        with torch.no_grad():
            feats = eng.model.encode_image(batch).float().numpy()
        f_orig[k], f_flip[k] = feats[0], feats[1]
        if (k + 1) % 50 == 0 or k + 1 == n:
            np.savez(CACHE, orig=f_orig, flip=f_flip, done=k + 1)
            el = time.time() - t_start
            print(f"  {k+1}/{n}  elapsed {el:.0f}s", flush=True)

    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    v_fast = norm(f_orig)
    v_acc = norm(norm(f_orig) + norm(f_flip))
    qc = {"fast": eng.scorer.center_vectors(v_fast), "accurate": eng.scorer.center_vectors(v_acc)}
    G = eng.emb_img_c[tidx]  # gallery = test items only

    arts = df.iloc[tidx]["articleType"].values
    gens = df.iloc[tidx]["gender"].values
    compat = eng.gender_compat
    rel_sets = []
    for i in range(n):
        cg = set(compat.get(gens[i], [gens[i]]))
        rel_sets.append({j for j in range(n) if j != i and arts[j] == arts[i] and gens[j] in cg})

    res = {"n_queries": n, "gallery_size": n, "device": "cpu", "precision": "float32",
           "protocol": __doc__.strip().splitlines()[2:12]}
    tops = {}
    for mode in ("fast", "accurate"):
        S = qc[mode] @ G.T
        # A) paired self-match
        ranks = np.array([1 + int((S[i] > S[i, i]).sum()) for i in range(n)])  # strict '>' => optimistic ties
        ranks_cons = np.array([1 + int((S[i] >= S[i, i]).sum()) - 1 + 0 for i in range(n)])  # conservative ties
        rk = ranks_cons
        A = {"recall@1": float(np.mean(rk <= 1)), "recall@5": float(np.mean(rk <= 5)),
             "recall@10": float(np.mean(rk <= 10)), "mAP": float(np.mean(1.0 / rk)),
             "mrr": float(np.mean(1.0 / rk)), "n_rank_gt1": int((rk > 1).sum()),
             "tie_rule": "conservative (>=)"}
        # B) NN excluding self
        S2 = S.copy(); np.fill_diagonal(S2, -1e9)
        order = np.argsort(-S2, axis=1)[:, :10]
        tops[mode] = order
        p10, hit, ap = [], [], []
        for i in range(n):
            rel = rel_sets[i]
            if not rel:
                continue
            retr = list(order[i])
            h = sum(1 for r in retr if r in rel)
            p10.append(h / 10.0); hit.append(1.0 if h > 0 else 0.0); ap.append(ap_at_k(retr, rel, 10))
        B = {"P@10": float(np.mean(p10)), "Hit@10": float(np.mean(hit)), "AP@10": float(np.mean(ap)),
             "n_valid_queries": len(p10)}
        res[mode] = {"A_paired_self_match": A, "B_attribute_nn_excl_self": B}
        print(mode, json.dumps(res[mode], indent=1))

    ov = [len(set(tops["fast"][i]) & set(tops["accurate"][i])) / 10.0 for i in range(n)]
    top1 = [int(tops["fast"][i][0] == tops["accurate"][i][0]) for i in range(n)]
    cos_q = np.sum(v_fast * v_acc, axis=1)
    res["C_fast_vs_accurate"] = {"top10_overlap_mean": float(np.mean(ov)),
                                 "top10_identical_frac": float(np.mean([o == 1.0 for o in ov])),
                                 "top1_agreement": float(np.mean(top1)),
                                 "query_vec_cosine_mean": float(cos_q.mean()), "query_vec_cosine_min": float(cos_q.min())}
    f, a = res["fast"], res["accurate"]
    res["delta_accurate_minus_fast"] = {
        "A_recall@1": a["A_paired_self_match"]["recall@1"] - f["A_paired_self_match"]["recall@1"],
        "A_recall@5": a["A_paired_self_match"]["recall@5"] - f["A_paired_self_match"]["recall@5"],
        "A_recall@10": a["A_paired_self_match"]["recall@10"] - f["A_paired_self_match"]["recall@10"],
        "A_mAP": a["A_paired_self_match"]["mAP"] - f["A_paired_self_match"]["mAP"],
        "B_P@10": a["B_attribute_nn_excl_self"]["P@10"] - f["B_attribute_nn_excl_self"]["P@10"],
        "B_Hit@10": a["B_attribute_nn_excl_self"]["Hit@10"] - f["B_attribute_nn_excl_self"]["Hit@10"],
        "B_AP@10": a["B_attribute_nn_excl_self"]["AP@10"] - f["B_attribute_nn_excl_self"]["AP@10"],
    }
    res["caveat"] = ("Gallery embeddings in index_v2 were built by the original pipeline (not recomputed here); "
                     "paired self-match is near-ceiling by construction (query image == gallery image).")
    res["lib"] = {"torch": torch.__version__, "numpy": np.__version__, "seed": CFG["eval"]["seed"]}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", OUT)
    print(json.dumps(res["C_fast_vs_accurate"], indent=1)); print(json.dumps(res["delta_accurate_minus_fast"], indent=1))


if __name__ == "__main__":
    main()
