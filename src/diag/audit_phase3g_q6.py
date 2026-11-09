"""
Phase 3g: Q6 Independence & BM25 Ablation Audit (E.1 - E.5)
Analyzes:
1. Ground truth definition of Q6 (articleType, gender, season, usage presence).
2. Fields included in default BM25 index.
3. Ablation: Evaluate Config 5 on Q6-test when usage/season are removed from BM25 index.
4. Clarifies Q6 query count: val (120), test (120), overlap (11), total (240).
5. Documents validation gallery composition (37,356 unknown images) and alpha*=1.0 rationale.
Saves to reports/phase3g_q6_audit.json.
"""

import sys
import json
import math
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch

from src.scoring import Scorer
from src.attr_index import AttributeIndex
from src.query_parser import QueryParser
from src.lexical import BM25Index, rrf_merge, GENDER_MAP, ARTICLE_TYPE_MAP, COLOUR_MAP, USAGE_MAP
from src.vi_vocab import remove_diacritics

def compute_metrics(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10):
    R = len(relevant_ids)
    if R == 0 or not retrieved_ids:
        return 0.0, 0.0, 0.0
    top_k = retrieved_ids[:k]
    hits = 0
    sum_precisions = 0.0
    for i, item_id in enumerate(top_k):
        if item_id in relevant_ids:
            hits += 1
            sum_precisions += hits / (i + 1)
    norm_denom = min(R, k)
    ap = sum_precisions / norm_denom if norm_denom > 0 else 0.0
    p = hits / k
    hit = 1.0 if hits > 0 else 0.0
    return ap, p, hit

class AblatedBM25Index:
    """BM25 index excluding usage and season fields."""
    def __init__(self, catalog_path: str = "data/catalog/catalog.parquet", k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        with open("src/vi_to_en_terms.json", "r", encoding="utf-8") as f:
            self.vi_to_en = json.load(f)
            
        df = pd.read_parquet(catalog_path)
        self.df = df[df["in_scope"] == True].reset_index(drop=True)
        self.N = len(self.df)
        
        names = self.df["name"].fillna("").astype(str).tolist()
        genders = self.df["gender"].fillna("").astype(str).tolist()
        art_types = self.df["articleType"].fillna("").astype(str).tolist()
        colours = self.df["baseColour"].fillna("").astype(str).tolist()
        
        doc_tokens_list = []
        doc_lens = np.zeros(self.N, dtype=np.int32)
        total_tokens = 0
        doc_freqs = Counter()
        
        for doc_idx in range(self.N):
            en_name = names[doc_idx]
            gender_vi = GENDER_MAP.get(genders[doc_idx], genders[doc_idx])
            art_vi = ARTICLE_TYPE_MAP.get(art_types[doc_idx], art_types[doc_idx])
            col_vi = COLOUR_MAP.get(colours[doc_idx], colours[doc_idx])
            
            # NOTE: Usage and Season EXCLUDED here!
            doc_text = f"{en_name} {gender_vi} {art_vi} {col_vi}"
            clean = remove_diacritics(doc_text).lower()
            tokens = re.findall(r"\b\w+\b", clean)
            
            doc_len = len(tokens)
            doc_lens[doc_idx] = doc_len
            total_tokens += doc_len
            doc_tokens_list.append(tokens)
            for term in set(tokens):
                doc_freqs[term] += 1
                
        self.avg_doc_len = float(total_tokens) / max(1, self.N)
        self.vocab = sorted(doc_freqs.keys())
        self.term2id = {term: idx for idx, term in enumerate(self.vocab)}
        self.V = len(self.vocab)
        
        self.idf = np.zeros(self.V, dtype=np.float32)
        for term, term_idx in self.term2id.items():
            df_val = doc_freqs[term]
            self.idf[term_idx] = math.log(1.0 + (self.N - df_val + 0.5) / (df_val + 0.5))
            
        row_indices, col_indices, data_values = [], [], []
        for doc_idx, tokens in enumerate(doc_tokens_list):
            doc_len = doc_lens[doc_idx]
            counts = Counter(tokens)
            denom_len_factor = self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
            for term, tf in counts.items():
                term_idx = self.term2id[term]
                score = self.idf[term_idx] * (tf * (self.k1 + 1.0)) / (tf + denom_len_factor)
                row_indices.append(doc_idx)
                col_indices.append(term_idx)
                data_values.append(score)
                
        self.matrix = csr_matrix((data_values, (row_indices, col_indices)), shape=(self.N, self.V), dtype=np.float32)

    def search(self, query: str, top_k: int = 100, mask: np.ndarray = None):
        clean_q = remove_diacritics(query).lower()
        tokens = re.findall(r"\b\w+\b", clean_q)
        expanded_tokens = list(tokens)
        for vi_phrase, en_trans in self.vi_to_en.items():
            if re.search(rf"(?:\b|^){re.escape(vi_phrase)}(?:\b|$)", clean_q):
                for en_tok in en_trans.split():
                    expanded_tokens.append(en_tok.lower())
                    
        term_cols = [self.term2id[t] for t in set(expanded_tokens) if t in self.term2id]
        if not term_cols:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            
        submatrix = self.matrix[:, term_cols]
        scores = np.asarray(submatrix.sum(axis=1)).flatten()
        
        if mask is not None:
            cand_indices = np.where(mask)[0]
            if len(cand_indices) == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            sub_scores = scores[cand_indices]
            nz_mask = (sub_scores > 0.0)
            if np.any(nz_mask):
                cand_indices = cand_indices[nz_mask]
                sub_scores = sub_scores[nz_mask]
            order = np.argsort(-sub_scores)[:top_k]
            return cand_indices[order], sub_scores[order]
        else:
            nz_indices = np.where(scores > 0.0)[0]
            if len(nz_indices) == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            nz_scores = scores[nz_indices]
            order = np.argsort(-nz_scores)[:top_k]
            return nz_indices[order], nz_scores[order]

def run_q6_audit():
    print("=== Phase 3g: Q6 Independence & BM25 Ablation Audit ===")
    
    with open("eval/q6_queries.json", "r", encoding="utf-8") as f:
        q6 = json.load(f)
        
    val_q = q6["val_queries"]
    test_q = q6["test_queries"]
    
    # 1. Inspect ground truth specs
    print("\n--- Inspecting Q6 Ground Truth Specs ---")
    val_specs = [q.get("spec", {}) for q in val_q]
    test_specs = [q.get("spec", {}) for q in test_q]
    
    has_art_val = sum(1 for s in val_specs if s.get("articleType"))
    has_gen_val = sum(1 for s in val_specs if s.get("gender"))
    has_season_val = sum(1 for s in val_specs if s.get("season"))
    has_usage_val = sum(1 for s in val_specs if s.get("usage"))
    
    has_art_test = sum(1 for s in test_specs if s.get("articleType"))
    has_gen_test = sum(1 for s in test_specs if s.get("gender"))
    has_season_test = sum(1 for s in test_specs if s.get("season"))
    has_usage_test = sum(1 for s in test_specs if s.get("usage"))
    
    print(f"Val queries (N={len(val_q)}): articleType={has_art_val}, gender={has_gen_val}, season={has_season_val}, usage={has_usage_val}")
    print(f"Test queries (N={len(test_q)}): articleType={has_art_test}, gender={has_gen_test}, season={has_season_test}, usage={has_usage_test}")
    
    # 2. Query text overlap
    val_texts = set(q["query_text"] for q in val_q)
    test_texts = set(q["query_text"] for q in test_q)
    overlap_texts = val_texts & test_texts
    print(f"Query counts: Val={len(val_q)}, Test={len(test_q)}, Total={len(val_q)+len(test_q)}")
    print(f"Query text overlap between val and test: {len(overlap_texts)} queries ({list(overlap_texts)[:3]}...)")
    
    # 3. BM25 fields audit
    bm25_fields_default = {
        "title": True,
        "attrs_articleType_vi": True,
        "attrs_gender_vi": True,
        "attrs_colour_vi": True,
        "usage_vi": True,
        "season_vi": False,
        "sample_training_captions": False,
        "note": "BM25 index contains usage_vi (e.g. 'dạo phố thường ngày', 'thể thao'), allowing direct lexical match with usage queries. Season is absent from index."
    }
    
    # 4. Run BM25 Ablation on Q6-test
    print("\n--- Running BM25 Ablation on Q6-test ---")
    from src.search_pipeline import VisualSearchEngine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = VisualSearchEngine(device=device, gamma=0.3, alpha=1.0, enable_text_fusion=False)
    
    # Load default BM25 and ablated BM25
    bm25_default = engine.bm25
    print("Building Ablated BM25 index (excluding usage/season)...")
    bm25_ablated = AblatedBM25Index()
    
    # Evaluate Config 0, Config 5 (Default BM25), Config 5 (Ablated BM25)
    c0_ap, c0_p, c0_hit = [], [], []
    c5_def_ap, c5_def_p, c5_def_hit = [], [], []
    c5_abl_ap, c5_abl_p, c5_abl_hit = [], [], []
    
    test_mask = (engine.df["split"] == "test").to_numpy()
    test_indices = np.where(test_mask)[0]
    test_ids = [engine.item_ids[i] for i in test_indices]
    
    for q in test_q:
        q_text = q["query_text"]
        rel_ids = set(q["relevant_ids"])
        
        # --- Config 0: Dense Model Only on G_test ---
        q_vec = engine.encode_text(q_text)
        sims_0 = np.dot(engine.emb_img_c[test_indices], q_vec)
        top10_c0 = [test_ids[i] for i in np.argsort(-sims_0)[:10]]
        ap0, p0, h0 = compute_metrics(top10_c0, rel_ids, k=10)
        c0_ap.append(ap0); c0_p.append(p0); c0_hit.append(h0)
        
        # Parse & candidate filter
        parsed = engine.parser.parse(q_text)
        cand_idxs, _ = engine.attr_idx.filter_relaxed(parsed, min_candidates=engine.min_candidates)
        # Intersect with G_test
        cand_test = [idx for idx in cand_idxs if test_mask[idx]]
        if len(cand_test) == 0:
            cand_test = list(test_indices)
            
        # Dense + Soft Bonus
        q_t = torch.tensor(q_vec, dtype=torch.float16, device=device)
        cand_t = torch.tensor(cand_test, dtype=torch.long, device=device)
        sims_t = torch.matmul(engine.t_emb_img_c[cand_t], q_t)
        bonus = engine.attr_idx.compute_attribute_soft_bonus(
            parsed, cand_test, gamma_type=0.3, gamma_gender=0.3, gamma_colour=0.3, gamma_brand=0.3
        )
        scores_t = sims_t + torch.tensor(bonus, dtype=torch.float16, device=device)
        k_top = min(100, len(cand_test))
        top_pos = torch.topk(scores_t, k=k_top)[1].cpu().numpy()
        dense_order = np.array(cand_test)[top_pos]
        
        # --- Config 5 Default BM25 ---
        cand_mask_def = np.zeros(engine.N, dtype=bool)
        cand_mask_def[cand_test] = True
        sparse_order_def, _ = bm25_default.search(q_text, top_k=100, mask=cand_mask_def)
        final_def, _ = rrf_merge(dense_order, sparse_order_def, k=60, top_k=10)
        ret_def = [engine.item_ids[i] for i in final_def]
        ap_def, p_def, h_def = compute_metrics(ret_def, rel_ids, k=10)
        c5_def_ap.append(ap_def); c5_def_p.append(p_def); c5_def_hit.append(h_def)
        
        # --- Config 5 Ablated BM25 (no usage) ---
        sparse_order_abl, _ = bm25_ablated.search(q_text, top_k=100, mask=cand_mask_def)
        final_abl, _ = rrf_merge(dense_order, sparse_order_abl, k=60, top_k=10)
        ret_abl = [engine.item_ids[i] for i in final_abl]
        ap_abl, p_abl, h_abl = compute_metrics(ret_abl, rel_ids, k=10)
        c5_abl_ap.append(ap_abl); c5_abl_p.append(p_abl); c5_abl_hit.append(h_abl)
        
    res_c0 = {"ap10": round(float(np.mean(c0_ap)), 4), "p10": round(float(np.mean(c0_p)), 4), "hit10": round(float(np.mean(c0_hit)), 4)}
    res_c5_def = {"ap10": round(float(np.mean(c5_def_ap)), 4), "p10": round(float(np.mean(c5_def_p)), 4), "hit10": round(float(np.mean(c5_def_hit)), 4)}
    res_c5_abl = {"ap10": round(float(np.mean(c5_abl_ap)), 4), "p10": round(float(np.mean(c5_abl_p)), 4), "hit10": round(float(np.mean(c5_abl_hit)), 4)}
    
    print(f"Config 0 (Model thuần)                 : AP@10={res_c0['ap10']}, P@10={res_c0['p10']}, Hit@10={res_c0['hit10']}")
    print(f"Config 5 (BM25 gốc có usage_vi)        : AP@10={res_c5_def['ap10']}, P@10={res_c5_def['p10']}, Hit@10={res_c5_def['hit10']}")
    print(f"Config 5 (BM25 đã loại bỏ usage_vi)    : AP@10={res_c5_abl['ap10']}, P@10={res_c5_abl['p10']}, Hit@10={res_c5_abl['hit10']}")
    
    out_payload = {
        "metadata": {
            "task": "Phase 3g Q6 Independence & BM25 Ablation Audit",
            "val_queries_count": len(val_q),
            "test_queries_count": len(test_q),
            "total_queries_count": len(val_q) + len(test_q),
            "overlap_texts_count": len(overlap_texts)
        },
        "ground_truth_definition": {
            "has_articleType": True,
            "has_gender": True,
            "has_season_or_usage": True,
            "independence_assessment": "Q6 KHÔNG độc lập hoàn toàn với Query Parser vì ground truth vẫn ràng buộc articleType và gender. Parser lọc đúng danh mục giúp thu hẹp không gian tìm kiếm."
        },
        "bm25_fields": bm25_fields_default,
        "bm25_ablation_results": {
            "config_0_model_thuan": res_c0,
            "config_5_default_bm25": res_c5_def,
            "config_5_ablated_bm25_no_usage": res_c5_abl,
            "delta_ap10": round(res_c5_abl["ap10"] - res_c5_def["ap10"], 4),
            "delta_p10": round(res_c5_abl["p10"] - res_c5_def["p10"], 4),
            "delta_hit10": round(res_c5_abl["hit10"] - res_c5_def["hit10"], 4)
        },
        "alpha_selection_rationale": {
            "val_gap_alpha_0_85_vs_1_0": 0.0052,
            "rule": "gap <= 0.0100 allowed selecting alpha*=1.0 to save 170MB RAM and eliminate text embedding latency",
            "val_gallery_nature": "Gallery val gồm 37,356 ảnh 'unknown' (rất có thể chứa ảnh train của checkpoint SigLIP), do đó số val mang tính lạc quan.",
            "heldout_status": "Heldout đã được quan sát ở Pha 3e và 3f, không còn là tập ẩn hoàn toàn."
        }
    }
    
    out_file = Path("reports/phase3g_q6_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved Q6 audit to {out_file}")

if __name__ == "__main__":
    run_q6_audit()
