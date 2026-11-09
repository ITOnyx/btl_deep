"""
Attribute Index for Fast Masked & Exact Subspace Retrieval (src/attr_index.py)
Phase 3e: Stores contiguous NumPy attribute arrays for all 38,646 in-scope catalog items.
Provides:
1. Exact vector search over boolean attribute masks (without HNSW approximation).
2. Fast mask construction (articleType, gender, baseColour, colour_family, brand, split).
3. Relaxed fallback mask construction when candidate sets are too small.
4. Attribute match bonus computation (soft scoring).
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

import sys
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from src.vi_vocab import remove_diacritics
except ImportError:
    from vi_vocab import remove_diacritics

class AttributeIndex:
    def __init__(
        self,
        catalog_path: str = "data/catalog/catalog.parquet",
        brands_path: str = "src/brands.json",
        families_path: str = "eval/families.json"
    ):
        # 1. Load catalog in_scope
        df = pd.read_parquet(catalog_path)
        self.df = df[df["in_scope"] == True].reset_index(drop=True)
        self.N = len(self.df)
        self.item_ids = self.df["item_id"].tolist()
        self.id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

        # 2. Load families
        fam_p = Path(families_path)
        self.color_to_family = {}
        if fam_p.exists():
            with open(fam_p, "r", encoding="utf-8") as f:
                fam_data = json.load(f)
                for fam_name, color_list in fam_data.get("colour_families", {}).items():
                    for col in color_list:
                        self.color_to_family[col.lower()] = fam_name

        # 3. Load brands
        b_p = Path(brands_path)
        self.known_brands = {}
        if b_p.exists():
            with open(b_p, "r", encoding="utf-8") as f:
                raw_b = json.load(f)
                if isinstance(raw_b, dict) and "canonical_brands" in raw_b:
                    self.known_brands = raw_b["canonical_brands"]
                else:
                    self.known_brands = raw_b

        # 4. Extract product brands
        self._extract_product_brands()

        # 5. Build ID mappings and integer arrays
        self._build_index_arrays()

    def _extract_product_brands(self):
        """Extracts brand for each catalog item matching known brands (cached for speed)."""
        cache_p = Path("data/catalog/product_brands.json")
        if cache_p.exists():
            with open(cache_p, "r", encoding="utf-8") as f:
                extracted_brands = json.load(f)
            if len(extracted_brands) == self.N:
                self.df["brand"] = extracted_brands
                return

        product_names = self.df["name"].fillna("").astype(str).tolist()
        extracted_brands = []

        # Sort known brands by length descending (longest match first)
        sorted_brands = sorted(self.known_brands.keys(), key=lambda b: len(b.split()), reverse=True)
        compiled_brand_regexes = [
            (b, re.compile(rf"(?:\b|^){re.escape(remove_diacritics(b))}(?:\b|$)", re.IGNORECASE))
            for b in sorted_brands
        ]

        for name in product_names:
            clean_name = remove_diacritics(name)
            matched = "Unknown"
            for b_name, pat in compiled_brand_regexes:
                if pat.search(clean_name):
                    matched = b_name
                    break
            extracted_brands.append(matched)

        self.df["brand"] = extracted_brands
        try:
            with open(cache_p, "w", encoding="utf-8") as f:
                json.dump(extracted_brands, f, ensure_ascii=False)
        except Exception:
            pass

    def _build_index_arrays(self):
        """Maps attributes to integer arrays for ultra-fast vectorized masking."""
        # Article Types
        unique_types = sorted(self.df["articleType"].fillna("Unknown").unique())
        self.art2id = {t: i for i, t in enumerate(unique_types)}
        self.id2art = {i: t for t, i in self.art2id.items()}
        self.article_type_ids = np.array([self.art2id.get(t, -1) for t in self.df["articleType"].fillna("Unknown")], dtype=np.int32)

        # Genders
        unique_genders = sorted(self.df["gender"].fillna("Unknown").unique())
        self.gen2id = {g: i for i, g in enumerate(unique_genders)}
        self.id2gen = {i: g for g, i in self.gen2id.items()}
        self.gender_ids = np.array([self.gen2id.get(g, -1) for g in self.df["gender"].fillna("Unknown")], dtype=np.int32)

        # Base Colours
        unique_colours = sorted(self.df["baseColour"].fillna("Unknown").unique())
        self.col2id = {c: i for i, c in enumerate(unique_colours)}
        self.id2col = {i: c for c, i in self.col2id.items()}
        self.colour_ids = np.array([self.col2id.get(c, -1) for c in self.df["baseColour"].fillna("Unknown")], dtype=np.int32)

        # Colour Families
        product_families = [self.color_to_family.get(str(c).lower(), str(c)) for c in self.df["baseColour"].fillna("Unknown")]
        unique_families = sorted(list(set(product_families)))
        self.fam2id = {f: i for i, f in enumerate(unique_families)}
        self.id2fam = {i: f for f, i in self.fam2id.items()}
        self.family_ids = np.array([self.fam2id.get(f, -1) for f in product_families], dtype=np.int32)

        # Brands
        unique_brands = sorted(self.df["brand"].unique())
        self.brand2id = {b: i for i, b in enumerate(unique_brands)}
        self.id2brand = {i: b for b, i in self.brand2id.items()}
        self.brand_ids = np.array([self.brand2id.get(b, -1) for b in self.df["brand"]], dtype=np.int32)

        # Split
        self.splits = self.df["split"].fillna("unknown").to_numpy()

    def get_mask(
        self,
        article_type: Optional[str] = None,
        gender: Optional[str] = None,
        colour: Optional[str] = None,
        colour_family: Optional[str] = None,
        brand: Optional[str] = None,
        split: Optional[str] = None
    ) -> np.ndarray:
        """Constructs an exact boolean mask based on provided attribute criteria."""
        mask = np.ones(self.N, dtype=bool)

        if split is not None:
            mask &= (self.splits == split)

        if article_type is not None and article_type in self.art2id:
            mask &= (self.article_type_ids == self.art2id[article_type])

        if gender is not None and gender in self.gen2id:
            # Handle unisex compatibility if requested
            if gender in ("Men", "Women", "Boys", "Girls"):
                target_ids = [self.gen2id[gender]]
                if "Unisex" in self.gen2id:
                    target_ids.append(self.gen2id["Unisex"])
                mask &= np.isin(self.gender_ids, target_ids)
            else:
                mask &= (self.gender_ids == self.gen2id[gender])

        if colour is not None and colour in self.col2id:
            mask &= (self.colour_ids == self.col2id[colour])
        elif colour_family is not None and colour_family in self.fam2id:
            mask &= (self.family_ids == self.fam2id[colour_family])

        if brand is not None and brand in self.brand2id:
            mask &= (self.brand_ids == self.brand2id[brand])

        return mask

    def get_relaxed_mask(
        self,
        parsed_spec: Dict[str, Any],
        split: Optional[str] = None,
        min_candidates: int = 10
    ) -> Tuple[np.ndarray, str]:
        """
        Progressively relaxes attributes if candidate count is below min_candidates:
        Level 0: type + gender + colour + brand
        Level 1: type + gender + colour
        Level 2: type + gender + colour_family
        Level 3: type + gender
        Level 4: type only
        Level 5: all gallery (fallback)
        """
        art = parsed_spec.get("articleType")
        gen = parsed_spec.get("gender")
        col = parsed_spec.get("baseColour")
        fam = parsed_spec.get("colour_family")
        brd = parsed_spec.get("brand")

        # Level 0 (if brand present)
        if brd:
            mask0 = self.get_mask(article_type=art, gender=gen, colour=col, brand=brd, split=split)
            if np.count_nonzero(mask0) >= min_candidates:
                return mask0, "L0_exact_with_brand"

        # Level 1: type + gender + colour
        if art and gen and col:
            mask1 = self.get_mask(article_type=art, gender=gen, colour=col, split=split)
            if np.count_nonzero(mask1) >= min_candidates:
                return mask1, "L1_type_gender_colour"

        # Level 2: type + gender + colour_family
        if art and gen and fam:
            mask2 = self.get_mask(article_type=art, gender=gen, colour_family=fam, split=split)
            if np.count_nonzero(mask2) >= min_candidates:
                return mask2, "L2_type_gender_family"

        # Level 3: type + gender
        if art and gen:
            mask3 = self.get_mask(article_type=art, gender=gen, split=split)
            if np.count_nonzero(mask3) >= min_candidates:
                return mask3, "L3_type_gender"

        # Level 4: type only
        if art:
            mask4 = self.get_mask(article_type=art, split=split)
            if np.count_nonzero(mask4) >= min_candidates:
                return mask4, "L4_type_only"

        # Level 5: brand only (if available)
        if brd:
            mask5 = self.get_mask(brand=brd, split=split)
            if np.count_nonzero(mask5) >= min_candidates:
                return mask5, "L5_brand_only"

        # Fallback: all split
        fallback_mask = np.ones(self.N, dtype=bool) if split is None else (self.splits == split)
        return fallback_mask, "L6_fallback_all"

    def filter_exact(self, parsed_spec: Dict[str, Any], mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns candidate indices strictly matching all non-null attributes in parsed_spec."""
        m = self.get_mask(
            article_type=parsed_spec.get("articleType"),
            gender=parsed_spec.get("gender"),
            colour=parsed_spec.get("baseColour"),
            brand=parsed_spec.get("brand")
        )
        if mask is not None:
            m = m & mask
        return np.where(m)[0]

    def filter_relaxed(
        self,
        parsed_spec: Dict[str, Any],
        min_candidates: int = 10,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, str]:
        """Returns candidate indices with multi-level attribute relaxation."""
        art = parsed_spec.get("articleType")
        gen = parsed_spec.get("gender")
        col = parsed_spec.get("baseColour")
        fam = parsed_spec.get("colour_family")
        brd = parsed_spec.get("brand")

        base_mask = np.ones(self.N, dtype=bool) if mask is None else mask.copy()

        # Level 0: all attributes + brand
        if brd:
            m0 = base_mask & self.get_mask(article_type=art, gender=gen, colour=col, brand=brd)
            if np.count_nonzero(m0) >= min_candidates:
                return np.where(m0)[0], "L0_exact_with_brand"

        # Level 1: type + gender + colour
        if art and gen and col:
            m1 = base_mask & self.get_mask(article_type=art, gender=gen, colour=col)
            if np.count_nonzero(m1) >= min_candidates:
                return np.where(m1)[0], "L1_type_gender_colour"

        # Level 2: type + gender + colour_family
        if art and gen and fam:
            m2 = base_mask & self.get_mask(article_type=art, gender=gen, colour_family=fam)
            if np.count_nonzero(m2) >= min_candidates:
                return np.where(m2)[0], "L2_type_gender_family"

        # Level 3: type + gender
        if art and gen:
            m3 = base_mask & self.get_mask(article_type=art, gender=gen)
            if np.count_nonzero(m3) >= min_candidates:
                return np.where(m3)[0], "L3_type_gender"

        # Level 4: type only
        if art:
            m4 = base_mask & self.get_mask(article_type=art)
            if np.count_nonzero(m4) >= min_candidates:
                return np.where(m4)[0], "L4_type_only"

        # Level 5: brand only (if available)
        if brd:
            m5 = base_mask & self.get_mask(brand=brd)
            if np.count_nonzero(m5) >= min_candidates:
                return np.where(m5)[0], "L5_brand_only"

        # Level 6: fallback to all candidates in base_mask
        return np.where(base_mask)[0], "L6_fallback_all"

    def search_masked(
        self,
        query_vec: np.ndarray,
        gallery_embs: np.ndarray,
        mask: Optional[np.ndarray] = None,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact subset search using boolean mask (avoids HNSW graph traversal on masked subsets).
        query_vec: (D,) or (1, D) normalized embedding
        gallery_embs: (N, D) normalized embeddings
        mask: (N,) bool array or None
        returns: (topk_indices, topk_scores)
        """
        q = query_vec.reshape(1, -1)

        if mask is None:
            cand_indices = np.arange(self.N)
            cand_embs = gallery_embs
        else:
            cand_indices = np.where(mask)[0]
            if len(cand_indices) == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            cand_embs = gallery_embs[cand_indices]

        # Exact dot product
        scores = np.dot(cand_embs, q.T).flatten()

        if len(scores) <= top_k:
            order = np.argsort(-scores)
            return cand_indices[order], scores[order]
        else:
            topk_sub = np.argpartition(-scores, top_k)[:top_k]
            topk_sub = topk_sub[np.argsort(-scores[topk_sub])]
            return cand_indices[topk_sub], scores[topk_sub]

    def compute_attribute_soft_bonus(
        self,
        parsed_spec: Dict[str, Any],
        candidate_indices: np.ndarray,
        gamma_type: float = 0.15,
        gamma_gender: float = 0.10,
        gamma_colour: float = 0.10,
        gamma_brand: float = 0.15
    ) -> np.ndarray:
        """
        Computes additive soft scoring bonus for candidate items based on parsed spec.
        """
        bonus = np.zeros(len(candidate_indices), dtype=np.float32)
        if len(candidate_indices) == 0:
            return bonus

        art = parsed_spec.get("articleType")
        gen = parsed_spec.get("gender")
        col = parsed_spec.get("baseColour")
        fam = parsed_spec.get("colour_family")
        brd = parsed_spec.get("brand")

        if art and art in self.art2id:
            cand_arts = self.article_type_ids[candidate_indices]
            bonus += np.where(cand_arts == self.art2id[art], gamma_type, 0.0).astype(np.float32)

        if gen and gen in self.gen2id:
            cand_gens = self.gender_ids[candidate_indices]
            target_ids = [self.gen2id[gen]]
            if "Unisex" in self.gen2id:
                target_ids.append(self.gen2id["Unisex"])
            bonus += np.where(np.isin(cand_gens, target_ids), gamma_gender, 0.0).astype(np.float32)

        if col and col in self.col2id:
            cand_cols = self.colour_ids[candidate_indices]
            match_col = (cand_cols == self.col2id[col])
            bonus += np.where(match_col, gamma_colour, 0.0).astype(np.float32)
            # Half bonus for matching colour family if not exact color
            if fam and fam in self.fam2id:
                cand_fams = self.family_ids[candidate_indices]
                match_fam = (cand_fams == self.fam2id[fam]) & (~match_col)
                bonus += np.where(match_fam, gamma_colour * 0.5, 0.0).astype(np.float32)
        elif fam and fam in self.fam2id:
            cand_fams = self.family_ids[candidate_indices]
            bonus += np.where(cand_fams == self.fam2id[fam], gamma_colour * 0.5, 0.0).astype(np.float32)

        if brd and brd in self.brand2id:
            cand_brds = self.brand_ids[candidate_indices]
            bonus += np.where(cand_brds == self.brand2id[brd], gamma_brand, 0.0).astype(np.float32)

        return bonus

if __name__ == "__main__":
    attr_idx = AttributeIndex()
    print(f"AttributeIndex loaded: {attr_idx.N} items")
    print(f"Sample unique brands: {len(attr_idx.brand2id)}")
    
    # Test mask
    mask = attr_idx.get_mask(article_type="Tshirts", gender="Men", colour="Black")
    print(f"Count for (Tshirts, Men, Black): {np.count_nonzero(mask)}")
    
    # Test relaxed mask
    parsed_sample = {"articleType": "Tshirts", "gender": "Men", "baseColour": "Black", "brand": "Puma"}
    r_mask, level = attr_idx.get_relaxed_mask(parsed_sample, min_candidates=10)
    print(f"Relaxed mask count: {np.count_nonzero(r_mask)} at level '{level}'")
