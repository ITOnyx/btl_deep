"""
Query Builder for E-Commerce Visual Search (Phase 3c / Phase 4).
Supports:
1. Rule c': Diacritic restoration applied ONLY when text is unaccented.
2. Method A: Raw free-text (with rule c' for unaccented).
3. Method B: Shallow parsing heuristics -> Structured pseudo-training query:
   f"[{category}] {query} ({attrs})"
4. Method C: Ensemble embedding (average of raw and structured embeddings, normalized).
"""
import re
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, Union, List

from src.vi_vocab import remove_diacritics
from src.vi_restore import DiacriticRestorer
from src.query_parser import parse_query

# Singleton restorer
_RESTORER = None

def get_restorer() -> DiacriticRestorer:
    global _RESTORER
    if _RESTORER is None:
        _RESTORER = DiacriticRestorer()
    return _RESTORER

# Mapping from articleType to hierarchical category string
ARTICLE_TO_CATEGORY = {
    # Apparel
    "Shirts": "Apparel > Topwear > Shirts",
    "Tshirts": "Apparel > Topwear > Tshirts",
    "Tops": "Apparel > Topwear > Tops",
    "Kurtas": "Apparel > Topwear > Kurtas",
    "Sweaters": "Apparel > Topwear > Sweaters",
    "Sweatshirts": "Apparel > Topwear > Sweatshirts",
    "Jackets": "Apparel > Topwear > Jackets",
    "Jeans": "Apparel > Bottomwear > Jeans",
    "Trousers": "Apparel > Bottomwear > Trousers",
    "Shorts": "Apparel > Bottomwear > Shorts",
    "Track Pants": "Apparel > Bottomwear > Track Pants",
    "Skirts": "Apparel > Bottomwear > Skirts",
    "Leggings": "Apparel > Bottomwear > Leggings",
    "Dresses": "Apparel > Dress > Dresses",
    "Innerwear Vests": "Apparel > Innerwear > Innerwear Vests",
    "Briefs": "Apparel > Innerwear > Briefs",
    "Bra": "Apparel > Innerwear > Bra",
    "Socks": "Apparel > Innerwear > Socks",
    # Footwear
    "Casual Shoes": "Footwear > Shoes > Casual Shoes",
    "Sports Shoes": "Footwear > Shoes > Sports Shoes",
    "Formal Shoes": "Footwear > Shoes > Formal Shoes",
    "Flats": "Footwear > Shoes > Flats",
    "Heels": "Footwear > Shoes > Heels",
    "Sandals": "Footwear > Flip Flops > Sandals",
    "Flip Flops": "Footwear > Flip Flops > Flip Flops",
    # Accessories
    "Watches": "Accessories > Watches > Watches",
    "Backpacks": "Accessories > Bags > Backpacks",
    "Handbags": "Accessories > Bags > Handbags",
    "Wallets": "Accessories > Wallets > Wallets",
    "Belts": "Accessories > Belts > Belts",
    "Sunglasses": "Accessories > Eyewear > Sunglasses",
    "Ties": "Accessories > Ties > Ties",
    "Scarves": "Accessories > Scarves > Scarves",
    "Caps": "Accessories > Headwear > Caps",
}

GENDER_VI_MAP = {
    "Men": "nam",
    "Women": "nữ",
    "Boys": "bé trai",
    "Girls": "bé gái",
    "Unisex": "unisex nam nữ"
}

def is_unaccented(text: str) -> bool:
    """Checks if a string has no Vietnamese diacritics."""
    return text.lower() == remove_diacritics(text).lower()

def preprocess_user_query(text: str) -> str:
    """Applies Rule c': restores accents ONLY if query is unaccented."""
    text_clean = text.strip()
    if is_unaccented(text_clean):
        restorer = get_restorer()
        return restorer.restore(text_clean)
    return text_clean

def build_structured_query(raw_query: str) -> str:
    """
    Method B: Heuristically parses free-text into pseudo-training query:
    [{category}] {query} ({attrs})
    """
    clean_q = preprocess_user_query(raw_query)
    parsed = parse_query(clean_q)

    article_raw = parsed.get("articleType")
    article = article_raw[0] if isinstance(article_raw, list) and article_raw else article_raw
    if isinstance(article, list):
        article = article[0] if article else None

    gender_raw = parsed.get("gender")
    gender = gender_raw[0] if isinstance(gender_raw, list) and gender_raw else gender_raw

    colour_raw = parsed.get("baseColour")
    colour = colour_raw[0] if isinstance(colour_raw, list) and colour_raw else colour_raw

    category = ARTICLE_TO_CATEGORY.get(article, "") if article else ""

    attr_parts = []
    if gender:
        attr_parts.append(GENDER_VI_MAP.get(gender, gender.lower()))
    if colour:
        attr_parts.append(f"màu {colour.lower()}")
    if article:
        attr_parts.append(article)

    parts = []
    if category:
        parts.append(f"[{category}]")
    parts.append(clean_q)
    if attr_parts:
        attr_str = ", ".join(attr_parts)
        if attr_str.lower() not in clean_q.lower():
            parts.append(f"({attr_str})")

    return " ".join(parts)[:256]

def encode_user_query(
    raw_query: str,
    model: Any,
    processor: Any,
    method: str = "B"
) -> np.ndarray:
    """
    Encodes a user query using Method A, B, or C.
    method: 'A' (raw with rule c') | 'B' (structured pseudo-training) | 'C' (average of A and B)
    returns: L2-normalized numpy array of shape (1, 1152)
    """
    text_a = preprocess_user_query(raw_query)

    if method == "A":
        text_to_encode = text_a
        inp = processor(text=[text_to_encode], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            feat = model.encode_text(inp["input_ids"].to(model.device), None)
        vec = feat.cpu().numpy()
        return vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    elif method == "B":
        text_to_encode = build_structured_query(raw_query)
        inp = processor(text=[text_to_encode], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            feat = model.encode_text(inp["input_ids"].to(model.device), None)
        vec = feat.cpu().numpy()
        return vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    elif method == "C":
        text_b = build_structured_query(raw_query)
        inp_a = processor(text=[text_a], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        inp_b = processor(text=[text_b], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            feat_a = model.encode_text(inp_a["input_ids"].to(model.device), None).cpu().numpy()
            feat_b = model.encode_text(inp_b["input_ids"].to(model.device), None).cpu().numpy()
        vec = (feat_a + feat_b) / 2.0
        return vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    else:
        raise ValueError(f"Unknown method: {method}")
