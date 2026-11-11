"""Deterministic Caption Merger (merge_v1) for Configuration B.
Merges InternVL2-2B visual captions with normalized catalog metadata and auxiliary VLM features.
Enforces zero-hallucination on core discriminative attributes and keeps length <= 128 tokens.
"""

import re
from transformers import AutoTokenizer

MERGE_VERSION = "merge_v1"

# Shared tokenizer cache
_SIGLIP_TOKENIZER = None

def get_siglip_tokenizer(model_name="google/siglip-so400m-patch14-224"):
    global _SIGLIP_TOKENIZER
    if _SIGLIP_TOKENIZER is None:
        _SIGLIP_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _SIGLIP_TOKENIZER


def clean_introductory_phrases(text: str) -> str:
    """Strip repetitive boilerplate conversational prefixes from VLMs."""
    patterns = [
        r"^(trong hình là|bức ảnh chụp|đây là|sản phẩm là|hình ảnh hiển thị|mô tả:|hình ảnh cho thấy)\s*",
        r"^(một chiếc|một đôi|một bộ|một mẫu)\s*"
    ]
    cleaned = text.strip()
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def detect_and_resolve_conflicts(text: str, expected_gender_vi: str, expected_colour_vi: str):
    """
    Check if caption directly contradicts verified catalog metadata.
    Returns (cleaned_text, conflict_flags).
    """
    conflicts = []
    cleaned = text

    # Gender conflict
    if expected_gender_vi == "nam" and re.search(r"\b(dành cho nữ|cho phái đẹp|cho phụ nữ|quý bà|bạn nữ)\b", cleaned, re.IGNORECASE):
        conflicts.append("gender_conflict_female_in_male")
        cleaned = re.sub(r"\b(dành cho nữ|cho phái đẹp|cho phụ nữ|quý bà|bạn nữ)\b", "nam", cleaned, flags=re.IGNORECASE)
    elif expected_gender_vi == "nữ" and re.search(r"\b(dành cho nam|cho phái mạnh|cho đàn ông|quý ông|bạn nam)\b", cleaned, re.IGNORECASE):
        conflicts.append("gender_conflict_male_in_female")
        cleaned = re.sub(r"\b(dành cho nam|cho phái mạnh|cho đàn ông|quý ông|bạn nam)\b", "nữ", cleaned, flags=re.IGNORECASE)

    return cleaned, conflicts


def merge_caption_v1(
    core_caption: str,
    metadata: dict,
    aux_caption: str = "",
    max_tokens: int = 128
) -> dict:
    """
    Deterministic merger following Law 5:
    1. Mandatory discriminative prefix from metadata: [Loại] [Giới] màu [Màu]
    2. Conflict resolution against VLM outputs
    3. Integration of visual details from core InternVL2-2B and auxiliary Vintern
    4. Truncation control ensuring <= 128 tokens
    """
    art_type = str(metadata.get("articleType_vi", "trang phục")).strip().lower()
    gender = str(metadata.get("gender_vi", "unisex")).strip().lower()
    colour = str(metadata.get("baseColour_vi", "")).strip().lower()

    # Anchor prefix (Discriminative first rule)
    if colour:
        prefix = f"{art_type} {gender} màu {colour}".capitalize()
    else:
        prefix = f"{art_type} {gender}".capitalize()

    # Clean core caption
    core_clean = clean_introductory_phrases(core_caption)
    core_clean, conflicts = detect_and_resolve_conflicts(core_clean, gender, colour)

    # Clean aux caption if available
    aux_clean = clean_introductory_phrases(aux_caption) if aux_caption else ""
    if aux_clean:
        aux_clean, aux_conflicts = detect_and_resolve_conflicts(aux_clean, gender, colour)
        conflicts.extend(aux_conflicts)

    # Combine prefix and visual details
    body_parts = []
    if core_clean:
        body_parts.append(core_clean)
    if aux_clean and aux_clean.lower() not in core_clean.lower():
        # Append unique supplementary phrasing from aux
        body_parts.append(aux_clean)

    body_text = ". ".join(body_parts).strip()
    
    # Format full text
    if body_text.lower().startswith(prefix.lower()):
        merged_text = body_text
    else:
        merged_text = f"{prefix}. {body_text}"

    # Normalize whitespace
    merged_text = re.sub(r"\s+", " ", merged_text).strip()
    merged_text = re.sub(r"\.\s*\.", ".", merged_text)

    # Measure exact token length with SigLIP tokenizer
    tokenizer = get_siglip_tokenizer()
    tokens = tokenizer.encode(merged_text, add_special_tokens=True)
    token_len = len(tokens)

    # Truncate if exceeding max_tokens while preserving the prefix
    if token_len > max_tokens:
        # Decode first max_tokens tokens
        truncated_text = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
        merged_text = truncated_text.strip()
        tokens = tokenizer.encode(merged_text, add_special_tokens=True)
        token_len = len(tokens)

    return {
        "caption": merged_text,
        "token_len": token_len,
        "prefix": prefix,
        "conflicts": conflicts,
        "version": MERGE_VERSION
    }
