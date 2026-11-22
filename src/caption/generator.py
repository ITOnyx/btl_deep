"""VLM Caption Generator for Configuration A, B, and C.
Supports InternVL2-2B (Core) and Vintern-1B-v3_5 (Auxiliary) in 8-bit mode.
Strictly follows Law 4, Law 5, Law D.1-D.8.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
import gc
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Sanitize environment SSL certificates
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("SSL_CERT_FILE", None)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from src.caption.prompts import (
    PROMPT_VERSION,
    INTERNVL2_PROMPT_TEMPLATE,
    VINTERN_PROMPT_TEMPLATE,
)
from src.caption.merger import (
    MERGE_VERSION,
    merge_caption_v1,
    clean_introductory_phrases,
    get_siglip_tokenizer,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size):
    best_ratio_fit = tuple()
    best_ratio_area = float("inf")
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        area = ratio[0] * ratio[1]
        if abs(target_aspect_ratio - aspect_ratio) < best_ratio_area:
            best_ratio_area = abs(target_aspect_ratio - aspect_ratio)
            best_ratio_fit = ratio
        elif abs(target_aspect_ratio - aspect_ratio) == best_ratio_area:
            if area > best_ratio_area:
                best_ratio_fit = ratio
    return best_ratio_fit


def dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=2):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(1, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if 1 <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_tensor(image_path: str, input_size=448, max_num=1):
    img = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(img, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(patch) for patch in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class VLMCaptionPipeline:
    def __init__(self, seed: int = 42, quantization: str = "8bit"):
        self.seed = seed
        self.quantization = quantization
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model_8bit(self, model_id: str):
        print(f"[{time.strftime('%X')}] Loading {model_id} in 8-bit...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModel.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda:0",
            trust_remote_code=True,
        ).eval()

        peak_b = torch.cuda.max_memory_allocated(0)
        print(f"[{time.strftime('%X')}] Loaded {model_id}! Peak VRAM: {peak_b / 1024**3:.2f} GB")
        return model, tokenizer

    def unload_model(self, model, tokenizer):
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{time.strftime('%X')}] Unloaded model. VRAM freed.")

    def run_core_generation(
        self,
        samples: List[Dict[str, Any]],
        model_id: str = "OpenGVLab/InternVL2-2B",
        revision: str = "e4f6747bd20f139e637642c6a058c6bd00b36919",
        cache_path: str = "eval/cache_internvl2_pilot.json",
    ) -> Dict[str, Dict[str, Any]]:
        cached = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"Loaded {len(cached)} existing InternVL2 captions from cache.")

        remaining = [s for s in samples if str(s["product_id"]) not in cached]
        if not remaining:
            print("All InternVL2 captions already cached.")
            return cached

        model, tokenizer = self.load_model_8bit(model_id)
        gen_config = dict(max_new_tokens=48, do_sample=False, num_beams=1)

        t_start = time.time()
        for idx, sample in enumerate(remaining):
            pid = str(sample["product_id"])
            img_p = sample["image_path"]
            
            t0 = time.time()
            try:
                pixel_values = load_image_tensor(img_p, max_num=1).to(torch.float16).cuda()
                with torch.no_grad():
                    response, _ = model.chat(
                        tokenizer,
                        pixel_values,
                        INTERNVL2_PROMPT_TEMPLATE,
                        gen_config,
                        history=None,
                        return_history=True,
                    )
                latency = time.time() - t0
                cleaned = clean_introductory_phrases(response)
                cached[pid] = {
                    "raw_caption": response.strip(),
                    "cleaned_caption": cleaned,
                    "latency_sec": round(latency, 3),
                    "model_id": model_id,
                    "revision": revision,
                    "seed": self.seed,
                }
            except Exception as e:
                print(f"Error on {pid} ({img_p}): {e}")
                cached[pid] = {
                    "raw_caption": "",
                    "cleaned_caption": "",
                    "latency_sec": round(time.time() - t0, 3),
                    "error": str(e),
                    "model_id": model_id,
                    "revision": revision,
                    "seed": self.seed,
                }

            if (idx + 1) % 25 == 0 or (idx + 1) == len(remaining):
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cached, f, ensure_ascii=False, indent=2)
                print(f"[{idx+1}/{len(remaining)}] Saved cache. Avg speed: {(time.time() - t_start)/(idx+1):.2f}s/img")

        self.unload_model(model, tokenizer)
        return cached

    def run_aux_generation(
        self,
        samples: List[Dict[str, Any]],
        model_id: str = "5CD-AI/Vintern-1B-v3_5",
        revision: str = "b98f263eab246eb5269ade64edbdca8a887dc44d",
        cache_path: str = "eval/cache_vintern_pilot.json",
    ) -> Dict[str, Dict[str, Any]]:
        cached = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"Loaded {len(cached)} existing Vintern captions from cache.")

        remaining = [s for s in samples if str(s["product_id"]) not in cached]
        if not remaining:
            print("All Vintern captions already cached.")
            return cached

        model, tokenizer = self.load_model_8bit(model_id)
        gen_config = dict(max_new_tokens=48, do_sample=False, num_beams=1)

        t_start = time.time()
        for idx, sample in enumerate(remaining):
            pid = str(sample["product_id"])
            img_p = sample["image_path"]
            
            t0 = time.time()
            try:
                pixel_values = load_image_tensor(img_p, max_num=1).to(torch.float16).cuda()
                with torch.no_grad():
                    response, _ = model.chat(
                        tokenizer,
                        pixel_values,
                        VINTERN_PROMPT_TEMPLATE,
                        gen_config,
                        history=None,
                        return_history=True,
                    )
                latency = time.time() - t0
                cleaned = clean_introductory_phrases(response)
                cached[pid] = {
                    "raw_caption": response.strip(),
                    "cleaned_caption": cleaned,
                    "latency_sec": round(latency, 3),
                    "model_id": model_id,
                    "revision": revision,
                    "seed": self.seed,
                }
            except Exception as e:
                print(f"Error on {pid} ({img_p}): {e}")
                cached[pid] = {
                    "raw_caption": "",
                    "cleaned_caption": "",
                    "latency_sec": round(time.time() - t0, 3),
                    "error": str(e),
                    "model_id": model_id,
                    "revision": revision,
                    "seed": self.seed,
                }

            if (idx + 1) % 25 == 0 or (idx + 1) == len(remaining):
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cached, f, ensure_ascii=False, indent=2)
                print(f"[{idx+1}/{len(remaining)}] Saved cache. Avg speed: {(time.time() - t_start)/(idx+1):.2f}s/img")

        self.unload_model(model, tokenizer)
        return cached


def check_chinese_characters(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def check_english_contamination(text: str) -> bool:
    english_words = [
        r"\bmen\b", r"\bwomen\b", r"\bboys\b", r"\bgirls\b", r"\bshoes\b",
        r"\bshirt\b", r"\bt-shirt\b", r"\bjeans\b", r"\bcolor\b", r"\bcolour\b",
        r"\byellow\b", r"\bblue\b", r"\bblack\b", r"\bwhite\b", r"\bred\b",
        r"\bgreen\b", r"\bcasual\b", r"\bsports\b", r"\bfashion\b"
    ]
    for w in english_words:
        if re.search(w, text, re.IGNORECASE):
            return True
    return False


def check_metadata_match(text: str, metadata: dict) -> Dict[str, bool]:
    art_vi = str(metadata.get("articleType_vi", "")).lower()
    gender_vi = str(metadata.get("gender_vi", "")).lower()
    colour_vi = str(metadata.get("baseColour_vi", "")).lower()

    text_lower = text.lower()
    art_match = art_vi in text_lower if art_vi else True
    gender_match = gender_vi in text_lower if gender_vi else True
    colour_match = colour_vi in text_lower if colour_vi else True

    return {
        "article_match": art_match,
        "gender_match": gender_match,
        "colour_match": colour_match,
        "all_match": (art_match and gender_match and colour_match),
    }
