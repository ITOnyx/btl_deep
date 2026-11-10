"""
Full CIR Diagnostics script for GĐ A.1
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding='utf-8')
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from app.engine import SearchService
from src.query_parser_v2 import QueryParserV2
from src.vi_vocab import COLOUR_MAP

def format_item(item: Dict[str, Any]) -> str:
    return f"[{item.get('item_id')}] {item.get('name')} | Type: {item.get('articleType')} | Col: {item.get('baseColour')} | Gen: {item.get('gender')} | Score: {item.get('score', 0.0):.4f}"

def run_diagnostics():
    print("=" * 70)
    print("BẮT ĐẦU CHẨN ĐOÁN CHI TIẾT CIR (GIAI ĐOẠN A.1)")
    print("=" * 70)
    
    service = SearchService("config.yaml")
    engine = service.engine
    parser = QueryParserV2()
    
    shirt_crop_path = ROOT_DIR / "eval" / "cir_test" / "crop_uploaded_shirt.jpg"
    cat_15970_path = ROOT_DIR / "data" / "fashion-dataset" / "images" / "15970.jpg"
    
    # -------------------------------------------------------------
    # 2a. CHỈ ẢNH (Image Only / I2I)
    # -------------------------------------------------------------
    print("\n--- 2a. CHỈ ẢNH (Visual Search I2I thuần, không text) ---")
    i2i_crop, _ = service.search_image(shirt_crop_path, top_k=10, mode="fast")
    print(f"\n[2a.1] Ảnh tải lên (crop_uploaded_shirt.jpg):")
    for r in i2i_crop[:8]:
        print(f"  #{r['rank']}: {format_item(r)}")
        
    i2i_15970, _ = service.search_image(cat_15970_path, top_k=10, mode="fast")
    print(f"\n[2a.2] Ảnh catalog ID 15970:")
    for r in i2i_15970[:8]:
        print(f"  #{r['rank']}: {format_item(r)}")

    # -------------------------------------------------------------
    # 2b. CHỈ TEXT (Text Only / T2I)
    # -------------------------------------------------------------
    print("\n--- 2b. CHỈ TEXT (Text2Img thuần: C0 Model thuần vs C5 Hệ thống) ---")
    queries_text = ["màu xanh navy", "navy blue", "áo sơ mi xanh navy"]
    for q in queries_text:
        print(f"\n[2b] Query: '{q}':")
        res_c0, _ = service.search_text(q, config_mode="c0", top_k=5)
        print(f"  Model thuần C0 (SigLIP Dense):")
        for r in res_c0[:5]:
            print(f"    #{r['rank']}: {format_item(r)}")
        res_c5, _ = service.search_text(q, config_mode="c5", top_k=5)
        print(f"  Hệ thống C5 (Parser + Soft Bonus):")
        for r in res_c5[:5]:
            print(f"    #{r['rank']}: {format_item(r)}")

    # -------------------------------------------------------------
    # 2c. CIR: Text tiếng Việt vs Text tiếng Anh
    # -------------------------------------------------------------
    print("\n--- 2c. CIR: Text tiếng Việt vs Text tiếng Anh (crop_uploaded_shirt.jpg) ---")
    for use_sb in [True, False]:
        print(f"\n>> Chế độ use_soft_bonus = {use_sb}:")
        cir_vi, _ = service.search_cir(shirt_crop_path, "màu xanh navy", top_k=8, use_soft_bonus=use_sb, force_upload=True)
        print(f"  CIR + 'màu xanh navy':")
        for r in cir_vi[:5]:
            print(f"    #{r['rank']}: {format_item(r)}")
        cir_en, _ = service.search_cir(shirt_crop_path, "navy blue", top_k=8, use_soft_bonus=use_sb, force_upload=True)
        print(f"  CIR + 'navy blue':")
        for r in cir_en[:5]:
            print(f"    #{r['rank']}: {format_item(r)}")

    # -------------------------------------------------------------
    # 2d. ĐỘ PHÂN GIẢI & NGUỒN ẢNH: crop vs catalog ID 15970
    # -------------------------------------------------------------
    print("\n--- 2d. NGUỒN ẢNH & ĐỘ PHÂN GIẢI: crop vs catalog ID 15970 (cir_upload vs cir_catalog) ---")
    cir_15970_upload, _ = service.search_cir(cat_15970_path, "màu xanh navy", top_k=8, use_soft_bonus=True, force_upload=True)
    print(f"\n[2d.1] ID 15970 qua luồng cir_upload (force_upload=True, ViT tính lại):")
    for r in cir_15970_upload[:5]:
        print(f"  #{r['rank']}: {format_item(r)}")

    cir_15970_cached, _ = service.search_cir(cat_15970_path, "màu xanh navy", top_k=8, use_soft_bonus=True, ref_id="15970", force_upload=False)
    print(f"\n[2d.2] ID 15970 qua luồng cir_catalog (dùng GPU cache emb_img_raw):")
    for r in cir_15970_cached[:5]:
        print(f"  #{r['rank']}: {format_item(r)}")

    # -------------------------------------------------------------
    # 2e. QUÉT TRỌNG SỐ KẾT HỢP ẢNH / TEXT (ALPHA SWEEP)
    # -------------------------------------------------------------
    print("\n--- 2e. QUÉT TRỌNG SỐ KẾT HỢP LINEAR FUSION (Alpha Sweep) ---")
    # Trực tiếp tính embedding ảnh và text, kết hợp tuyến tính q = norm((1-alpha)*v_img + alpha*v_txt)
    im_pil = Image.open(shirt_crop_path).convert("RGB")
    t_im = engine.img_transform(im_pil).unsqueeze(0).to(engine.device)
    if engine.device == "cuda":
        t_im = t_im.half()
    with torch.no_grad():
        v_img_t = engine.model.encode_image(t_im)
        v_img = v_img_t.cpu().numpy()[0]
        v_img = v_img / np.linalg.norm(v_img)

    tok = engine.processor(text=["màu xanh navy"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    with torch.no_grad():
        v_txt_t = engine.model.encode_text(tok["input_ids"].to(engine.device))
        v_txt = v_txt_t.cpu().numpy()[0]
        v_txt = v_txt / np.linalg.norm(v_txt)

    # CIR combiner embedding
    with torch.no_grad():
        cir_feat = engine.model.compose_query(t_im, tok["input_ids"].to(engine.device))
        cir_comb_vec = cir_feat.cpu().numpy()[0]
        cir_comb_vec = cir_comb_vec / np.linalg.norm(cir_comb_vec)

    # Kiểm tra cổng gate và attention của CIR Combiner
    with torch.no_grad():
        v_img_inp = v_img_t if v_img_t.dtype == next(engine.model.cir_combiner.parameters()).dtype else v_img_t.to(next(engine.model.cir_combiner.parameters()).dtype)
        v_txt_inp = v_txt_t if v_txt_t.dtype == next(engine.model.cir_combiner.parameters()).dtype else v_txt_t.to(next(engine.model.cir_combiner.parameters()).dtype)
        p_img = engine.model.cir_combiner.image_proj(v_img_inp)
        p_txt = engine.model.cir_combiner.text_proj(v_txt_inp)
        concat_gate = torch.cat([p_img, p_txt], dim=-1)
        gate_alpha = engine.model.cir_combiner.gate(concat_gate).item()
        print(f"\n[CIR Combiner Internal State]")
        print(f"  Gate alpha (trọng số text trong gate): {gate_alpha:.4f}")
        sim_comb_img = float(np.dot(cir_comb_vec, v_img))
        sim_comb_txt = float(np.dot(cir_comb_vec, v_txt))
        sim_img_txt = float(np.dot(v_img, v_txt))
        print(f"  Cosine(v_img, v_txt): {sim_img_txt:.4f}")
        print(f"  Cosine(cir_comb, v_img): {sim_comb_img:.4f}")
        print(f"  Cosine(cir_comb, v_txt): {sim_comb_txt:.4f}")

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for a in alphas:
        q_blend = (1.0 - a) * v_img + a * v_txt
        q_blend = q_blend / np.linalg.norm(q_blend)
        q_c = engine.scorer.center_vectors(q_blend[np.newaxis, :])[0]
        sims = np.dot(engine.emb_img_c, q_c)
        top5_idx = np.argsort(-sims)[:5]
        print(f"\n  [Alpha = {a:.2f}] (1-{a:.2f})*Img + {a:.2f}*Txt:")
        for rank, idx in enumerate(top5_idx, 1):
            row = engine.df.iloc[idx]
            print(f"    #{rank}: [{row['item_id']}] {row['name']} | Type: {row['articleType']} | Col: {row['baseColour']} | Sim: {sims[idx]:.4f}")

    # -------------------------------------------------------------
    # 3. XÁC NHẬN ÁNH XẠ MÀU TIẾNG VIỆT
    # -------------------------------------------------------------
    print("\n--- 3. XÁC NHẬN ÁNH XẠ MÀU TIẾNG VIỆT ---")
    test_colors = ["xanh navy", "màu xanh navy", "xanh dương", "màu xanh dương", "xanh đen", "màu xanh đen", "xanh biển", "màu xanh biển"]
    for c in test_colors:
        parsed = parser.parse(c)
        print(f"  '{c}' -> baseColour: {parsed.get('baseColour')}, colour_family: {parsed.get('colour_family')}")

    # -------------------------------------------------------------
    # 4. KIỂM TRA XUNG ĐỘT NGUỒN ẢNH TRÊN UI
    # -------------------------------------------------------------
    print("\n--- 4. KIỂM TRA NGUỒN ẢNH (app/main.py logic) ---")
    print("  Logic trong app/main.py:")
    print("    if cir_upload is not None: dùng cir_upload, cir_ref_id=None, force_upload=True")
    print("    elif cir_sample is not None: dùng cir_sample, cir_ref_id=stem")
    print("  => KẾT QUẢ: cir_upload ĐƯỢC ƯU TIÊN TUYỆT ĐỐI.")
    print("  => VẤN ĐỀ GIAO DIỆN: Selectbox 'Hoặc chọn ảnh có sẵn' vẫn hiện ID 15970, KHÔNG có thông báo mờ/vô hiệu hóa, gây hiểu lầm là đang dùng ID 15970.")

if __name__ == "__main__":
    run_diagnostics()
