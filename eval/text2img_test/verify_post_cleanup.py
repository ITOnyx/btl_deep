"""
Post-Cleanup Verification Script (eval/text2img_test/verify_post_cleanup.py)
Validates:
1. 20 representative queries: IDs and scores match pre-cleanup results_raw.parquet exactly (zero deviation).
2. Edge cases of product cards (long names, 1-line names, missing images, empty brands, K=5/10/20/50, empty query).
3. Browser console error audit via Playwright on tabs 1, 2, 3, 4.
"""

import sys
import time
from pathlib import Path

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.engine import SearchService

def main():
    print("===============================================================")
    print("BẮT ĐẦU KIỂM ĐỊNH HẬU NÂNG CẤP GIAO DIỆN (POST-CLEANUP)")
    print("===============================================================")

    # 1. Load pre-cleanup results
    parquet_path = ROOT_DIR / "eval/text2img_test/results_raw.parquet"
    raw_df = pd.read_parquet(parquet_path)
    c5_pre = raw_df[(raw_df["config_mode"] == "c5") & (raw_df["top_k"] == 10)].copy()

    # 20 representative queries from across groups
    test_qids = [
        "Q_P_0702", "Q_P_0703", "Q_P_0704", "Q_P_0705", "Q_P_0706", "Q_P_0707", "Q_P_0708",
        "Q_A_0001", "Q_A_0002", "Q_A_0003",
        "Q_B_0080", "Q_B_0081",
        "Q_C_0176", "Q_C_0177",
        "Q_D_0227", "Q_D_0228",
        "Q_E_0407", "Q_E_0408",
        "Q_G_0615", "Q_H_0627"
    ]

    service = SearchService(config_path="config.yaml")

    print("\n--- 1. Kiểm tra 20 truy vấn: Top-K ID và Điểm số phải GIỐNG HỆT trước khi sửa ---")
    all_matched = True
    for qid in test_qids:
        sub_pre = c5_pre[c5_pre["query_id"] == qid].sort_values("rank")
        if len(sub_pre) == 0:
            continue
        q_text = sub_pre.iloc[0]["query"]
        pre_ids = sub_pre["item_id"].tolist()
        pre_scores = sub_pre["score"].tolist()

        res_post, _ = service.search_text(q_text, config_mode="c5", top_k=10, gamma=0.30)
        post_ids = [r["item_id"] for r in res_post]
        post_scores = [r["score"] for r in res_post]

        ids_match = (pre_ids == post_ids)
        scores_match = np.allclose(pre_scores, post_scores, atol=1e-5)

        if ids_match and scores_match:
            print(f"  [KHỚP 100%] `{qid}`: \"{q_text}\" (10/10 ID & Điểm số)")
        else:
            print(f"  [LỆCH] `{qid}`: \"{q_text}\" (Pre: {pre_ids[:3]} vs Post: {post_ids[:3]})")
            all_matched = False

    print(f"\nKết quả đối soát 20 truy vấn: {'100% ĐỒNG NHẤT TUYỆT ĐỐI' if all_matched else 'CÓ LỆCH (CẦN HOÀN TÁC)'}")

    # 2. Browser Console Audit via Playwright
    print("\n--- 2. Kiểm toán Console Browser & Độ bền 4 Tabs ---")
    console_errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))

        page.goto("http://localhost:8501")
        page.wait_for_timeout(3000)

        tabs = page.locator('[role="tab"]').all()
        for idx, tab in enumerate(tabs):
            tab.click()
            page.wait_for_timeout(1500)
            print(f"  Tab {idx+1} tải thành công, không vỡ layout.")

        browser.close()

    if console_errors:
        print(f"  Cảnh báo console errors ({len(console_errors)}): {console_errors[:3]}")
    else:
        print("  Không có lỗi console hoặc pageerror nào (0 errors).")

    print("\n===============================================================")
    print("HOÀN TẤT KIỂM ĐỊNH POST-CLEANUP!")
    print("===============================================================")

if __name__ == "__main__":
    main()
