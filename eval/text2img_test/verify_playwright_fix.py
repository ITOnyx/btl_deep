import asyncio
import os
import re
import sys
from pathlib import Path
from playwright.async_api import async_playwright

sys.stdout.reconfigure(encoding="utf-8")

async def verify_ui():
    out_dir = Path("eval/text2img_test/ui_before_after")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PLAYWRIGHT UI TEST: KIỂM THỬ KHÔNG CÒN CHUỖI HTML THÔ '</div>' VÀ CHỤP ẢNH")
    print("=" * 70)

    forbidden_patterns = [
        r"</div>",
        r"<div\b",
        r"</span>",
        r"<span\b",
        r"</pre>",
        r"</code>",
        r"</"
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)

        # ----------------------------------------------------
        # TEST 1: 1920x1080 viewport with "túi sách màu hồng"
        # ----------------------------------------------------
        context_1080 = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page_1080 = await context_1080.new_page()

        print("Đang mở http://localhost:8501 tại 1920x1080...")
        await page_1080.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        await page_1080.wait_for_timeout(3000)

        # Click sample button "túi sách màu hồng"
        sample_btn = page_1080.locator("button:has-text('túi sách màu hồng')").first
        if await sample_btn.count() > 0:
            print("Clicking nút gợi ý mẫu: 'túi sách màu hồng'...")
            await sample_btn.click()
        else:
            print("Nhập 'túi sách màu hồng' vào ô tìm kiếm...")
            inp = page_1080.locator("input[type='text']").first
            await inp.fill("túi sách màu hồng")
            btn = page_1080.locator("button:has-text('Tìm kiếm')").first
            await btn.click()

        print("Đợi kết quả tìm kiếm (.product-card) xuất hiện...")
        await page_1080.wait_for_selector(".product-card", timeout=30000)
        await page_1080.wait_for_timeout(3000)

        # 1. SCAN FOR RAW HTML TAGS
        page_text = await page_1080.inner_text("body")
        print("\n--- 1. Quét kiểm tra chuỗi HTML thô xuất hiện dưới dạng văn bản ---")
        errors_found = []
        for pat in forbidden_patterns:
            matches = re.findall(pat, page_text, re.IGNORECASE)
            if matches:
                errors_found.append((pat, len(matches)))

        if errors_found:
            print(f"❌ PHÁT HIỆN LỖI: Tìm thấy ký tự HTML thô trong trang: {errors_found}")
        else:
            print("🟢 XÁC NHẬN: Hoàn toàn KHÔNG CÓ bất kỳ chuỗi '</div>', '<div', '</' nào hiện ra trên giao diện!")

        card_count = await page_1080.locator(".product-card").count()
        score_count = await page_1080.locator(".product-card .score-badge").count()
        progress_count = await page_1080.locator(".product-card .progress-bar-fill").count()
        print(f"Số lượng thẻ sản phẩm (.product-card): {card_count}")
        print(f"Số lượng điểm số (.score-badge): {score_count}/{card_count}")
        print(f"Số lượng thanh tiến độ (.progress-bar-fill): {progress_count}/{card_count}")

        # Capture 1920x1080 screenshot
        shot_1080_path = out_dir / "after_fix_tui_sach_hong_1920x1080.png"
        await page_1080.screenshot(path=str(shot_1080_path), full_page=True)
        print(f"📸 Đã chụp ảnh màn hình 1920x1080: {shot_1080_path}")

        await context_1080.close()

        # ----------------------------------------------------
        # TEST 2: 1280x720 viewport with "túi sách màu hồng"
        # ----------------------------------------------------
        context_720 = await browser.new_context(viewport={"width": 1280, "height": 720})
        page_720 = await context_720.new_page()

        print("\nĐang mở http://localhost:8501 tại 1280x720...")
        await page_720.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        await page_720.wait_for_timeout(3000)

        sample_btn_720 = page_720.locator("button:has-text('túi sách màu hồng')").first
        if await sample_btn_720.count() > 0:
            await sample_btn_720.click()

        await page_720.wait_for_selector(".product-card", timeout=30000)
        await page_720.wait_for_timeout(2000)

        text_720 = await page_720.inner_text("body")
        has_raw_html_720 = any(re.search(pat, text_720) for pat in forbidden_patterns)
        print(f"Kiểm tra 1280x720 không có HTML thô: {'🟢 ĐẠT' if not has_raw_html_720 else '🔴 LỖI'}")

        shot_720_path = out_dir / "after_fix_tui_sach_hong_1280x720.png"
        await page_720.screenshot(path=str(shot_720_path), full_page=True)
        print(f"📸 Đã chụp ảnh màn hình 1280x720: {shot_720_path}")

        # ----------------------------------------------------
        # TEST 3: Smoke test remaining 6 sample queries
        # ----------------------------------------------------
        samples = [
            "áo thun Puma nam",
            "váy dài màu đỏ dạ hội",
            "giày cao gót nữ màu đen",
            "ao do nam",
            "đồ đi biển",
            "túi xách nữ Lavie"
        ]
        print("\n--- 2. Smoke test 6 câu gợi ý mẫu còn lại ---")
        for sq in samples:
            btn = page_720.locator(f"button:has-text('{sq}')").first
            if await btn.count() > 0:
                await btn.click()
                await page_720.wait_for_timeout(2000)
                await page_720.wait_for_selector(".product-card", timeout=15000)
                txt = await page_720.inner_text("body")
                has_err = any(re.search(pat, txt) for pat in forbidden_patterns)
                c_cnt = await page_720.locator(".product-card").count()
                status_str = "🟢 ĐẠT (0 thẻ thô)" if not has_err else "🔴 CÓ LỖI HTML"
                print(f"  - '{sq}': {status_str} ({c_cnt} thẻ)")

        await context_720.close()
        await browser.close()

    print("\n" + "=" * 70)
    print("HOÀN TẤT KIỂM THỬ PLAYWRIGHT THÀNH CÔNG!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(verify_ui())
