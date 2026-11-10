import asyncio
import re
import sys
import json
from pathlib import Path
from playwright.async_api import async_playwright

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 30 diverse queries covering various categories
DIVERSE_QUERIES = [
    # 7 sample queries
    "túi sách màu hồng",
    "áo thun Puma nam",
    "váy dài màu đỏ dạ hội",
    "giày cao gót nữ màu đen",
    "ao do nam",
    "đồ đi biển",
    "túi xách nữ Lavie",
    # Additional 23 queries
    "áo sơ mi nam trắng công sở",
    "quần jean nữ xanh",
    "giày thể thao nam Nike",
    "đồng hồ nam dây kim loại",
    "kính mát nữ màu đen",
    "balo unisex màu xám",
    "đầm nữ màu hồng dự tiệc",
    "áo khoác gió nam màu xanh",
    "dép tông nam đi biển",
    "ví da nam màu nâu",
    "áo thun nữ cổ tròn màu vàng",
    "quần short nam thể thao",
    "chân váy nữ màu đen",
    "áo len nữ mùa đông",
    "thắt lưng da nam màu đen",
    "tất vớ nam màu trắng",
    "quần tây nam công sở màu xám",
    "sandal nữ cao gót",
    "áo blazer nam thanh lịch",
    "áo thun polo nam",
    "áo sơ mi nữ kẻ ca rô",
    "giày búp bê nữ màu be",
    "túi đeo chéo nữ nhỏ xinh"
]

# Catalog English words that MUST NOT leak into UI card text in user mode
FORBIDDEN_CATALOG_WORDS = [
    r"\bWomen\b",
    r"\bMen\b",
    r"\bPink\b",
    r"\bHandbag\b",
    r"\bHandbags\b",
    r"\bTshirts\b",
    r"\bShirts\b",
    r"\bHeels\b",
    r"\bCasual Shoes\b",
    r"\bSports Shoes\b",
    r"\bYellow\b",
    r"\bNavy Blue\b"
]

async def run_playwright_verification():
    out_dir = Path("eval/text2img_test/ui_before_after")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PLAYWRIGHT: KIỂM THỬ TỰ ĐỘNG TÊN TIẾNG VIỆT TRÊN 30 TRUY VẤN VÀ CHỤP ẢNH")
    print("=" * 70)

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        print("Đang truy cập http://localhost:8501...")
        await page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        print("Đợi ứng dụng Streamlit khởi tạo hoàn tất...")
        await page.wait_for_selector("button:has-text('túi sách màu hồng')", timeout=60000)
        print("Ứng dụng Streamlit đã nạp xong giao diện!")

        # -----------------------------------------------------------------
        # STEP 1: Screenshot & verification for "túi sách màu hồng"
        # -----------------------------------------------------------------
        print("\n--- 1. Kiểm thử 'túi sách màu hồng' & Chụp ảnh 1920x1080 ---")
        btn_pink_bag = page.locator("button:has-text('túi sách màu hồng')").first
        await btn_pink_bag.click()

        print("Đợi kết quả tìm kiếm (.product-card) hiển thị...")
        await page.wait_for_selector(".product-card", timeout=30000)
        await page.wait_for_timeout(2000)

        # Inspect first 3 card titles
        card_titles = await page.locator(".product-card .card-title-vi").all_inner_texts()
        print(f"Top 3 tên hiển thị cho 'túi sách màu hồng':")
        for idx, t in enumerate(card_titles[:3], 1):
            print(f"  Thẻ #{idx}: {t}")

        # Capture Screenshot
        shot_pink_path = out_dir / "after_vi_title_tui_sach_hong_1920x1080.png"
        await page.screenshot(path=str(shot_pink_path), full_page=True)
        print(f"📸 Đã lưu ảnh: {shot_pink_path}")

        # -----------------------------------------------------------------
        # STEP 2: Screenshot & verification for "áo thun Puma nam"
        # -----------------------------------------------------------------
        print("\n--- 2. Kiểm thử 'áo thun Puma nam' & Chụp ảnh 1920x1080 ---")
        btn_puma = page.locator("button:has-text('áo thun Puma nam')").first
        await btn_puma.click()

        await page.wait_for_timeout(2500)
        await page.wait_for_selector(".product-card", timeout=30000)

        puma_titles = await page.locator(".product-card .card-title-vi").all_inner_texts()
        print(f"Top 3 tên hiển thị cho 'áo thun Puma nam':")
        for idx, t in enumerate(puma_titles[:3], 1):
            print(f"  Thẻ #{idx}: {t}")

        shot_puma_path = out_dir / "after_vi_title_ao_thun_puma_1920x1080.png"
        await page.screenshot(path=str(shot_puma_path), full_page=True)
        print(f"📸 Đã lưu ảnh: {shot_puma_path}")

        # -----------------------------------------------------------------
        # STEP 3: Automated Scan on 30 Diverse Queries for Catalog English Leaks
        # -----------------------------------------------------------------
        print("\n--- 3. Quét rà soát 30 truy vấn: Kiểm tra không rò rỉ từ vựng tiếng Anh ---")
        leaks_found = {}

        for q_idx, q in enumerate(DIVERSE_QUERIES, 1):
            # Check if button exists or use text input
            btn_q = page.locator(f"button:has-text('{q}')").first
            if await btn_q.count() > 0 and await btn_q.is_visible():
                await btn_q.click()
            else:
                inp = page.locator(".stTextInput input").first
                await inp.fill(q)
                btn = page.locator("button:has-text('Tìm kiếm')").first
                await btn.click()

            await page.wait_for_timeout(1500)
            await page.wait_for_selector(".product-card", timeout=20000)

            # Get text of all cards
            cards_text = await page.locator(".product-card .card-title-vi, .product-card .card-tags-vi").all_inner_texts()
            joined_text = " \n ".join(cards_text)

            # Check for forbidden catalog words
            q_leaks = []
            for pat in FORBIDDEN_CATALOG_WORDS:
                matches = re.findall(pat, joined_text)
                if matches:
                    q_leaks.append((pat, len(matches)))

            if q_leaks:
                leaks_found[q] = q_leaks
                print(f"  [{q_idx:02d}/30] '{q}': 🔴 LỖI RÒ RỈ TIẾNG ANH: {q_leaks}")
            else:
                print(f"  [{q_idx:02d}/30] '{q}': 🟢 ĐẠT (100% Tiếng Việt chuẩn hóa)")

        print("\n" + "=" * 70)
        print("TỔNG KẾT RÀ SOÁT TỪ VỰNG TIẾNG ANH TRÊN 30 TRUY VẤN:")
        if leaks_found:
            print(f"❌ Có {len(leaks_found)} truy vấn phát hiện từ tiếng Anh chưa dịch:")
            for q, errs in leaks_found.items():
                print(f"  - '{q}': {errs}")
        else:
            print("🟢 100% HOÀN HẢO: Số lần xuất hiện từ tiếng Anh catalog trong thẻ = 0!")
        print("=" * 70)

        await context.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_playwright_verification())
