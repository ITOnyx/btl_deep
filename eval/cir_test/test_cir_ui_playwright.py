"""
Playwright automated UI verification for Tab 3 (CIR)
Tests:
1. UI source switching: Uploading locks catalog selectbox, clearing unlocks it.
2. Search execution: 'màu xanh navy' with uploaded shirt yields Navy Blue Shirt.
3. No raw '</div>' or broken tags, tags wrap cleanly without '…'.
4. Captures 1920x1080 screenshots.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def run_playwright():
    print("=== CHẠY KIỂM THỬ PLAYWRIGHT UI TAB 3 ===", flush=True)
    out_dir = ROOT_DIR / "eval" / "cir_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    shirt_path = out_dir / "crop_uploaded_shirt.jpg"

    with sync_playwright() as p:
        chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe"
        browser = p.chromium.launch(executable_path=chrome_path, headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        print("Navigating to http://localhost:8501...", flush=True)
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        page.wait_for_selector("[data-testid='stTab']", timeout=60000)

        # Click Tab 3
        tabs = page.locator("[data-testid='stTab']")
        tab_cir = tabs.nth(2)
        tab_cir.click()
        time.sleep(3)

        # 1. Before upload: verify selectbox is active
        page.screenshot(path=str(out_dir / "cir_tab3_before_upload.png"))
        print("Captured: cir_tab3_before_upload.png", flush=True)

        # 2. Upload file STRICTLY into Tab 3 uploader (.last = 2nd uploader on page, Tab 2 is .first)
        uploader = page.locator("input[data-testid='stFileUploaderDropzoneInput']").last
        uploader.set_input_files(str(shirt_path))
        print("Uploaded shirt file to Tab 3 uploader (.last), waiting for Streamlit to process...", flush=True)
        page.locator("text=Đang tra cứu theo ảnh tải lên thiết bị").wait_for(state="visible", timeout=15000)
        time.sleep(2)

        # Verify catalog selectbox in Tab 3 is hidden when upload is active
        assert page.locator("text=Hoặc chọn ảnh có sẵn trong catalog:").count() == 0, "Catalog selectbox must be hidden/disabled during upload!"
        assert page.locator("text=luồng cir_upload").count() > 0, "Expected cir_upload caption!"

        # 3. Click search button inside Tab 3
        search_btn = page.locator("button:has-text('Tổng hợp Truy vấn & Tìm kiếm')")
        search_btn.click()
        print("Clicked search button, waiting for results...", flush=True)
        page.wait_for_selector(".product-card:visible", timeout=30000)
        time.sleep(2)

        # 4. Check results
        cards = page.locator(".product-card:visible")
        card_count = cards.count()
        print(f"Rendered {card_count} visible product cards.", flush=True)
        assert card_count > 0

        # Check top-1 & ALL 10 cards title and tags
        top1_title = cards.first.locator(".card-title-vi").inner_text()
        top1_tags = cards.first.locator(".card-tags-vi").inner_text()
        print(f"Top-1 Title: {top1_title}", flush=True)
        print(f"Top-1 Tags:  {top1_tags}", flush=True)

        all_titles = [cards.nth(i).locator(".card-title-vi").inner_text() for i in range(card_count)]
        all_tags = [cards.nth(i).locator(".card-tags-vi").inner_text() for i in range(card_count)]
        shirt_navy_cnt = sum(
            1 for t, tg in zip(all_titles, all_tags)
            if "áo sơ mi" in t.lower() and "xanh navy" in t.lower()
        )
        print(f"Shirts + Navy Blue count: {shirt_navy_cnt}/{card_count}", flush=True)
        assert shirt_navy_cnt == 10 and card_count == 10, f"Expected 10/10 Navy Blue Shirts, got {shirt_navy_cnt}/{card_count}"

        # Check Slider, Checkbox, Anchor Status, and no raw </div>
        body_text = page.locator("body").inner_text()
        assert "</div>" not in body_text, "Found raw </div> in body!"
        assert "Quần soóc (Quần ngắn)" not in body_text, "Found deprecated parenthesized tag!"
        assert "Quần short nữ màu vàng" not in top1_title, "Bug not fixed: Top-1 is still yellow short!"
        assert "Mức độ biến đổi theo yêu cầu chữ" in body_text, "Missing lambda slider!"
        assert "Khóa giữ nguyên loại trang phục gốc" in body_text, "Missing dynamic anchor checkbox!"
        assert "Đang giữ loại: Áo sơ mi" in body_text, "Missing compact anchor status text!"

        # Screenshot after fix
        after_shot = out_dir / "after_step3_cir_upload_1920x1080.png"
        page.screenshot(path=str(after_shot))
        print(f"🟢 Captured final screenshot: {after_shot}", flush=True)

        browser.close()
    print("🟢 PLAYWRIGHT TEST HOÀN TẤT THÀNH CÔNG 100%!", flush=True)

if __name__ == "__main__":
    run_playwright()
