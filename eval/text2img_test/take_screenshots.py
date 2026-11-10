import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    out_dir = Path("eval/text2img_test/ui_before_after")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolutions = [
        ("1920x1080", 1920, 1080),
        ("1280x720", 1280, 720)
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        for res_name, width, height in resolutions:
            print(f"Capturing resolution {res_name}...")
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto("http://localhost:8501")
            page.wait_for_timeout(3000)

            # Find tabs using role="tab" or text
            tabs = page.locator('[role="tab"]').all()
            print(f"Found {len(tabs)} tabs at {res_name}")

            if len(tabs) == 0:
                # Fallback: find by tab container
                tabs = page.locator('div[data-testid="stTabs"] button').all()
                print(f"Fallback found {len(tabs)} tabs")

            for i, tab in enumerate(tabs):
                tab.click()
                page.wait_for_timeout(1500)
                file_path = out_dir / f"tab{i+1}_before_{res_name}.png"
                page.screenshot(path=str(file_path), full_page=True)
                print(f"Saved: {file_path}")

            page.close()
        browser.close()
    print("Done capturing before screenshots.")

if __name__ == "__main__":
    main()
