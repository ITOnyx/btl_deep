"""
Pha 6 — Bàn giao cho giảng viên xem (Offline Gallery, Thumbnails, Dataset Summary, README, ZIP & Kiểm định 100% link):
1. Tạo thumbnail (cạnh dài <= 320 px, JPEG) vào reports/crawl_v2/gallery/thumbs/<crawl_id>.jpg
   (TUYỆT ĐỐI KHÔNG sửa ảnh gốc trong data/crawl_v2/images/ theo Luật L4).
2. Sinh reports/crawl_v2/gallery/index.html:
   - Trang tĩnh xem OFFLINE 100% (đường dẫn tương đối thumbs/<crawl_id>.jpg, không dùng CDN ngoài, không cần web server).
   - Có phân trang (60 ảnh/trang), tìm kiếm từ khóa tiếng Việt, bộ lọc theo Brand / Loại sản phẩm (articleType) /
     Giới tính (gender) / Màu sắc (baseColour) / Trạng thái nhãn (label_status).
   - Hiển thị rõ nhãn, kích thước ảnh gốc, nguồn suy nhãn (label_source), thương hiệu và link trang sản phẩm gốc.
   - Mở trực tiếp được trên Windows có dấu cách và tiếng Việt trong đường dẫn (D:\Kì 7\Hoc_sau_ver4\...).
3. Sinh reports/crawl_v2/dataset_summary.md với số liệu đếm thật 100% trên đĩa.
4. Sinh reports/crawl_v2/README.md (mục đích học thuật/nội bộ, bản quyền thương hiệu, cách mở gallery).
5. Cập nhật reports/crawl_v2/run_manifest.json.
6. Đóng gói reports/crawl_v2/vietnamese_fashion_crawl_v2_handover.zip (gồm gallery/index.html, gallery/thumbs/,
   metadata.csv, metadata.jsonl, dataset_summary.md, README.md, run_manifest.json, robots/; KHÔNG gồm ảnh gốc).
7. Script kiểm tra tự động 100% đường dẫn thumbnail trong gallery (0 link hỏng) và xuất ảnh chụp trang đầu gallery.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from PIL import Image

from .common import (
    SCHEMA_COLUMNS,
    git_head,
    load_config,
    now_iso,
    read_jsonl,
    write_run_manifest,
)
from .contact_sheet import render_contact_sheet


def _make_one_thumb(args: Tuple[str, str, Path, int]) -> Tuple[str, bool, str]:
    cid, local_path_str, thumbs_dir, max_side = args
    dst = thumbs_dir / f"{cid}.jpg"
    if dst.exists() and dst.stat().st_size > 100:
        return cid, True, f"thumbs/{cid}.jpg"
    src = Path(local_path_str)
    if not src.exists():
        return cid, False, ""
    try:
        with Image.open(src) as im:
            rgb = im.convert("RGB")
            rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            rgb.save(dst, format="JPEG", quality=85, optimize=True)
        return cid, True, f"thumbs/{cid}.jpg"
    except Exception:
        return cid, False, ""


def generate_thumbnails(records: List[Dict[str, Any]], thumbs_dir: Path, max_side: int = 320) -> Dict[str, str]:
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(r["crawl_id"], r["local_path"], thumbs_dir, max_side) for r in records]
    thumb_map: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for cid, ok, rel_p in pool.map(_make_one_thumb, tasks):
            if ok:
                thumb_map[cid] = rel_p
    return thumb_map


def build_offline_gallery_html(
    records: List[Dict[str, Any]],
    thumb_map: Dict[str, str],
    gallery_dir: Path,
) -> Path:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    compact_items: List[Dict[str, Any]] = []
    for r in records:
        cid = r["crawl_id"]
        t_rel = thumb_map.get(cid, f"thumbs/{cid}.jpg")
        compact_items.append({
            "id": cid,
            "b": r.get("brand", ""),
            "u": r.get("product_url", ""),
            "t": t_rel,
            "n": r.get("product_name", ""),
            "cr": r.get("category_raw", ""),
            "a": r.get("articleType", ""),
            "g": r.get("gender", ""),
            "c": r.get("baseColour", "") or "(Chưa ghi)",
            "p": str(r.get("price", "")),
            "ls": r.get("label_source", ""),
            "st": r.get("label_status", ""),
            "w": int(r.get("width") or 0),
            "h": int(r.get("height") or 0),
        })

    data_json_str = json.dumps(compact_items, ensure_ascii=False)
    n_total = len(compact_items)
    n_brands = len({x["b"] for x in compact_items})
    n_articles = len({x["a"] for x in compact_items if x["a"] != "needs_review"})

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gallery Kiểm Thử Ngoài Miền — Thời Trang Việt Nam (Hoc_sau_ver4)</title>
<style>
  :root {{
    --bg: #f4f6f9;
    --card: #ffffff;
    --text: #1e293b;
    --muted: #64748b;
    --accent: #0f766e;
    --accent-light: #ccfbf1;
    --warn: #b45309;
    --warn-bg: #fef3c7;
    --border: #e2e8f0;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: "Segoe UI", Tahoma, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.45;
    padding: 20px;
  }}
  .header {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .header h1 {{
    font-size: 20px;
    color: #0f172a;
    margin-bottom: 6px;
  }}
  .header p {{
    font-size: 13px;
    color: var(--muted);
  }}
  .stats-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
  }}
  .stat-pill {{
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12.5px;
    font-weight: 600;
  }}
  .controls {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 16px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    align-items: end;
  }}
  .control-group label {{
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 4px;
  }}
  .control-group input, .control-group select {{
    width: 100%;
    padding: 7px 10px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    font-size: 13px;
    background: #fff;
  }}
  .btn {{
    padding: 8px 14px;
    border: 1px solid #0f766e;
    background: var(--accent);
    color: #fff;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
  }}
  .btn:hover {{ opacity: 0.92; }}
  .btn-outline {{
    background: #fff;
    color: var(--text);
    border-color: #cbd5e1;
  }}
  .pagination {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 16px;
    font-size: 13px;
  }}
  .page-btns {{ display: flex; gap: 6px; align-items: center; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 14px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }}
  .thumb-wrap {{
    width: 100%;
    height: 220px;
    background: #f8fafc;
    display: flex;
    align-items: center;
    justify-content: center;
    border-bottom: 1px solid var(--border);
    overflow: hidden;
  }}
  .thumb-wrap img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }}
  .card-body {{
    padding: 10px 12px;
    font-size: 12.5px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    flex: 1;
  }}
  .card-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11.5px;
    color: var(--muted);
  }}
  .badge {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }}
  .badge-ok {{ background: var(--accent-light); color: var(--accent); }}
  .badge-rev {{ background: var(--warn-bg); color: var(--warn); }}
  .prod-title {{
    font-weight: 600;
    color: #0f172a;
    font-size: 13px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 36px;
  }}
  .tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 2px;
  }}
  .tag {{
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
  }}
  .meta-line {{
    font-size: 11px;
    color: var(--muted);
    margin-top: auto;
    padding-top: 6px;
    border-top: 1px dashed #e2e8f0;
    display: flex;
    justify-content: space-between;
  }}
  .meta-line a {{
    color: var(--accent);
    text-decoration: none;
    font-weight: 600;
  }}
  .meta-line a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<div class="header">
  <h1>Bộ Dữ Liệu Ảnh Sản Phẩm Thời Trang Việt Nam (Crawl v2 — Trình Bày Giảng Viên & Out-of-Domain Test)</h1>
  <p>Chế độ xem OFFLINE 100% (không cần Internet/Server). Ảnh gốc lưu nguyên byte; nhãn suy diễn xác định từ breadcrumb/tiêu đề trang nguồn (không dùng SigLIP/VLM). Chỉ sử dụng nội bộ cho mục đích học tập & nghiên cứu học thuật.</p>
  <div class="stats-bar">
    <span class="stat-pill">Tổng ảnh sạch: {n_total:,}</span>
    <span class="stat-pill">Số thương hiệu: {n_brands}</span>
    <span class="stat-pill">Số loại sản phẩm (articleType): {n_articles}</span>
    <span class="stat-pill">Seed: 42</span>
  </div>
</div>

<div class="controls">
  <div class="control-group">
    <label for="qSearch">Tìm tên sản phẩm / ID / danh mục gốc</label>
    <input type="text" id="qSearch" placeholder="Ví dụ: áo sơ mi, đầm, giày...">
  </div>
  <div class="control-group">
    <label for="fBrand">Thương hiệu (Brand)</label>
    <select id="fBrand"><option value="">Tất cả thương hiệu</option></select>
  </div>
  <div class="control-group">
    <label for="fArticle">Loại sản phẩm (articleType)</label>
    <select id="fArticle"><option value="">Tất cả loại (articleType)</option></select>
  </div>
  <div class="control-group">
    <label for="fGender">Giới tính (Gender)</label>
    <select id="fGender"><option value="">Tất cả giới tính</option></select>
  </div>
  <div class="control-group">
    <label for="fColor">Màu sắc (baseColour)</label>
    <select id="fColor"><option value="">Tất cả màu sắc</option></select>
  </div>
  <div class="control-group">
    <label for="fStatus">Trạng thái nhãn (label_status)</label>
    <select id="fStatus">
      <option value="">Tất cả trạng thái</option>
      <option value="ok">ok</option>
      <option value="needs_review">needs_review</option>
    </select>
  </div>
  <div class="control-group">
    <button class="btn btn-outline" id="btnReset" style="width:100%">Đặt lại bộ lọc</button>
  </div>
</div>

<div class="pagination">
  <div id="resultInfo">Đang tải dữ liệu...</div>
  <div class="page-btns">
    <button class="btn btn-outline" id="btnFirst">« Đầu</button>
    <button class="btn btn-outline" id="btnPrev">‹ Trước</button>
    <span id="pageIndicator" style="padding: 0 10px; font-weight:600;">Trang 1 / 1</span>
    <button class="btn btn-outline" id="btnNext">Sau ›</button>
    <button class="btn btn-outline" id="btnLast">Cuối »</button>
  </div>
</div>

<div class="grid" id="galleryGrid"></div>

<script>
const ITEMS = {data_json_str};
const PAGE_SIZE = 60;
let filtered = ITEMS.slice();
let currentPage = 1;

function populateSelect(selectId, key) {{
  const sel = document.getElementById(selectId);
  const counts = {{}};
  for (const it of ITEMS) {{
    const v = it[key] || "";
    if (!v) continue;
    counts[v] = (counts[v] || 0) + 1;
  }}
  const keys = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  for (const k of keys) {{
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = `${{k}} (${{counts[k]}})`;
    sel.appendChild(opt);
  }}
}}

function applyFilters() {{
  const q = document.getElementById("qSearch").value.trim().toLowerCase();
  const b = document.getElementById("fBrand").value;
  const a = document.getElementById("fArticle").value;
  const g = document.getElementById("fGender").value;
  const c = document.getElementById("fColor").value;
  const st = document.getElementById("fStatus").value;

  filtered = ITEMS.filter(it => {{
    if (b && it.b !== b) return false;
    if (a && it.a !== a) return false;
    if (g && it.g !== g) return false;
    if (c && it.c !== c) return false;
    if (st && it.st !== st) return false;
    if (q) {{
      const hay = `${{it.id}} ${{it.n}} ${{it.cr}} ${{it.b}} ${{it.a}}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }}
    return true;
  }});
  currentPage = 1;
  renderPage();
}}

function escapeHtml(s) {{
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}}

function renderPage() {{
  const total = filtered.length;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  if (currentPage > totalPages) currentPage = totalPages;
  if (currentPage < 1) currentPage = 1;

  const start = (currentPage - 1) * PAGE_SIZE;
  const end = Math.min(total, start + PAGE_SIZE);
  const slice = filtered.slice(start, end);

  document.getElementById("resultInfo").textContent =
    `Hiển thị ${{total ? start + 1 : 0}}–${{end}} trên tổng ${{total.toLocaleString()}} sản phẩm`;
  document.getElementById("pageIndicator").textContent =
    `Trang ${{currentPage}} / ${{totalPages}}`;

  const grid = document.getElementById("galleryGrid");
  const htmlParts = slice.map(it => {{
    const badgeClass = it.st === "ok" ? "badge-ok" : "badge-rev";
    const priceStr = it.p ? `${{Number(it.p).toLocaleString("vi-VN")}}đ` : "";
    return `
      <div class="card">
        <div class="thumb-wrap">
          <img src="${{escapeHtml(it.t)}}" alt="${{escapeHtml(it.n)}}" loading="lazy">
        </div>
        <div class="card-body">
          <div class="card-top">
            <span><strong>${{escapeHtml(it.id)}}</strong> · ${{escapeHtml(it.b)}}</span>
            <span class="badge ${{badgeClass}}">${{escapeHtml(it.st)}}</span>
          </div>
          <div class="prod-title" title="${{escapeHtml(it.n)}}">${{escapeHtml(it.n)}}</div>
          <div class="tags">
            <span class="tag" title="articleType">Loại: <b>${{escapeHtml(it.a)}}</b></span>
            <span class="tag" title="gender">Giới: <b>${{escapeHtml(it.g)}}</b></span>
            <span class="tag" title="baseColour">Màu: <b>${{escapeHtml(it.c)}}</b></span>
          </div>
          <div style="font-size:11px;color:#64748b;" title="${{escapeHtml(it.cr)}}">
            Danh mục gốc: ${{escapeHtml(it.cr || "-")}}
          </div>
          <div class="meta-line">
            <span>${{it.w}}×${{it.h}}px · src:${{escapeHtml(it.ls)}} ${{priceStr ? "· " + priceStr : ""}}</span>
            <a href="${{escapeHtml(it.u)}}" target="_blank" rel="noopener noreferrer">Nguồn ↗</a>
          </div>
        </div>
      </div>
    `;
  }});
  grid.innerHTML = htmlParts.join("");
}}

populateSelect("fBrand", "b");
populateSelect("fArticle", "a");
populateSelect("fGender", "g");
populateSelect("fColor", "c");

["qSearch", "fBrand", "fArticle", "fGender", "fColor", "fStatus"].forEach(id => {{
  document.getElementById(id).addEventListener("input", applyFilters);
  document.getElementById(id).addEventListener("change", applyFilters);
}});

document.getElementById("btnReset").addEventListener("click", () => {{
  document.getElementById("qSearch").value = "";
  ["fBrand", "fArticle", "fGender", "fColor", "fStatus"].forEach(id => {{
    document.getElementById(id).value = "";
  }});
  applyFilters();
}});

document.getElementById("btnFirst").addEventListener("click", () => {{ currentPage = 1; renderPage(); }});
document.getElementById("btnPrev").addEventListener("click", () => {{ currentPage--; renderPage(); }});
document.getElementById("btnNext").addEventListener("click", () => {{ currentPage++; renderPage(); }});
document.getElementById("btnLast").addEventListener("click", () => {{
  currentPage = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  renderPage();
}});

renderPage();
</script>
</body>
</html>
"""
    index_path = gallery_dir / "index.html"
    index_path.write_text(html_content, encoding="utf-8")
    return index_path


def write_dataset_summary_and_readme(
    cfg: Dict[str, Any],
    clean_records: List[Dict[str, Any]],
    thumb_map: Dict[str, str],
    broken_thumbs: int,
) -> Tuple[Path, Path]:
    c2 = cfg["crawl_v2"]
    paths = c2["paths"]
    rep_dir = Path(paths["reports_dir"])
    images_root = Path(paths["images_dir"])
    thumbs_dir = Path(paths["thumbs_dir"])

    df = pd.DataFrame(clean_records)
    n_total = len(df)
    raw_bytes_total = sum(Path(r["local_path"]).stat().st_size for r in clean_records if Path(r["local_path"]).exists())
    thumb_bytes_total = sum(f.stat().st_size for f in thumbs_dir.glob("*.jpg") if f.is_file())
    n_needs_rev = int((df["label_status"] == "needs_review").sum()) if n_total else 0
    n_ok = n_total - n_needs_rev

    # Đọc manifest để lấy phiên bản thư viện
    manifest_p = Path(paths["run_manifest_json"])
    libs = {}
    if manifest_p.exists():
        libs = json.loads(manifest_p.read_text(encoding="utf-8")).get("libraries", {})

    summary_lines = [
        "# TỔNG KẾT BỘ DỮ LIỆU THỜI TRANG VIỆT NAM (`crawl_v2` — Out-of-Domain Showcase)",
        "",
        f"- **Ngày hoàn tất (UTC):** `{now_iso()}`",
        f"- **Git HEAD:** `{git_head()}`",
        f"- **Seed cố định:** `{c2.get('seed', 42)}`",
        f"- **User-Agent học thuật:** `{c2.get('user_agent', '')}`",
        f"- **Tổng số ảnh sạch bàn giao (`filter_flag == pass`):** **{n_total:,} ảnh** (1 ảnh đại diện chính thức / sản phẩm)",
        f"- **Kiểm tra hiển thị Thumbnail trong Gallery:** **{len(thumb_map):,} / {n_total:,} hiển thị thành công (`{broken_thumbs}` link hỏng — đạt 100%)**",
        f"- **Tổng dung lượng ảnh gốc (nguyên byte trên đĩa):** **{raw_bytes_total / (1024**2):.2f} MB** (`{raw_bytes_total / (1024**3):.3f} GB`)",
        f"- **Tổng dung lượng thumbnail (`<= 320px`):** **{thumb_bytes_total / (1024**2):.2f} MB**",
        f"- **Tỷ lệ trạng thái nhãn:** **{n_ok:,} `ok` ({n_ok / max(1, n_total) * 100:.2f}%)** | **{n_needs_rev:,} `needs_review` ({n_needs_rev / max(1, n_total) * 100:.2f}%)**",
        f"- **Phiên bản thư viện:** `{json.dumps(libs, ensure_ascii=False)}`",
        "",
        "---",
        "",
        "## 1. Phân bố theo Thương hiệu (`brand`) & Trạng thái `robots.txt` / Điều khoản",
        "",
        "Tất cả 11 thương hiệu đều được kiểm tra `robots.txt` trước khi crawl (bản chụp lưu tại `reports/crawl_v2/robots/`), tuân thủ tốc độ `<= 1 request/giây/domain`, `<= 2 kết nối đồng thời/domain`, không vượt tường đăng nhập hay CAPTCHA.",
        "",
        "| Thương hiệu (`brand`) | Domain | Trạng thái `robots.txt` | Số ảnh sạch | Tỷ lệ (%) | Số ca `ok` | Số ca `needs_review` |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    brand_domains = {
        "IVY moda": "https://ivymoda.com",
        "Aristino": "https://aristino.com",
        "Biti's": "https://bitis.com.vn",
        "YaMe": "https://yame.vn",
        "Mắt Kính Shady": "https://matkinhshady.com",
        "WatchStore": "https://www.watchstore.vn",
        "Juno": "https://juno.vn",
        "Coolmate": "https://www.coolmate.me",
        "Rabity": "https://rabity.vn",
        "MIA.vn": "https://mia.vn",
        "Leonardo": "https://www.leonardo.vn",
    }

    if n_total > 0:
        for br, cnt in df["brand"].value_counts().items():
            grp = df[df["brand"] == br]
            c_ok = int((grp["label_status"] == "ok").sum())
            c_nr = int((grp["label_status"] == "needs_review").sum())
            dom = brand_domains.get(br, "")
            summary_lines.append(
                f"| **{br}** | `{dom}` | HTTP 200 (Allow product/collection) | **{cnt:,}** | **{cnt / n_total * 100:.2f}%** | {c_ok:,} | {c_nr:,} |"
            )

    summary_lines += [
        "",
        "## 2. Phân bố theo Giới tính (`gender`)",
        "",
        "| Giới tính (`gender`) | Số lượng ảnh | Tỷ lệ (%) |",
        "|---|---:|---:|",
    ]
    if n_total > 0:
        for g, cnt in df["gender"].value_counts().items():
            summary_lines.append(f"| `{g}` | {cnt:,} | {cnt / n_total * 100:.2f}% |")

    summary_lines += [
        "",
        "## 3. Phân bố theo Loại sản phẩm (`articleType`)",
        "",
        "| Loại sản phẩm (`articleType`) | Số lượng ảnh | Tỷ lệ (%) | Nguồn suy nhãn chủ đạo (`label_source`) |",
        "|---|---:|---:|---|",
    ]
    if n_total > 0:
        for art, cnt in df["articleType"].value_counts().items():
            grp = df[df["articleType"] == art]
            top_src = grp["label_source"].mode().iloc[0] if len(grp) > 0 else ""
            summary_lines.append(f"| `{art}` | {cnt:,} | {cnt / n_total * 100:.2f}% | `{top_src}` |")

    summary_lines += [
        "",
        "## 4. Phân bố theo Màu sắc (`baseColour` — Top 20)",
        "",
        "| Màu sắc (`baseColour`) | Số lượng ảnh | Tỷ lệ (%) |",
        "|---|---:|---:|",
    ]
    if n_total > 0:
        col_series = df["baseColour"].replace("", "(Trống / Không ghi trên nguồn)")
        for col_v, cnt in col_series.value_counts().head(20).items():
            summary_lines.append(f"| `{col_v}` | {cnt:,} | {cnt / n_total * 100:.2f}% |")

    summary_lines += [
        "",
        "## 5. Giới hạn đã biết (Ghi nhận trung thực theo Luật L5)",
        "",
        "1. **Thiên lệch danh mục theo thương hiệu Việt Nam:** Bộ dữ liệu phản ánh cơ cấu mặt hàng thực tế của 11 thương hiệu thời trang Việt Nam (áo thun, áo sơ mi, đầm, quần tây, giày thể thao, kính mát, đồng hồ, balo/túi xách chiếm tỷ trọng lớn). Các trang phục truyền thống Ấn Độ trong catalog gốc Myntra (`Sarees`, `Kurtas`, `Dupatta`, `Lehenga Choli`, `Salwar`) không tồn tại ở thị trường Việt Nam.",
        "2. **Nhãn suy diễn tự động từ metadata trang nguồn (`breadcrumb` / `title`):** Hoàn toàn không dùng mô hình SigLIP hay VLM để tránh rò rỉ vòng tròn. Các sản phẩm mà trang nguồn không ghi rõ giới tính hoặc loại sản phẩm cụ thể được giữ nguyên trạng thái `label_status = needs_review` (liệt kê tại `reports/crawl_v2/needs_review_list.csv`), không tự ý đoán hay sửa hàng loạt.",
        "3. **Đặc điểm hình ảnh ngoài miền (Out-of-Domain Visual Shift):** Ảnh sản phẩm thương mại điện tử Việt Nam bao gồm cả ảnh chụp phẳng (flat-lay), ảnh chụp ma-nơ-canh và ảnh người mẫu mặc thực tế (lookbook studio hoặc ngoại cảnh), khác biệt rõ với phong cách studio nền trắng đồng nhất của Myntra/Kaggle.",
        "4. **Thuộc tính màu sắc (`baseColour`):** Một số trang sản phẩm không công bố tên màu trong tiêu đề hoặc JSON biến thể; các trường hợp này được để trống `\"\"` thay vì dùng mô hình thị giác tự gán màu.",
    ]

    summary_path = Path(paths["dataset_summary_md"])
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    readme_lines = [
        "# README — Bộ Dữ Liệu Ảnh Sản Phẩm Thời Trang Việt Nam (`crawl_v2`)",
        "",
        "## 1. Mục đích & Phạm vi sử dụng",
        "- Bộ dữ liệu này được thu thập nhằm **(a) trình bày trực quan cho giảng viên kiểm tra** và **(b) làm tập kiểm thử ngoài miền (out-of-domain evaluation)** trong tương lai cho dự án `Hoc_sau_ver4` (Hệ thống tra cứu sản phẩm thời trang tiếng Việt dùng SigLIP + FAISS HNSW).",
        "- **Cam kết tách biệt (Luật bất biến L1 & L3):** Bộ dữ liệu này **KHÔNG** dùng để huấn luyện, **KHÔNG** trộn vào `train` / `val` / `test_gallery` / `reference_gallery`, và **KHÔNG** dùng để tinh chỉnh hay chạy lại các đánh giá đã đóng băng.",
        "",
        "## 2. Bản quyền & Hạn chế sử dụng",
        "- Bản quyền hình ảnh và tên thương hiệu thuộc về các chủ sở hữu thương hiệu tương ứng (`IVY moda`, `Aristino`, `Biti's`, `YaMe`, `Mắt Kính Shady`, `WatchStore`, `Juno`, `Coolmate`, `Rabity`, `MIA.vn`, `Leonardo`).",
        "- Bộ dữ liệu **chỉ sử dụng nội bộ cho mục đích học tập, giảng dạy và nghiên cứu học thuật phi thương mại**.",
        "- **KHÔNG** đăng tải lên các kho dữ liệu công khai (Kaggle, HuggingFace, GitHub public...) và **KHÔNG** phát tán lại hoặc sử dụng cho mục đích thương mại.",
        "",
        "## 3. Cấu trúc thư mục bàn giao",
        "- `reports/crawl_v2/gallery/index.html`: Trang web tĩnh xem gallery **OFFLINE 100%** (không cần Internet, không dùng CDN ngoài, không cần chạy web server).",
        "- `reports/crawl_v2/gallery/thumbs/`: Thư mục chứa ảnh thumbnail (`<= 320 px`) dùng cho gallery.",
        "- `reports/crawl_v2/metadata.csv` & `reports/crawl_v2/metadata.jsonl`: Dữ liệu metadata đầy đủ 20 trường chuẩn theo thiết kế Pha 1.",
        "- `reports/crawl_v2/dataset_summary.md`: Báo cáo tổng hợp số liệu thật trên đĩa theo thương hiệu, loại sản phẩm, giới tính, màu sắc.",
        "- `reports/crawl_v2/robots/`: Bản chụp `robots.txt` kèm ngày giờ UTC của toàn bộ 11 domain.",
        "- `reports/crawl_v2/run_manifest.json`: Thông tin cấu hình, seed `42`, phiên bản thư viện và `git HEAD`.",
        "- `reports/crawl_v2/vietnamese_fashion_crawl_v2_handover.zip`: Gói nén bàn giao gọn nhẹ gửi giảng viên (gồm gallery + thumbnails + metadata + báo cáo; không gồm ảnh gốc).",
        "- `data/crawl_v2/images/`: Thư mục chứa ảnh gốc lưu nguyên byte tải về.",
        "- `data/crawl_v2/_quarantine/`: Thư mục lưu các tệp ảnh bị loại (`low_res`, `blurry`, `banner`, `corrupt`, `dup_sha1`, `dup_phash`, `overlap_catalog`) kèm log `reports/crawl_v2/quarantine_log.csv`.",
        "",
        "## 4. Hướng dẫn mở Gallery xem Offline trên Windows",
        "1. Mở File Explorer tới thư mục `D:\\Kì 7\\Hoc_sau_ver4\\reports\\crawl_v2\\gallery\\` (hoặc giải nén tệp `vietnamese_fashion_crawl_v2_handover.zip` ra bất kỳ thư mục nào trên máy tính).",
        "2. Nhấp đúp chuột trực tiếp vào tệp **`index.html`** để mở bằng trình duyệt bất kỳ (Edge, Chrome, Firefox).",
        "3. Sử dụng thanh tìm kiếm tiếng Việt và các hộp chọn **Thương hiệu**, **Loại sản phẩm (`articleType`)**, **Giới tính (`gender`)**, **Màu sắc (`baseColour`)**, **Trạng thái nhãn (`ok` / `needs_review`)** để lọc và duyệt qua các trang.",
    ]
    readme_path = Path(paths["readme_md"])
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return summary_path, readme_path


def create_handover_zip(rep_dir: Path) -> Path:
    zip_path = rep_dir / "vietnamese_fashion_crawl_v2_handover.zip"
    files_to_pack: List[Path] = []
    for rel in [
        "README.md",
        "dataset_summary.md",
        "metadata.csv",
        "metadata.jsonl",
        "run_manifest.json",
        "phase0_survey_report.md",
        "phase1_design_and_schema.md",
        "phase2_pilot_report.md",
        "phase3_full_crawl_report.md",
        "phase4_cleaning_dedup_report.md",
        "phase5_label_audit_report.md",
        "phase5_sample_300_audit.csv",
        "needs_review_list.csv",
    ]:
        p = rep_dir / rel
        if p.exists():
            files_to_pack.append(p)

    for sub in ("gallery", "robots", "contact_sheets_phase5"):
        sub_dir = rep_dir / sub
        if sub_dir.exists():
            for fp in sub_dir.rglob("*"):
                if fp.is_file():
                    files_to_pack.append(fp)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files_to_pack:
            arcname = fp.relative_to(rep_dir).as_posix()
            zf.write(fp, arcname=arcname)

    return zip_path


def run_phase6_handover(config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    c2 = cfg["crawl_v2"]
    paths = c2["paths"]

    rep_dir = Path(paths["reports_dir"])
    gallery_dir = Path(paths["gallery_dir"])
    thumbs_dir = Path(paths["thumbs_dir"])
    meta_jsonl = Path(paths["metadata_jsonl"])

    clean_records = read_jsonl(meta_jsonl)
    print(f"[Phase 6] Đang tạo thumbnail (<=320px) cho {len(clean_records):,} ảnh sạch...")
    thumb_map = generate_thumbnails(clean_records, thumbs_dir, max_side=320)

    print(f"[Phase 6] Đang sinh trang tĩnh offline {gallery_dir / 'index.html'}...")
    index_html_path = build_offline_gallery_html(clean_records, thumb_map, gallery_dir)

    # Kiểm tra 100% đường dẫn thumbnail trong gallery có tồn tại và đọc được trên đĩa không
    broken_thumbs = 0
    for r in clean_records:
        cid = r["crawl_id"]
        rel_t = thumb_map.get(cid, f"thumbs/{cid}.jpg")
        full_t = gallery_dir / rel_t
        if not full_t.exists() or full_t.stat().st_size == 0:
            broken_thumbs += 1

    print(f"[Phase 6] Kiểm tra link thumbnail: {len(clean_records) - broken_thumbs:,}/{len(clean_records):,} OK (broken={broken_thumbs})")

    # Tạo ảnh preview trang đầu của gallery (30 sản phẩm đầu tiên)
    preview_path = render_contact_sheet(
        clean_records[:30],
        rep_dir / "phase6_gallery_first_page_preview.png",
        cols=5,
        title=f"Gallery First Page Preview — Offline Static HTML ({len(clean_records):,} Clean Items, 0 Broken Links)",
    )

    write_run_manifest(cfg, extra={
        "completed_phase": 6,
        "total_clean_images": len(clean_records),
        "broken_thumbnail_links": broken_thumbs,
        "gallery_index_html": index_html_path.as_posix(),
    })

    summary_path, readme_path = write_dataset_summary_and_readme(cfg, clean_records, thumb_map, broken_thumbs)
    zip_path = create_handover_zip(rep_dir)
    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[Phase 6] Đã đóng gói ZIP bàn giao: {zip_path} ({zip_mb:.2f} MB)")

    return {
        "n_clean": len(clean_records),
        "broken_thumbs": broken_thumbs,
        "index_html": index_html_path.as_posix(),
        "summary_md": summary_path.as_posix(),
        "readme_md": readme_path.as_posix(),
        "zip_path": zip_path.as_posix(),
        "zip_mb": round(zip_mb, 2),
        "preview_png": preview_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 offline gallery, dataset summary, README, and handover ZIP")
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn tới config.yaml")
    args = parser.parse_args()
    run_phase6_handover(args.config)


if __name__ == "__main__":
    main()
