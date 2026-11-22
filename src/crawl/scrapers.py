"""
11 Scrapers cho Crawl v2 (1 ảnh sản phẩm chính thức đại diện / sản phẩm):
1. Haravan/Shopify JSON API (bitis, yame, aristino, rabity, leonardo, matkinhshady, watchstore):
   ưu tiên lấy theo danh sách collections (phủ đa dạng loại/giới tính) rồi bổ sung từ /collections/all/products.json hoặc /products.json.
2. IVY moda (ivymoda): duyệt phân trang danh mục công khai /danh-muc/<slug>/<page> (24 SP/trang, ảnh CDN 1400px), fallback sitemap.
3. Juno (juno): duyệt sitemap /sitemap/product/1..15 + JSON-LD Product & breadcrumb.
4. Coolmate (coolmate): duyệt sitemap product0.xml + product1.xml + HTML/RSC.
5. MIA.vn (mia): duyệt danh mục /vali, /balo, /tui-xach, /phu-kien + fallback sitemap/products.xml.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, Iterator, List, Set, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .common import PoliteHttpClient

_COLOR_RE = re.compile(
    r"\b(Đen|Trắng|Xám|Xanh\s+Navy|Xanh\s+Đen|Xanh\s+Dương|Xanh\s+Rêu|Xanh\s+Lá|Xanh\s+Ngọc|"
    r"Be|Kem|Nâu|Đỏ|Vàng|Hồng|Tím|Cam|Kaki|Bạc|Vàng\s+Kim|Xanh|Bò|Rêu|"
    r"Black|White|Grey|Gray|Navy|Blue|Green|Olive|Beige|Cream|Brown|Red|Yellow|Pink|Purple|Orange|Silver|Gold|Charcoal|Champagne)\b",
    flags=re.IGNORECASE,
)


def _extract_color_from_json_product(p: Dict[str, Any], title: str, tags_str: str = "") -> str:
    # Kiểm tra tag dạng 'Màu Đen' hoặc 'GR.COR/Trắng'
    m_tag = re.search(r"(?:GR\.COR/|Màu\s+)([A-Za-zÀ-ỹ\s]+?)(?:,|;|$)", tags_str, flags=re.IGNORECASE)
    if m_tag:
        return m_tag.group(1).strip()

    variants = p.get("variants") or []
    for v in variants[:3]:
        for opt_key in ("option1", "option2", "option3"):
            val = (v.get(opt_key) or "").strip()
            if not val or val.isdigit():
                continue
            if val.lower() in ("default title", "mặc định", "freesize", "free size", "s", "m", "l", "xl", "xxl", "xs"):
                continue
            mc = _COLOR_RE.search(val)
            if mc:
                return mc.group(1).strip()
    m = _COLOR_RE.search(title)
    return m.group(1).strip() if m else ""


def _extract_price_from_json_product(p: Dict[str, Any]) -> str:
    variants = p.get("variants") or []
    if variants:
        pr = variants[0].get("price")
        if pr is not None and str(pr).strip():
            return str(pr).strip()
    return ""


def _gender_from_collection_slug(col: str, default_gender: str) -> str:
    c = f"-{col.lower()}-"
    if "be-trai" in c:
        return "Boys"
    if "be-gai" in c:
        return "Girls"
    if "unisex" in c or "dong-ho-doi" in c:
        return "Unisex"
    has_nam = bool(re.search(r"-(nam|men)-", c))
    has_nu = bool(re.search(r"-(nu|women)-", c))
    if has_nam and has_nu:
        return "Unisex"
    if has_nu:
        return "Women"
    if has_nam:
        return "Men"
    return default_gender


SOURCE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "yame": {
        "base_url": "https://yame.vn",
        "brand": "YaMe",
        "default_gender": "Men",
        "has_root_products_json": True,
        "extra_delay_sec": 0.6,
        "collections": [
            "ao-thun", "ao-so-mi", "ao-thun-polo", "ao-khoac", "quan-jeans",
            "quan-tay", "quan-short", "quan-dai", "balo", "tui-deo",
            "non", "day-nit", "vi-ngang", "vo-cong-thai-hoc", "dep", "giay", "gang-tay",
        ],
    },
    "bitis": {
        "base_url": "https://bitis.com.vn",
        "brand": "Biti's",
        "default_gender": "",
        "has_root_products_json": False,
        "extra_delay_sec": 0.2,
        "collections": [
            "hunter-nam", "hunter-nu", "giay-the-thao-nam", "giay-the-thao-nu",
            "sandal-nam", "sandal-nu", "dep-nam", "dep-nu", "giay-tay-nam",
            "giay-bup-be-nu", "giay-the-thao-be-trai", "giay-the-thao-be-gai", "balo-tui-xach",
        ],
    },
    "aristino": {
        "base_url": "https://aristino.com",
        "brand": "Aristino",
        "default_gender": "Men",
        "has_root_products_json": False,
        "extra_delay_sec": 0.2,
        "collections": [
            "ao-so-mi", "ao-polo", "ao-thun", "ao-blazer", "suit", "ao-khoac",
            "ao-len", "ao-ni-sweatshirt", "quan-au", "quan-kaki", "quan-jeans",
            "quan-short", "quan-the-thao", "quan-boxer", "ca-vat", "that-lung",
            "vi-da", "giay-da", "cap-da-tui-xach", "tat",
        ],
    },
    "rabity": {
        "base_url": "https://rabity.vn",
        "brand": "Rabity",
        "default_gender": "",
        "has_root_products_json": False,
        "extra_delay_sec": 0.2,
        "collections": [
            "ao-thun-be-trai", "ao-thun-be-gai", "ao-so-mi-be-trai", "ao-so-mi-be-gai",
            "ao-khoac-be-gai", "ao-khoac-be-trai", "ao-hoodie-be-trai", "dam-vay-be-gai",
            "quan-short-be-trai", "quan-short-be-gai", "quan-dai-be-trai", "quan-dai-be-gai",
            "do-bo-be-trai", "do-bo-ngan-tay-be-gai", "do-boi-be-trai",
            "sandal-be-trai", "sneaker-be-trai", "dep-be-trai", "giay-bup-be-be-gai",
            "mu-be-gai", "balo-tui-xach-be-trai",
        ],
    },
    "leonardo": {
        "base_url": "https://www.leonardo.vn",
        "brand": "Leonardo",
        "default_gender": "Men",
        "has_root_products_json": True,
        "extra_delay_sec": 0.5,
        "collections": [
            "vi-da-nam", "that-lung-nam", "balo", "balo-nu", "tui-deo-cheo",
            "tui-bao-t", "tui-tote", "tui-xach-nu", "giay-da-dep-nam", "dep-da-nam", "dep-sandals",
        ],
    },
    "matkinhshady": {
        "base_url": "https://matkinhshady.com",
        "brand": "Mắt Kính Shady",
        "default_gender": "Unisex",
        "has_root_products_json": False,
        "extra_delay_sec": 0.2,
        "collections": [
            "kinh-mat-nam", "kinh-mat-nu", "kinh-mat-unisex",
            "gong-kinh-can-nam", "gong-kinh-can-nu", "gong-kinh",
        ],
    },
    "watchstore": {
        "base_url": "https://www.watchstore.vn",
        "brand": "WatchStore",
        "default_gender": "Unisex",
        "has_root_products_json": True,
        "extra_delay_sec": 0.5,
        "collections": [
            "dong-ho-nam", "dong-ho-nu", "dong-ho-doi", "dong-ho-g-shock",
            "dong-ho-baby-g", "dong-ho-casio", "dong-ho-citizen", "dong-ho-orient",
            "dong-ho-seiko", "dong-ho-tissot", "dong-ho-daniel-wellington", "day-chuyen", "bong-tai",
        ],
    },
}


def make_json_store_scraper(source_name: str):
    def _crawl(
        client: PoliteHttpClient,
        limit: int,
        seen_urls: Set[str],
        seed: int = 42,
    ) -> Iterator[Dict[str, Any]]:
        meta = SOURCE_DEFAULTS[source_name]
        base_url = meta["base_url"].rstrip("/")
        brand = meta["brand"]
        default_gender = meta["default_gender"]
        extra_delay = float(meta.get("extra_delay_sec", 0.2))
        collections = list(meta["collections"])

        rng = random.Random(seed)
        yielded = 0

        # Đối với các shop có root /products.json (như Leonardo, YaMe), nếu danh sách collections ít hoặc cần lấy nhanh
        # ta ưu tiên quét trực tiếp /products.json để tránh spam từng collection bị 429
        if meta.get("has_root_products_json"):
            all_cols = ["__ALL__"] + collections
        else:
            all_cols = collections + ["__ALL__"]

        per_col_quota = max(10, (limit // max(1, len(all_cols))) + 25)

        for col in all_cols:
            if yielded >= limit:
                break
            is_all = col == "__ALL__"
            col_quota = (limit - yielded) if is_all else per_col_quota
            col_count = 0
            page = 1
            while col_count < col_quota and yielded < limit:
                if is_all:
                    if meta["has_root_products_json"]:
                        url = f"{base_url}/products.json?limit=250&page={page}"
                    else:
                        url = f"{base_url}/collections/all/products.json?limit=250&page={page}"
                else:
                    url = f"{base_url}/collections/{col}/products.json?limit=250&page={page}"

                resp = client.get(url, check_robots=True, extra_delay_sec=extra_delay)
                if resp is None:
                    break
                try:
                    data = resp.json()
                except Exception:
                    break
                products = data.get("products") or []
                if not products:
                    break

                page_items = list(products)
                rng.shuffle(page_items)

                for p in page_items:
                    if col_count >= col_quota or yielded >= limit:
                        break
                    handle = p.get("handle")
                    if not handle:
                        continue
                    product_url = f"{base_url}/products/{handle}"
                    if product_url in seen_urls:
                        continue

                    images = [img.get("src") for img in (p.get("images") or []) if img.get("src")]
                    if not images:
                        continue

                    title = (p.get("title") or "").strip()
                    if not title:
                        continue
                    product_type = (p.get("product_type") or "").strip()
                    tags = p.get("tags") or ""
                    if isinstance(tags, list):
                        tags_str = ", ".join(str(x).strip() for x in tags if str(x).strip())
                    else:
                        tags_str = str(tags).strip()

                    col_readable = "" if is_all else col.replace("-", " ")
                    cat_parts = [x for x in [col_readable, product_type] if x]
                    if not cat_parts and source_name == "watchstore":
                        cat_parts = ["Đồng hồ"]
                    elif not cat_parts and source_name == "matkinhshady":
                        cat_parts = ["Kính mát"]
                    category_raw = " > ".join(cat_parts)

                    color = _extract_color_from_json_product(p, title, tags_str)
                    price = _extract_price_from_json_product(p)
                    gender_hint = _gender_from_collection_slug("" if is_all else col, default_gender)

                    seen_urls.add(product_url)
                    col_count += 1
                    yielded += 1
                    yield {
                        "source": source_name,
                        "brand": brand,
                        "product_url": product_url,
                        "image_url": images[0],
                        "product_name": title,
                        "category_raw": category_raw,
                        "tags_raw": tags_str,
                        "gender_hint": gender_hint,
                        "color_raw": color,
                        "price": price,
                    }

                if len(products) < 10:
                    break
                page += 1
                if page > 120:
                    break

    return _crawl


def crawl_ivymoda(
    client: PoliteHttpClient,
    limit: int,
    seen_urls: Set[str],
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    """
    Duyệt các trang danh mục công khai /danh-muc/<slug>/<page> của IVY moda (24 SP/trang),
    lấy ảnh gốc 1400px từ pubcdn.ivymoda.com, tên sản phẩm, giá, và category_raw từ danh mục.
    """
    brand = "IVY moda"
    # Danh sách danh mục kèm breadcrumb rõ ràng & giới tính mặc định
    categories: List[Tuple[str, str, str, int]] = [
        ("ao-so-mi-nu", "Nữ > Áo > Áo sơ mi nữ", "Women", 20),
        ("ao-thun-nu", "Nữ > Áo > Áo thun nữ", "Women", 20),
        ("ao-vest-nu", "Nữ > Áo khoác > Áo blazer vest nữ", "Women", 15),
        ("ao-khoac-nu-dep", "Nữ > Áo khoác > Áo khoác nữ", "Women", 15),
        ("ao-len-nu", "Nữ > Áo > Áo len nữ", "Women", 12),
        ("ao-gile-nu", "Nữ > Áo > Áo gile nữ", "Women", 8),
        ("ao-phao-nu", "Nữ > Áo khoác > Áo phao nữ", "Women", 8),
        ("dam-nu", "Nữ > Đầm > Đầm nữ", "Women", 30),
        ("dam-maxi", "Nữ > Đầm > Đầm maxi", "Women", 12),
        ("dam-da-hoi-senora-2410", "Nữ > Đầm > Đầm dạ hội", "Women", 10),
        ("ao-dai", "Nữ > Đầm > Áo dài", "Women", 10),
        ("jumpsuit", "Nữ > Đầm > Jumpsuit", "Women", 10),
        ("bst-chan-vay-nu", "Nữ > Chân váy > Chân váy nữ", "Women", 25),
        ("quan-dai-nu", "Nữ > Quần > Quần tây dài nữ", "Women", 20),
        ("quan-jean-nu", "Nữ > Quần > Quần jean nữ", "Women", 12),
        ("quan-lung-short-nu", "Nữ > Quần > Quần short nữ", "Women", 12),
        ("ao-so-mi-nam", "Nam > Áo > Áo sơ mi nam", "Men", 15),
        ("ao-polo-nam", "Nam > Áo > Áo polo nam", "Men", 15),
        ("ao-thun-nam", "Nam > Áo > Áo thun nam", "Men", 15),
        ("ao-khoac-nam", "Nam > Áo khoác > Áo khoác nam", "Men", 12),
        ("ao-len-nam", "Nam > Áo > Áo len nam", "Men", 10),
        ("quan-dai-nam", "Nam > Quần > Quần tây dài nam", "Men", 12),
        ("quan-jean-nam", "Nam > Quần > Quần jean nam", "Men", 10),
        ("quan-lung-nam", "Nam > Quần > Quần short nam", "Men", 10),
        ("hang-moi-ve-tre-em", "Trẻ em > Quần áo trẻ em", "", 5),
        ("nu", "Nữ > Thời trang nữ", "Women", 125),
        ("nam", "Nam > Thời trang nam", "Men", 30),
    ]

    yielded = 0
    for slug, cat_label, gender_hint, max_pages in categories:
        if yielded >= limit:
            break
        for page in range(1, max_pages + 1):
            if yielded >= limit:
                break
            url = f"https://ivymoda.com/danh-muc/{slug}" if page == 1 else f"https://ivymoda.com/danh-muc/{slug}/{page}"
            resp = client.get(url, check_robots=True)
            if resp is None:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.select("div.product, .item-new-prod .product")
            if not cards:
                break

            new_in_page = 0
            for card in cards:
                if yielded >= limit:
                    break
                a_el = card.select_one("h3.title-product a") or card.select_one(".thumb-product a")
                if not a_el or not a_el.get("href"):
                    continue
                purl = a_el["href"].strip()
                if purl.startswith("/"):
                    purl = f"https://ivymoda.com{purl}"
                if purl in seen_urls:
                    continue

                title = a_el.get_text(" ", strip=True)
                img_el = card.select_one(".thumb-product img")
                if not img_el:
                    continue
                if not title:
                    title = (img_el.get("alt") or "").strip()
                if not title:
                    continue

                img_src = (img_el.get("src") or img_el.get("data-original") or "").strip()
                if not img_src or "pubcdn.ivymoda.com" not in img_src:
                    continue
                # Nâng lên bản 1400px sắc nét của CDN IVY moda
                img_hi = img_src.replace("/thumab/400/", "/thumab/1400/")

                price_el = card.select_one(".price-product ins span") or card.select_one(".price-product span")
                price = ""
                if price_el:
                    price = re.sub(r"[^\d]", "", price_el.get_text(strip=True))

                m_col = _COLOR_RE.search(title)
                color_raw = m_col.group(1).strip() if m_col else ""

                seen_urls.add(purl)
                new_in_page += 1
                yielded += 1
                yield {
                    "source": "ivymoda",
                    "brand": brand,
                    "product_url": purl,
                    "image_url": img_hi,
                    "product_name": title,
                    "category_raw": cat_label,
                    "tags_raw": "",
                    "gender_hint": gender_hint,
                    "color_raw": color_raw,
                    "price": price,
                }

            # Kiểm tra xem có trang kế tiếp không
            if new_in_page == 0 and page > 2:
                break


def crawl_juno(
    client: PoliteHttpClient,
    limit: int,
    seen_urls: Set[str],
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    brand = "Juno"
    resp = client.get("https://juno.vn/sitemap/product", check_robots=True)
    if resp is None:
        return
    sub_sitemaps = re.findall(r"<loc>(https?://[^<]+)</loc>", resp.text)
    if not sub_sitemaps:
        sub_sitemaps = ["https://juno.vn/sitemap/product"]

    rng = random.Random(seed)
    rng.shuffle(sub_sitemaps)

    product_urls: List[str] = []
    for sm_url in sub_sitemaps:
        r_sm = client.get(sm_url, check_robots=True)
        if r_sm is None:
            continue
        urls = re.findall(r"<loc>(https?://[^<]+/products/[^<]+)</loc>", r_sm.text)
        product_urls.extend(urls)
        if len(product_urls) >= limit * 2 + 100:
            break

    rng.shuffle(product_urls)
    yielded = 0

    for purl in product_urls:
        if yielded >= limit:
            break
        if purl in seen_urls:
            continue
        r = client.get(purl, check_robots=True)
        if r is None:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        ld_prod: Dict[str, Any] = {}
        for s in soup.find_all("script", type="application/ld+json"):
            if not s.string:
                continue
            try:
                obj = json.loads(s.string)
                if isinstance(obj, dict) and obj.get("@type") == "Product":
                    ld_prod = obj
                    break
            except Exception:
                continue

        title = (ld_prod.get("name") or "").strip()
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(" ", strip=True) if h1 else ""
        if not title:
            continue

        imgs = ld_prod.get("image") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        if not imgs:
            og = soup.find("meta", property="og:image")
            if og and og.get("content"):
                imgs = [og["content"]]
        imgs = [u for u in imgs if isinstance(u, str) and u.startswith("http")]
        if not imgs:
            continue

        bc_parts: List[str] = []
        for el in soup.select("ol li a, .breadcrumb a"):
            txt = el.get_text(" ", strip=True)
            if txt and txt not in ("/", "Trang chủ") and txt not in bc_parts:
                bc_parts.append(txt)

        desc = ld_prod.get("description") or ""
        color = ""
        m_col = re.search(r"Màu sắc\s*:\s*([^\n]+?)(?:\s+Kiểu dáng|\s+Chất liệu|\s+Kích cỡ|$)", desc)
        if m_col:
            color = m_col.group(1).strip()
        m_sty = re.search(r"Kiểu dáng\s*:\s*([^\n]+?)(?:\s+Chất liệu|\s+Kích cỡ|$)", desc)
        if m_sty and m_sty.group(1).strip() not in bc_parts:
            bc_parts.append(m_sty.group(1).strip())
        if not bc_parts and ld_prod.get("category"):
            bc_parts.append(str(ld_prod["category"]))

        price = ""
        offers = ld_prod.get("offers")
        if isinstance(offers, dict) and offers.get("price"):
            price = str(offers["price"])
        elif isinstance(offers, list) and offers and isinstance(offers[0], dict):
            price = str(offers[0].get("price", ""))

        seen_urls.add(purl)
        yielded += 1
        yield {
            "source": "juno",
            "brand": brand,
            "product_url": purl,
            "image_url": imgs[0],
            "product_name": title,
            "category_raw": " > ".join(bc_parts),
            "tags_raw": "",
            "gender_hint": "Women",
            "color_raw": color,
            "price": price,
        }


def crawl_coolmate(
    client: PoliteHttpClient,
    limit: int,
    seen_urls: Set[str],
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    brand = "Coolmate"
    sitemaps = [
        "https://www.coolmate.me/sitemap/product0.xml",
        "https://www.coolmate.me/sitemap/product1.xml",
    ]
    product_urls: List[str] = []
    for sm_url in sitemaps:
        r_sm = client.get(sm_url, check_robots=True)
        if r_sm is None:
            continue
        urls = re.findall(r"<loc>(https?://[^<]+/product/[^<]+)</loc>", r_sm.text)
        product_urls.extend(urls)

    rng = random.Random(seed)
    rng.shuffle(product_urls)
    yielded = 0

    for purl in product_urls:
        if yielded >= limit:
            break
        if purl in seen_urls:
            continue
        r = client.get(purl, check_robots=True)
        if r is None:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else ""
        if not title and soup.title:
            title = soup.title.get_text(strip=True).replace(" - Coolmate", "").strip()
        if not title:
            continue

        imgs: List[str] = []
        og = soup.find("meta", property="og:image")
        if og and og.get("content") and ("coolmate.me" in og["content"]):
            imgs.append(og["content"])
        for img in soup.find_all("img"):
            src = img.get("src") or ""
            if (
                "coolmate.me" in src
                and "handle_cert" not in src
                and "dmca" not in src
                and "logo" not in src.lower()
                and "mceclip" not in src.lower()
                and src not in imgs
            ):
                imgs.append(src)
        if not imgs:
            continue

        unescaped = r.text.replace('\\"', '"')
        bc_parts: List[str] = []
        m_cat = re.search(r'"category":\{"[^}]*"title":"([^"]+)"', unescaped)
        if m_cat:
            bc_parts.append(m_cat.group(1).strip())
        slug = purl.rstrip("/").split("/")[-1].replace("-", " ")
        if slug not in bc_parts:
            bc_parts.append(slug)
        category_raw = " > ".join(bc_parts)

        price = ""
        m_pr = re.search(r'"price":(\d{4,8})', unescaped)
        if m_pr:
            price = m_pr.group(1)

        m_col = _COLOR_RE.search(title)
        color = m_col.group(1).strip() if m_col else ""
        if not color:
            low_u = imgs[0].lower()
            for slug_k, col_v in [
                ("-den", "Đen"), ("-trang", "Trắng"), ("-xam", "Xám"),
                ("-navy", "Xanh Navy"), ("-xanh-den", "Xanh Đen"), ("-be", "Be"),
                ("-nau", "Nâu"), ("-do", "Đỏ"), ("-hong", "Hồng"), ("-reu", "Xanh Rêu"),
            ]:
                if slug_k in low_u:
                    color = col_v
                    break

        seen_urls.add(purl)
        yielded += 1
        yield {
            "source": "coolmate",
            "brand": brand,
            "product_url": purl,
            "image_url": imgs[0],
            "product_name": title,
            "category_raw": category_raw,
            "tags_raw": "",
            "gender_hint": "Men",
            "color_raw": color,
            "price": price,
        }


def crawl_mia(
    client: PoliteHttpClient,
    limit: int,
    seen_urls: Set[str],
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    brand = "MIA.vn"
    # Duyệt danh mục công khai /vali, /balo, /tui-xach, /phu-kien trước (nhanh, 20 SP/request)
    cat_pages = [
        ("vali", "Vali", 10),
        ("balo", "Balo", 10),
        ("tui-xach", "Túi du lịch > Túi xách", 6),
        ("phu-kien", "Phụ kiện du lịch", 8),
    ]
    yielded = 0
    for slug, cat_label, max_p in cat_pages:
        if yielded >= limit:
            break
        for page in range(1, max_p + 1):
            if yielded >= limit:
                break
            url = f"https://mia.vn/{slug}" if page == 1 else f"https://mia.vn/{slug}?page={page}"
            resp = client.get(url, check_robots=True)
            if resp is None:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.select("a.item-product-grid")
            if not cards:
                break
            new_cnt = 0
            for a_el in cards:
                if yielded >= limit:
                    break
                href = (a_el.get("href") or "").strip().replace("&amp;", "&")
                if not href or "/san-pham/" not in href:
                    continue
                full_u = f"https://mia.vn{href}" if href.startswith("/") else href
                parsed = urlparse(full_u)
                # Khóa theo đường dẫn gốc không kèm query size để mỗi mẫu sản phẩm lấy 1 ảnh đại diện
                base_purl = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if base_purl in seen_urls:
                    continue

                img_el = a_el.select_one("img.product-image") or a_el.find("img")
                if not img_el:
                    continue
                img_src = (img_el.get("src") or "").strip()
                if not img_src or "mia.vn/media/uploads" not in img_src:
                    continue
                title = (img_el.get("alt") or "").strip()
                if not title:
                    title_el = a_el.select_one(".product-name, h3, h4")
                    title = title_el.get_text(" ", strip=True) if title_el else ""
                if not title:
                    continue

                price = ""
                pr_el = a_el.select_one(".price-new, .price, [class*=price]")
                if pr_el:
                    price = re.sub(r"[^\d]", "", pr_el.get_text(strip=True))

                m_col = _COLOR_RE.search(f"{title} {href}")
                color = m_col.group(1).strip() if m_col else ""

                seen_urls.add(base_purl)
                new_cnt += 1
                yielded += 1
                yield {
                    "source": "mia",
                    "brand": brand,
                    "product_url": base_purl,
                    "image_url": img_src,
                    "product_name": title,
                    "category_raw": cat_label,
                    "tags_raw": "",
                    "gender_hint": "Unisex",
                    "color_raw": color,
                    "price": price,
                }
            if new_cnt == 0:
                break


SCRAPERS = {
    "yame": make_json_store_scraper("yame"),
    "bitis": make_json_store_scraper("bitis"),
    "aristino": make_json_store_scraper("aristino"),
    "rabity": make_json_store_scraper("rabity"),
    "leonardo": make_json_store_scraper("leonardo"),
    "matkinhshady": make_json_store_scraper("matkinhshady"),
    "watchstore": make_json_store_scraper("watchstore"),
    "ivymoda": crawl_ivymoda,
    "juno": crawl_juno,
    "coolmate": crawl_coolmate,
    "mia": crawl_mia,
}
