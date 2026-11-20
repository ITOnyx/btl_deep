"""
Tiện ích dùng chung cho src.crawl (Pha 2–6):
- PoliteHttpClient: tuân thủ robots.txt, <= 1 req/s/domain, <= 2 kết nối/domain, backoff 429/5xx,
  ghi từng request (ts, domain, url, status, elapsed_ms, bytes, event) vào reports/crawl_v2/crawl_log.csv.
- Tải ảnh gốc LƯU NGUYÊN BYTE (Luật L4: không crop, không sửa, không nén lại ảnh gốc).
- Tính SHA-1 trên byte gốc, pHash 64-bit, width, height, blur_score (Laplacian variance).
- Quản lý quarantine: chuyển file bị loại vào data/crawl_v2/_quarantine/<lý_do>/ và ghi log CSV (Luật L2).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

# Dọn biến môi trường TLS bị lỗi trên Windows (PostgreSQL ca-bundle.crt không tồn tại)
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

import cv2
import imagehash
import numpy as np
import requests
import yaml
from PIL import Image


SCHEMA_COLUMNS: List[str] = [
    "crawl_id",
    "brand",
    "product_url",
    "image_url",
    "local_path",
    "product_name",
    "category_raw",
    "articleType",
    "gender",
    "baseColour",
    "price",
    "label_source",
    "label_status",
    "crawled_at",
    "http_status",
    "sha1",
    "phash",
    "width",
    "height",
    "filter_flag",
]


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    p = Path(config_path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_run_manifest(cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Path:
    c2 = cfg.get("crawl_v2", {})
    out_path = Path(c2.get("paths", {}).get("run_manifest_json", "reports/crawl_v2/run_manifest.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    import PIL
    manifest = {
        "project": "Hoc_sau_ver4 - Vietnamese Fashion Out-of-Domain Crawl v2",
        "timestamp_utc": now_iso(),
        "git_head": git_head(),
        "seed": int(c2.get("seed", 42)),
        "user_agent": c2.get("user_agent", ""),
        "python_version": sys.version,
        "platform": platform.platform(),
        "libraries": {
            "requests": requests.__version__,
            "Pillow": PIL.__version__,
            "opencv-python": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "imagehash": getattr(imagehash, "__version__", "4.3.1"),
            "PyYAML": yaml.__version__,
        },
        "crawl_v2_config": c2,
    }
    if extra:
        manifest.update(extra)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def append_jsonl(path: str | Path, record: Dict[str, Any], lock: Optional[threading.Lock] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    if lock:
        with lock:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
    else:
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    items: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


class QuarantineLogger:
    """Quản lý chuyển file bị loại vào data/crawl_v2/_quarantine/<lý_do>/ và ghi log CSV (Luật L2)."""

    def __init__(self, quarantine_dir: str | Path, log_csv_path: str | Path):
        self.quarantine_dir = Path(quarantine_dir)
        self.log_csv_path = Path(log_csv_path)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.log_csv_path.exists():
            with open(self.log_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "crawl_id",
                    "brand",
                    "original_path",
                    "quarantine_path",
                    "reason",
                    "metric_detail",
                    "sha1",
                    "phash",
                    "timestamp_utc",
                ])

    def quarantine_bytes(
        self,
        raw_bytes: bytes,
        ext: str,
        crawl_id: str,
        brand: str,
        reason: str,
        metric_detail: str = "",
        sha1_str: str = "",
        phash_str: str = "",
    ) -> str:
        """Lưu nguyên byte tệp bị loại ngay lúc tải vào thư mục quarantine/<reason>/."""
        reason_dir = self.quarantine_dir / reason
        reason_dir.mkdir(parents=True, exist_ok=True)
        clean_ext = ext if ext.startswith(".") else f".{ext}"
        dst_path = reason_dir / f"{crawl_id}{clean_ext}"
        with open(dst_path, "wb") as f:
            f.write(raw_bytes)
        if not sha1_str and raw_bytes:
            sha1_str = hashlib.sha1(raw_bytes).hexdigest()
        rel_dst = dst_path.as_posix()
        with self._lock:
            with open(self.log_csv_path, "a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([
                    crawl_id,
                    brand,
                    "",
                    rel_dst,
                    reason,
                    metric_detail,
                    sha1_str,
                    phash_str,
                    now_iso(),
                ])
        return rel_dst

    def quarantine_existing_file(
        self,
        src_path: str | Path,
        crawl_id: str,
        brand: str,
        reason: str,
        metric_detail: str = "",
        sha1_str: str = "",
        phash_str: str = "",
    ) -> str:
        """Di chuyển tệp đã có trên đĩa vào quarantine/<reason>/ (không xóa vĩnh viễn)."""
        src_p = Path(src_path)
        reason_dir = self.quarantine_dir / reason
        reason_dir.mkdir(parents=True, exist_ok=True)
        dst_p = reason_dir / src_p.name
        if dst_p.exists():
            dst_p = reason_dir / f"{crawl_id}_{src_p.name}"
        if src_p.exists():
            shutil.move(str(src_p), str(dst_p))
        rel_src = src_p.as_posix()
        rel_dst = dst_p.as_posix()
        with self._lock:
            with open(self.log_csv_path, "a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([
                    crawl_id,
                    brand,
                    rel_src,
                    rel_dst,
                    reason,
                    metric_detail,
                    sha1_str,
                    phash_str,
                    now_iso(),
                ])
        return rel_dst


class PoliteHttpClient:
    """
    HTTP client lịch sự:
    - Kiểm tra robots.txt (và lưu bản chụp vào reports/crawl_v2/robots/ nếu chưa có)
    - Tốc độ tối đa 1 request/giây/domain, đồng thời <= 2 kết nối/domain
    - Ghi log mọi request vào reports/crawl_v2/crawl_log.csv
    - Backoff lũy thừa khi gặp 429/5xx; dừng domain nếu gặp captcha/403 liên tiếp.
    """

    def __init__(self, crawl_cfg: Dict[str, Any]):
        self.user_agent = crawl_cfg.get(
            "user_agent",
            "AcademicResearchBot/2.0 (Hoc_sau_ver4 Vietnamese Fashion Visual Search; non-commercial academic evaluation; contact: nguyen-student@university.edu.vn)",
        )
        max_rps = float(crawl_cfg.get("max_req_per_sec_per_domain", 1.0))
        self.min_interval = 1.0 / max(0.1, max_rps)
        self.max_concurrent_per_domain = min(2, int(crawl_cfg.get("max_concurrent_per_domain", 2)))
        self.timeout = float(crawl_cfg.get("timeout_sec", 15))
        self.max_retries = int(crawl_cfg.get("max_retries", 3))
        self.backoff_factor = float(crawl_cfg.get("backoff_factor", 1.5))

        paths = crawl_cfg.get("paths", {})
        self.log_csv_path = Path(paths.get("crawl_log_csv", "reports/crawl_v2/crawl_log.csv"))
        self.robots_dir = Path(paths.get("robots_dir", "reports/crawl_v2/robots"))
        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.robots_dir.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._domain_rate_locks: Dict[str, threading.Lock] = {}
        self._domain_semaphores: Dict[str, threading.Semaphore] = {}
        self._domain_last_ts: Dict[str, float] = {}
        self._robots_cache: Dict[str, Optional[RobotFileParser]] = {}
        self._domain_halted: Dict[str, str] = {}
        self._domain_403_streak: Dict[str, int] = {}
        self._meta_lock = threading.Lock()
        self._log_lock = threading.Lock()

        if not self.log_csv_path.exists():
            with open(self.log_csv_path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp_utc",
                    "domain",
                    "url",
                    "http_status",
                    "elapsed_ms",
                    "bytes",
                    "event",
                ])

    def _get_session(self) -> requests.Session:
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update({
                "User-Agent": self.user_agent,
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            })
            self._local.session = sess
        return sess

    def _log_request(
        self,
        domain: str,
        url: str,
        http_status: int,
        elapsed_ms: float,
        n_bytes: int,
        event: str,
    ) -> None:
        with self._log_lock:
            with open(self.log_csv_path, "a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([
                    now_iso(),
                    domain,
                    url,
                    http_status,
                    round(elapsed_ms, 1),
                    n_bytes,
                    event,
                ])

    def _get_domain_sync(self, domain: str) -> Tuple[threading.Lock, threading.Semaphore]:
        with self._meta_lock:
            if domain not in self._domain_rate_locks:
                self._domain_rate_locks[domain] = threading.Lock()
                self._domain_semaphores[domain] = threading.Semaphore(self.max_concurrent_per_domain)
            return self._domain_rate_locks[domain], self._domain_semaphores[domain]

    def is_domain_halted(self, domain: str) -> Optional[str]:
        with self._meta_lock:
            return self._domain_halted.get(domain)

    def halt_domain(self, domain: str, reason: str) -> None:
        with self._meta_lock:
            self._domain_halted[domain] = reason

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        domain = parsed.netloc
        with self._meta_lock:
            rp = self._robots_cache.get(base, "UNSET")  # type: ignore[arg-type]
        if rp == "UNSET":
            rp_obj = RobotFileParser()
            robots_url = f"{base}/robots.txt"
            sess = self._get_session()
            t0 = time.time()
            try:
                resp = sess.get(robots_url, timeout=10)
                dt_ms = (time.time() - t0) * 1000.0
                self._log_request(domain, robots_url, resp.status_code, dt_ms, len(resp.content), "robots_fetch")
                if resp.status_code == 200:
                    rp_obj.parse(resp.text.splitlines())
                    snap_file = self.robots_dir / f"{domain.replace(':', '_')}_robots.txt"
                    if not snap_file.exists():
                        header = (
                            f"# Fetched from: {robots_url}\n"
                            f"# Timestamp (UTC): {now_iso()}\n"
                            f"# HTTP Status: {resp.status_code}\n"
                            f"# User-Agent: {self.user_agent}\n\n"
                        )
                        snap_file.write_text(header + resp.text, encoding="utf-8")
                elif resp.status_code in (401, 403):
                    rp_obj.parse(["User-agent: *", "Disallow: /"])
                else:
                    rp_obj.allow_all = True
            except Exception:
                rp_obj.allow_all = True
            with self._meta_lock:
                self._robots_cache[base] = rp_obj
            rp = rp_obj
        if rp is None:
            return True
        return rp.can_fetch(self.user_agent, url) and rp.can_fetch("*", url)

    def _respect_rate_limit(self, domain: str, rate_lock: threading.Lock) -> None:
        with rate_lock:
            last = self._domain_last_ts.get(domain, 0.0)
            elapsed = time.time() - last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._domain_last_ts[domain] = time.time()

    def get(
        self,
        url: str,
        check_robots: bool = True,
        expect_binary: bool = False,
        extra_delay_sec: float = 0.0,
    ) -> Optional[requests.Response]:
        parsed = urlparse(url)
        domain = parsed.netloc
        if self.is_domain_halted(domain):
            return None

        if check_robots and not self.can_fetch(url):
            self._log_request(domain, url, 0, 0.0, 0, "robots_disallowed")
            return None

        rate_lock, sem = self._get_domain_sync(domain)
        sess = self._get_session()

        for attempt in range(1, self.max_retries + 1):
            if self.is_domain_halted(domain):
                return None
            with sem:
                self._respect_rate_limit(domain, rate_lock)
                if extra_delay_sec > 0:
                    time.sleep(extra_delay_sec)
                t0 = time.time()
                try:
                    resp = sess.get(url, timeout=self.timeout, allow_redirects=True)
                    dt_ms = (time.time() - t0) * 1000.0
                    status = resp.status_code
                    n_bytes = len(resp.content) if resp.content else 0

                    if status == 200:
                        with self._meta_lock:
                            self._domain_403_streak[domain] = 0
                        if not expect_binary:
                            head = resp.text[:4000].lower()
                            if any(
                                tok in head
                                for tok in (
                                    "cf-challenge",
                                    "just a moment...",
                                    "verify you are human",
                                    "g-recaptcha",
                                    "hcaptcha",
                                )
                            ):
                                self._log_request(domain, url, status, dt_ms, n_bytes, "blocked_captcha")
                                self.halt_domain(domain, "captcha_or_cf_challenge")
                                return None
                        self._log_request(domain, url, status, dt_ms, n_bytes, "ok")
                        return resp

                    if status in (401, 403):
                        self._log_request(domain, url, status, dt_ms, n_bytes, f"blocked_http_{status}")
                        with self._meta_lock:
                            streak = self._domain_403_streak.get(domain, 0) + 1
                            self._domain_403_streak[domain] = streak
                            if streak >= 5:
                                self._domain_halted[domain] = f"consecutive_http_{status}"
                        return None

                    if status == 404:
                        self._log_request(domain, url, status, dt_ms, n_bytes, "not_found_404")
                        return None

                    if status in (429, 500, 502, 503, 504):
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            wait = min(float(retry_after), 10.0)
                        else:
                            base_wait = 3.0 if status == 429 else 1.5
                            wait = base_wait * (self.backoff_factor ** attempt)
                        self._log_request(domain, url, status, dt_ms, n_bytes, f"retry_{status}_attempt_{attempt}")
                        time.sleep(wait)
                        continue

                    self._log_request(domain, url, status, dt_ms, n_bytes, f"unexpected_http_{status}")
                    return None
                except requests.RequestException as exc:
                    dt_ms = (time.time() - t0) * 1000.0
                    self._log_request(domain, url, 0, dt_ms, 0, f"exception_{type(exc).__name__}_attempt_{attempt}")
                    time.sleep(self.backoff_factor ** attempt)
        return None


REJECT_URL_KEYWORDS = [
    "size-chart",
    "sizechart",
    "bang-size",
    "bang_size",
    "huong-dan-chon-size",
    "banner",
    "promo",
    "khuyen-mai",
    "voucher",
    "logo",
    "icon",
    "placeholder",
    "no-image",
    "avatar",
    "review",
]


def is_rejected_image_url(url: str) -> bool:
    low = (url or "").lower()
    return any(kw in low for kw in REJECT_URL_KEYWORDS)


def detect_banner_or_placeholder(rgb_img: Image.Image) -> Tuple[bool, str]:
    """Kiểm tra tỷ lệ khung hình cực đoan, ảnh trắng/rỗng, hoặc dải banner chữ khuyến mãi lớn."""
    w, h = rgb_img.size
    aspect = w / max(1, h)
    if aspect < 0.35 or aspect > 2.8:
        return True, f"extreme_aspect_{aspect:.2f}"

    arr = np.array(rgb_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if float(np.std(gray)) < 6.0:
        return True, "near_blank_placeholder"

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    top_h = max(1, int(h * 0.12))
    bot_h = max(1, int(h * 0.12))
    edges = cv2.Canny(gray, 100, 200)

    top_sat = float(np.mean(hsv[:top_h, :, 1]))
    bot_sat = float(np.mean(hsv[-bot_h:, :, 1]))
    top_edge = float(np.mean(edges[:top_h, :] > 0))
    bot_edge = float(np.mean(edges[-bot_h:, :] > 0))

    if (top_sat > 140.0 and top_edge > 0.12) or (bot_sat > 140.0 and bot_edge > 0.12):
        return True, "promo_banner_overlay"

    return False, "ok"


def _ext_from_format_or_url(pil_fmt: str, image_url: str) -> str:
    fmt = (pil_fmt or "").upper()
    if fmt in ("JPEG", "JPG"):
        return ".jpg"
    if fmt == "PNG":
        return ".png"
    if fmt == "WEBP":
        return ".webp"
    path_low = urlparse(image_url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if path_low.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def download_and_inspect_raw_image(
    client: PoliteHttpClient,
    quarantine: QuarantineLogger,
    image_url: str,
    source_name: str,
    brand: str,
    crawl_id: str,
    images_root: Path,
    min_short_side: int = 224,
    blur_threshold: float = 12.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Tải ảnh từ CDN, LƯU NGUYÊN BYTE GỐC (Luật L4).
    Nếu ảnh không qua kiểm tra chất lượng (corrupt, low_res < 224, blurry < 12.0, banner),
    lưu nguyên byte vào data/crawl_v2/_quarantine/<lý_do>/ và ghi log CSV (Luật L2).
    """
    if not image_url or is_rejected_image_url(image_url):
        return False, "banner_url_keyword", {"http_status": 0}

    resp = client.get(image_url, check_robots=False, expect_binary=True)
    if resp is None or not resp.content:
        return False, "download_failed", {"http_status": 0}

    raw_bytes = resp.content
    http_status = int(resp.status_code)
    sha1_str = hashlib.sha1(raw_bytes).hexdigest()

    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            fmt = (im.format or "").upper()
            if fmt not in ("JPEG", "JPG", "PNG", "WEBP"):
                q_path = quarantine.quarantine_bytes(
                    raw_bytes, ".bin", crawl_id, brand, "corrupt", f"unsupported_format_{fmt}", sha1_str, ""
                )
                return False, "corrupt", {"http_status": http_status, "sha1": sha1_str, "local_path": q_path}
            rgb = im.convert("RGB")
            ext = _ext_from_format_or_url(fmt, image_url)
    except Exception as exc:
        q_path = quarantine.quarantine_bytes(
            raw_bytes, ".bin", crawl_id, brand, "corrupt", f"decode_error_{type(exc).__name__}", sha1_str, ""
        )
        return False, "corrupt", {"http_status": http_status, "sha1": sha1_str, "local_path": q_path}

    w, h = rgb.size
    short_side = min(w, h)
    phash_str = str(imagehash.phash(rgb.convert("L")))
    gray_arr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray_arr, cv2.CV_32F).var())

    base_metrics = {
        "http_status": http_status,
        "sha1": sha1_str,
        "phash": phash_str,
        "width": int(w),
        "height": int(h),
        "short_side": int(short_side),
        "blur_score": round(blur_score, 4),
        "bytes": len(raw_bytes),
    }

    if short_side < min_short_side:
        q_path = quarantine.quarantine_bytes(
            raw_bytes, ext, crawl_id, brand, "low_res", f"short_side={short_side}<{min_short_side}", sha1_str, phash_str
        )
        base_metrics["local_path"] = q_path
        return False, "low_res", base_metrics

    is_banner, banner_reason = detect_banner_or_placeholder(rgb)
    if is_banner:
        q_path = quarantine.quarantine_bytes(
            raw_bytes, ext, crawl_id, brand, "banner", banner_reason, sha1_str, phash_str
        )
        base_metrics["local_path"] = q_path
        return False, "banner", base_metrics

    if blur_score < blur_threshold:
        q_path = quarantine.quarantine_bytes(
            raw_bytes, ext, crawl_id, brand, "blurry", f"laplacian={blur_score:.2f}<{blur_threshold}", sha1_str, phash_str
        )
        base_metrics["local_path"] = q_path
        return False, "blurry", base_metrics

    # Lưu NGUYÊN BYTE GỐC vào data/crawl_v2/images/<source>/<crawl_id>.<ext>
    src_dir = images_root / source_name
    src_dir.mkdir(parents=True, exist_ok=True)
    save_path = src_dir / f"{crawl_id}{ext}"
    with open(save_path, "wb") as f:
        f.write(raw_bytes)

    base_metrics["local_path"] = save_path.as_posix()
    return True, "pass", base_metrics
