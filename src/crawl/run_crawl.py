"""
Script chạy Crawl v2 (Pha 2 Pilot 200 ảnh & Pha 3 Crawl đầy đủ ~13,000 ảnh thô -> >=10,000 ảnh sạch).
- Hỗ trợ resume từ data/crawl_v2/raw_crawled.jsonl.
- Chạy song song giữa các domain (mỗi domain bị khóa chặt <= 1 req/s/domain và <= 2 kết nối/domain).
- Ghi từng request vào reports/crawl_v2/crawl_log.csv.
- Kiểm tra dung lượng đĩa trống (>= 5 GB), tỷ lệ lỗi tải, captcha/403.
- Sau mỗi nguồn: đếm số ảnh thật trên đĩa và đối chiếu với bản ghi pass.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .common import (
    PoliteHttpClient,
    QuarantineLogger,
    append_jsonl,
    download_and_inspect_raw_image,
    load_config,
    now_iso,
    read_jsonl,
    write_run_manifest,
)
from .contact_sheet import render_contact_sheet
from .scrapers import SCRAPERS
from .taxonomy_mapper import TaxonomyMapper


PILOT_QUOTAS: Dict[str, int] = {
    "ivymoda": 22,
    "aristino": 20,
    "bitis": 20,
    "yame": 18,
    "matkinhshady": 18,
    "watchstore": 18,
    "juno": 18,
    "coolmate": 18,
    "rabity": 18,
    "mia": 15,
    "leonardo": 15,
}


def check_free_disk_gb(drive: str = "D:/") -> float:
    return shutil.disk_usage(drive).free / 1e9


def crawl_one_source(
    source_name: str,
    target_pass: int,
    cfg: Dict[str, Any],
    client: PoliteHttpClient,
    quarantine: QuarantineLogger,
    mapper: TaxonomyMapper,
    existing_records: List[Dict[str, Any]],
    state_lock: threading.Lock,
    id_counter: List[int],
) -> Dict[str, Any]:
    c2 = cfg["crawl_v2"]
    paths = c2["paths"]
    q_cfg = c2["quality_filter"]
    raw_state_file = Path(paths["raw_state_file"])
    images_root = Path(paths["images_dir"])
    seed = int(c2.get("seed", 42))

    seen_urls: Set[str] = {
        r["product_url"]
        for r in existing_records
        if r.get("source") == source_name and r.get("product_url")
    }
    already_pass = sum(
        1
        for r in existing_records
        if r.get("source") == source_name and r.get("filter_flag") == "pass"
    )

    stats: Dict[str, Any] = {
        "source": source_name,
        "target_pass": target_pass,
        "already_pass": already_pass,
        "new_pass": 0,
        "dropped_out_of_scope": 0,
        "dropped_quality": 0,
        "download_errors": 0,
        "drop_reasons": {},
        "bytes_downloaded": 0,
        "started_ts": time.time(),
        "elapsed_sec": 0.0,
        "halted_reason": "",
    }

    remaining = max(0, target_pass - already_pass)
    if remaining == 0:
        print(f"[{source_name}] Đã đủ {already_pass}/{target_pass} ảnh pass (resume skip).")
        return stats

    scraper_fn = SCRAPERS[source_name]
    candidate_budget = int(remaining * 1.45) + 30
    print(f"[{source_name}] Bắt đầu crawl (cần thêm {remaining}, đã có {already_pass}/{target_pass})...")

    for cand in scraper_fn(client=client, limit=candidate_budget, seen_urls=seen_urls, seed=seed):
        if stats["new_pass"] >= remaining:
            break

        # Kiểm tra dung lượng đĩa trống < 5 GB
        if (stats["new_pass"] + stats["dropped_quality"]) % 25 == 0:
            free_gb = check_free_disk_gb("D:/")
            if free_gb < 5.0:
                stats["halted_reason"] = f"low_disk_space_{free_gb:.2f}GB"
                print(f"[{source_name}] DỪNG KHẨN CẤP: Đĩa trống < 5GB ({free_gb:.2f} GB)!")
                break

        crawl_ts = now_iso()
        cat_raw = cand.get("category_raw", "")
        pname = cand.get("product_name", "")
        brand = cand.get("brand", source_name)

        with state_lock:
            id_counter[0] += 1
            crawl_id = f"cr2_{id_counter[0]:06d}"

        if mapper.is_out_of_scope(cat_raw, pname):
            stats["dropped_out_of_scope"] += 1
            stats["drop_reasons"]["out_of_scope"] = stats["drop_reasons"].get("out_of_scope", 0) + 1
            rec = {
                "crawl_id": crawl_id,
                "source": source_name,
                "brand": brand,
                "product_url": cand["product_url"],
                "image_url": cand["image_url"],
                "local_path": "",
                "product_name": pname,
                "category_raw": cat_raw,
                "articleType": "out_of_scope",
                "gender": "",
                "baseColour": "",
                "price": cand.get("price", ""),
                "label_source": "breadcrumb",
                "label_status": "needs_review",
                "crawled_at": crawl_ts,
                "http_status": 200,
                "sha1": "",
                "phash": "",
                "width": 0,
                "height": 0,
                "blur_score": 0.0,
                "filter_flag": "out_of_scope",
            }
            append_jsonl(raw_state_file, rec, lock=state_lock)
            continue

        ok, flag, img_metrics = download_and_inspect_raw_image(
            client=client,
            quarantine=quarantine,
            image_url=cand["image_url"],
            source_name=source_name,
            brand=brand,
            crawl_id=crawl_id,
            images_root=images_root,
            min_short_side=int(q_cfg.get("min_short_side", 224)),
            blur_threshold=float(q_cfg.get("blur_laplacian_threshold", 12.0)),
        )

        if not ok:
            if flag == "download_failed":
                stats["download_errors"] += 1
            else:
                stats["dropped_quality"] += 1
            stats["drop_reasons"][flag] = stats["drop_reasons"].get(flag, 0) + 1
            rec = {
                "crawl_id": crawl_id,
                "source": source_name,
                "brand": brand,
                "product_url": cand["product_url"],
                "image_url": cand["image_url"],
                "local_path": img_metrics.get("local_path", ""),
                "product_name": pname,
                "category_raw": cat_raw,
                "articleType": "",
                "gender": "",
                "baseColour": "",
                "price": cand.get("price", ""),
                "label_source": "",
                "label_status": "needs_review",
                "crawled_at": crawl_ts,
                "http_status": img_metrics.get("http_status", 0),
                "sha1": img_metrics.get("sha1", ""),
                "phash": img_metrics.get("phash", ""),
                "width": img_metrics.get("width", 0),
                "height": img_metrics.get("height", 0),
                "blur_score": img_metrics.get("blur_score", 0.0),
                "filter_flag": flag,
            }
            append_jsonl(raw_state_file, rec, lock=state_lock)

            # Kiểm tra tỷ lệ lỗi tải > 10% (sau tối thiểu 30 lần thử)
            total_attempts = stats["new_pass"] + stats["dropped_quality"] + stats["download_errors"]
            if total_attempts >= 30 and (stats["download_errors"] / total_attempts) > 0.10:
                stats["halted_reason"] = f"error_rate_exceeded_{stats['download_errors']}/{total_attempts}"
                print(f"[{source_name}] DỪNG NGUỒN: Tỷ lệ lỗi tải > 10% ({stats['download_errors']}/{total_attempts})!")
                break
            continue

        stats["bytes_downloaded"] += int(img_metrics.get("bytes", 0))
        article_type, label_source = mapper.map_category(cat_raw, pname)
        gender = mapper.map_gender(
            category_raw=cat_raw,
            title_raw=pname,
            tags_str=cand.get("tags_raw", ""),
            site_default_gender=cand.get("gender_hint", ""),
        )
        base_colour = mapper.map_color(cand.get("color_raw", ""), pname)

        if article_type and gender:
            label_status = "ok"
        else:
            label_status = "needs_review"

        if label_source not in ("breadcrumb", "title", "manual"):
            label_source = "breadcrumb" if cat_raw else "title"

        rec = {
            "crawl_id": crawl_id,
            "source": source_name,
            "brand": brand,
            "product_url": cand["product_url"],
            "image_url": cand["image_url"],
            "local_path": img_metrics["local_path"],
            "product_name": pname,
            "category_raw": cat_raw,
            "articleType": article_type if article_type else "needs_review",
            "gender": gender if gender else "needs_review",
            "baseColour": base_colour,
            "price": cand.get("price", ""),
            "label_source": label_source,
            "label_status": label_status,
            "crawled_at": crawl_ts,
            "http_status": img_metrics["http_status"],
            "sha1": img_metrics["sha1"],
            "phash": img_metrics["phash"],
            "width": img_metrics["width"],
            "height": img_metrics["height"],
            "blur_score": img_metrics["blur_score"],
            "filter_flag": "pass",
        }
        append_jsonl(raw_state_file, rec, lock=state_lock)
        stats["new_pass"] += 1

        if stats["new_pass"] % 25 == 0 or stats["new_pass"] == remaining:
            print(
                f"  [{source_name}] Tiến độ: {already_pass + stats['new_pass']}/{target_pass} "
                f"(loại_chất_lượng={stats['dropped_quality']}, ngoài_phạm_vi={stats['dropped_out_of_scope']}, lỗi_tải={stats['download_errors']})"
            )

    stats["elapsed_sec"] = round(time.time() - stats["started_ts"], 2)
    src_img_dir = images_root / source_name
    disk_files = [f for f in src_img_dir.glob("*") if f.is_file()] if src_img_dir.exists() else []
    stats["disk_image_count"] = len(disk_files)
    return stats


def write_phase2_pilot_report(
    cfg: Dict[str, Any],
    all_stats: List[Dict[str, Any]],
    total_elapsed: float,
) -> None:
    c2 = cfg["crawl_v2"]
    paths = c2["paths"]
    rep_dir = Path(paths["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)

    all_recs = read_jsonl(paths["raw_state_file"])
    pass_recs = [r for r in all_recs if r.get("filter_flag") == "pass" and Path(r.get("local_path", "")).exists()]

    total_attempts = len(all_recs)
    n_pass = len(pass_recs)
    n_oos = sum(1 for r in all_recs if r.get("filter_flag") == "out_of_scope")
    n_err = sum(1 for r in all_recs if r.get("filter_flag") == "download_failed")
    n_qdrop = total_attempts - n_pass - n_oos - n_err
    n_needs_rev = sum(1 for r in pass_recs if r.get("label_status") == "needs_review")

    total_bytes = sum(Path(r["local_path"]).stat().st_size for r in pass_recs)
    avg_kb = (total_bytes / max(1, n_pass)) / 1024.0
    img_per_sec = n_pass / max(0.1, total_elapsed)

    # Xuất contact sheet 50 ảnh phân tầng theo nguồn (seed 42)
    rng = random.Random(int(c2.get("seed", 42)))
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for r in pass_recs:
        by_src.setdefault(r["source"], []).append(r)
    sample_50: List[Dict[str, Any]] = []
    for sname in sorted(by_src.keys()):
        items = list(by_src[sname])
        rng.shuffle(items)
        sample_50.extend(items[:4])
    remaining_pool = [r for r in pass_recs if r not in sample_50]
    rng.shuffle(remaining_pool)
    sample_50 = (sample_50 + remaining_pool)[:50]

    cs_path = render_contact_sheet(
        sample_50,
        rep_dir / "phase2_pilot_contact_sheet_50.png",
        cols=5,
        title="Phase 2 Pilot Contact Sheet (50 Stratified Samples, Seed 42)",
    )

    est_full_raw = int(c2["quotas"]["full_raw_target"])
    est_time_min = (est_full_raw / max(0.1, img_per_sec)) / 60.0
    est_disk_gb = (est_full_raw * avg_kb) / (1024.0 * 1024.0)

    lines = [
        "# BÁO CÁO PHA 2 — PILOT 200 ẢNH TRẢI ĐỀU 11 NGUỒN",
        "",
        f"- **Thời điểm hoàn tất (UTC):** `{now_iso()}`",
        f"- **Seed cố định:** `{c2.get('seed', 42)}`",
        f"- **Contact sheet 50 ảnh:** `{cs_path.as_posix()}`",
        "",
        "## 1. Số liệu thực tế đo trên đĩa (Pilot)",
        "",
        "| Chỉ số | Giá trị đo thật | Ghi chú |",
        "|---|---:|---|",
        f"| Tổng ứng viên đi qua pipeline | **{total_attempts}** | 11 nguồn thương hiệu VN |",
        f"| Số ảnh hợp lệ (`filter_flag == pass`) trên đĩa | **{n_pass}** | Lưu nguyên byte gốc (Luật L4) |",
        f"| Tốc độ thu thập thực tế (song song 11 domain) | **{img_per_sec:.2f} ảnh/giây** | Tổng thời gian tường: `{total_elapsed:.1f}s` |",
        f"| Tỷ lệ lỗi tải HTTP (`download_failed`) | **{n_err / max(1, total_attempts) * 100:.2f}%** ({n_err}/{total_attempts}) | Ngưỡng dừng an toàn: < 10% |",
        f"| Tỷ lệ loại ngoài phạm vi (`out_of_scope`) | **{n_oos / max(1, total_attempts) * 100:.2f}%** ({n_oos}/{total_attempts}) | Mỹ phẩm/quà tặng/lens |",
        f"| Tỷ lệ loại chất lượng ảnh (`low_res`/`blurry`/`banner`) | **{n_qdrop / max(1, total_attempts) * 100:.2f}%** ({n_qdrop}/{total_attempts}) | Đã chuyển vào `_quarantine/` |",
        f"| Tỷ lệ nhãn `ok` / `needs_review` | **{(n_pass - n_needs_rev) / max(1, n_pass) * 100:.1f}% `ok`** / **{n_needs_rev / max(1, n_pass) * 100:.1f}% `needs_review`** | `{n_pass - n_needs_rev}` ok, `{n_needs_rev}` needs_review |",
        f"| Dung lượng trung bình 1 ảnh gốc | **{avg_kb:.1f} KB** | Tổng `{total_bytes / (1024*1024):.2f} MB` cho `{n_pass}` ảnh |",
        f"| Ước tính thời gian cho {est_full_raw:,} ảnh (Pha 3) | **~{est_time_min:.1f} phút** | Chạy song song 11 domain, <=1 req/s/domain |",
        f"| Ước tính dung lượng đĩa cho {est_full_raw:,} ảnh gốc | **~{est_disk_gb:.2f} GB** | Ổ `D:` đang trống `{check_free_disk_gb('D:/'):.2f} GB` |",
        "",
        "## 2. Thống kê theo từng nguồn trong Pilot",
        "",
        "| Nguồn (`source`) | Mục tiêu Pilot | Đạt `pass` | Đếm file trên đĩa | Lệch đĩa | Ngoài phạm vi | Loại chất lượng | Lỗi tải | Thời gian (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for st in sorted(all_stats, key=lambda x: x["source"]):
        tot_p = st["already_pass"] + st["new_pass"]
        disk_c = st.get("disk_image_count", tot_p)
        diff = disk_c - tot_p
        lines.append(
            f"| `{st['source']}` | {st['target_pass']} | {tot_p} | {disk_c} | {diff} | "
            f"{st['dropped_out_of_scope']} | {st['dropped_quality']} | {st['download_errors']} | {st['elapsed_sec']:.1f} |"
        )

    (rep_dir / "phase2_pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Phase 2] Đã xuất báo cáo Pilot tại {rep_dir / 'phase2_pilot_report.md'} và contact sheet tại {cs_path}")


def write_phase3_crawl_report(
    cfg: Dict[str, Any],
    all_stats: List[Dict[str, Any]],
    total_elapsed: float,
) -> None:
    c2 = cfg["crawl_v2"]
    paths = c2["paths"]
    rep_dir = Path(paths["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)

    all_recs = read_jsonl(paths["raw_state_file"])
    pass_recs = [r for r in all_recs if r.get("filter_flag") == "pass" and Path(r.get("local_path", "")).exists()]
    total_bytes = sum(Path(r["local_path"]).stat().st_size for r in pass_recs)
    flag_counts = Counter(r.get("filter_flag", "unknown") for r in all_recs)

    lines = [
        "# BÁO CÁO PHA 3 — CRAWL ĐẦY ĐỦ & ĐỐI CHIẾU TỆP TRÊN ĐĨA",
        "",
        f"- **Thời điểm hoàn tất (UTC):** `{now_iso()}`",
        f"- **Tổng thời gian chạy Pha 3:** `{total_elapsed:.1f} giây` (`{total_elapsed/60.0:.1f} phút`)",
        f"- **Tổng bản ghi trong `raw_crawled.jsonl`:** **{len(all_recs):,}**",
        f"- **Tổng ảnh hợp lệ (`filter_flag == pass`) tồn tại trên đĩa:** **{len(pass_recs):,}**",
        f"- **Tổng dung lượng ảnh gốc trên đĩa:** **{total_bytes / (1024**3):.3f} GB** (`{total_bytes:,} bytes`)",
        f"- **Dung lượng ổ `D:` còn trống:** **{check_free_disk_gb('D:/'):.2f} GB**",
        f"- **Phân bố `filter_flag`:** `{dict(flag_counts)}`",
        "",
        "## Đối chiếu số ảnh thật trên đĩa với log theo từng nguồn",
        "",
        "| Nguồn (`source`) | Hạn ngạch (`target`) | Số bản ghi `pass` | Số file thật trên đĩa | Lệch (Đĩa - Log) | Ngoài phạm vi | Loại chất lượng | Lỗi tải | Trạng thái dừng |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for st in sorted(all_stats, key=lambda x: x["source"]):
        tot_p = st["already_pass"] + st["new_pass"]
        disk_c = st.get("disk_image_count", tot_p)
        diff = disk_c - tot_p
        halt = st.get("halted_reason") or "completed_ok"
        lines.append(
            f"| `{st['source']}` | {st['target_pass']} | {tot_p} | {disk_c} | {diff} | "
            f"{st['dropped_out_of_scope']} | {st['dropped_quality']} | {st['download_errors']} | `{halt}` |"
        )

    (rep_dir / "phase3_full_crawl_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Phase 3] Đã xuất báo cáo Crawl đầy đủ tại {rep_dir / 'phase3_full_crawl_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl v2 runner (pilot 200 or full >=10,000)")
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn tới config.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot", help="pilot (200 ảnh) hoặc full (13,000 ảnh)")
    parser.add_argument("--sources", nargs="*", default=None, help="Chỉ chạy các nguồn chỉ định")
    args = parser.parse_args()

    cfg = load_config(args.config)
    c2 = cfg["crawl_v2"]
    paths = c2["paths"]

    Path(paths["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["images_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["quarantine_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["reports_dir"]).mkdir(parents=True, exist_ok=True)

    write_run_manifest(cfg, extra={"last_mode": args.mode})

    client = PoliteHttpClient(c2)
    quarantine = QuarantineLogger(paths["quarantine_dir"], paths["quarantine_log_csv"])
    mapper = TaxonomyMapper()

    existing_records = read_jsonl(paths["raw_state_file"])
    max_num = 0
    for r in existing_records:
        cid = str(r.get("crawl_id", ""))
        if cid.startswith("cr2_") and cid[4:].isdigit():
            max_num = max(max_num, int(cid[4:]))
    id_counter = [max_num]
    state_lock = threading.Lock()

    quotas_map = PILOT_QUOTAS if args.mode == "pilot" else c2["quotas"]["per_source"]
    active_sources = list(quotas_map.keys())
    if args.sources:
        active_sources = [s for s in args.sources if s in quotas_map]

    max_workers = min(len(active_sources), int(c2.get("max_domain_workers", 11)))
    print(f"=== CRAWL V2 [{args.mode.upper()}] | Sources={len(active_sources)} | Workers={max_workers} ===")

    t0 = time.time()
    all_stats: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                crawl_one_source,
                sname,
                int(quotas_map[sname]),
                cfg,
                client,
                quarantine,
                mapper,
                existing_records,
                state_lock,
                id_counter,
            ): sname
            for sname in active_sources
        }
        for fut in as_completed(futures):
            sname = futures[fut]
            try:
                st = fut.result()
                all_stats.append(st)
            except Exception as exc:
                print(f"[{sname}] LỖI NGOẠI LỆ: {type(exc).__name__}: {exc}")

    total_elapsed = time.time() - t0
    if args.mode == "pilot":
        write_phase2_pilot_report(cfg, all_stats, total_elapsed)
    else:
        write_phase3_crawl_report(cfg, all_stats, total_elapsed)


if __name__ == "__main__":
    main()
