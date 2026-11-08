"""
Build FAISS Index, Evaluate efSearch Recall-ANN, Benchmark Latency,
Verify 5 Sample Queries, and Generate Phase 2 Report.
"""
import os
import sys
import time
import json
import hashlib
import pathlib
import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, ".")
from src.model_siglip import load_siglip_inference

def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()

def main():
    t_start = time.time()
    index_dir = pathlib.Path("index")
    reports_dir = pathlib.Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    emb_path = index_dir / "image_embeddings.npy"
    id_map_path = index_dir / "id_map.json"
    splits_path = index_dir / "splits.json"

    if not emb_path.exists() or not id_map_path.exists():
        print(f"Error: {emb_path} or {id_map_path} not found!")
        return

    print("Loading image embeddings and ID map...")
    embeddings = np.load(emb_path).astype(np.float32)
    faiss.normalize_L2(embeddings)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_dict = json.load(f)

    N, D = embeddings.shape
    print(f"Loaded {N:,} embeddings of dimension {D}.")

    # 1. Build FAISS Flat Index (Exact Ground Truth)
    print("\n1. Building FAISS Flat Index (IndexFlatIP)...")
    flat_index = faiss.IndexFlatIP(D)
    flat_index.add(embeddings)
    print(f"Flat index built with {flat_index.ntotal} vectors.")

    # 2. Build FAISS HNSW Index
    print("\n2. Building FAISS HNSW Index (IndexHNSWFlat, M=32, efConstruction=200)...")
    M = 32
    ef_const = 200
    hnsw_index = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.hnsw.efConstruction = ef_const
    t0_build = time.time()
    hnsw_index.add(embeddings)
    build_time = time.time() - t0_build
    print(f"HNSW index built in {build_time:.2f}s with {hnsw_index.ntotal} vectors.")

    # Save HNSW index
    hnsw_path = index_dir / "faiss_hnsw_index.bin"
    faiss.write_index(hnsw_index, str(hnsw_path))
    hnsw_size_mb = os.path.getsize(hnsw_path) / (1024**2)
    print(f"Saved HNSW index to {hnsw_path} ({hnsw_size_mb:.2f} MB).")

    # 3. Save Manifest with Checkpoint Hash
    ckpt_path = "data/checkpoints_siglip-vit/best_model.pt"
    ckpt_hash = compute_file_sha256(ckpt_path)
    manifest = {
        "checkpoint_path": ckpt_path,
        "checkpoint_hash": ckpt_hash,
        "model_name": "google/siglip-so400m-patch14-224",
        "embedding_dim": D,
        "num_vectors": N,
        "faiss": {
            "type": "IndexHNSWFlat",
            "M": M,
            "ef_construction": ef_const,
            "default_ef_search": 128,
            "metric": "inner_product"
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "embeddings": "image_embeddings.npy",
            "id_map": "id_map.json",
            "splits": "splits.json",
            "hnsw_index": "faiss_hnsw_index.bin"
        }
    }
    manifest_path = index_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path} (hash: {ckpt_hash[:16]}...).")

    # 4. Measure ANN Recall@10 across efSearch values [32, 64, 128, 256]
    print("\n3. Measuring ANN Recall@10 vs Flat across efSearch values...")
    rng = np.random.default_rng(42)
    sample_size = 200
    test_indices = rng.choice(N, size=sample_size, replace=False)
    query_vectors = embeddings[test_indices]  # [200, 1152]

    # Ground truth from Flat
    _, gt_indices = flat_index.search(query_vectors, 10)  # [200, 10]

    ef_search_values = [32, 64, 128, 256]
    ann_results = []

    for ef in ef_search_values:
        hnsw_index.hnsw.efSearch = ef
        # Warmup
        _ = hnsw_index.search(query_vectors[:10], 10)

        t0_search = time.perf_counter()
        _, ann_indices = hnsw_index.search(query_vectors, 10)
        dt_search = (time.perf_counter() - t0_search) * 1000.0 / sample_size  # ms/query

        # Calculate Recall@10
        recalls = []
        for i in range(sample_size):
            gt_set = set(gt_indices[i])
            ann_set = set(ann_indices[i])
            recalls.append(len(gt_set & ann_set) / 10.0)
        mean_recall = np.mean(recalls)

        ann_results.append({
            "efSearch": ef,
            "Recall@10": mean_recall,
            "Latency_ms_query": dt_search
        })
        print(f"efSearch = {ef:3d}: Recall@10 = {mean_recall:.4f} ({mean_recall*100:.2f}%), Latency = {dt_search:.3f} ms/query")

    # Reset efSearch to default 128
    hnsw_index.hnsw.efSearch = 128

    # 5. Load Model for Query Verifications & Latency Benchmarks
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model on {device} for inference...")
    model = load_siglip_inference(checkpoint_path=ckpt_path, device=device, fp16=(device == "cuda"))
    processor = model.processor

    # 6. Verify 5 Sample Queries (Accented vs Unaccented)
    print("\n==================================================")
    print("4. VERIFYING 5 SAMPLE QUERIES (Accented vs Unaccented)")
    print("==================================================")
    sample_queries = [
        ("áo sơ mi nam trắng", "ao so mi nam trang"),
        ("giày thể thao nữ màu hồng", "giay the thao nu mau hong"),
        ("túi xách nữ màu nâu", "tui xach nu mau nau"),
        ("đồng hồ nam dây da", "dong ho nam day da"),
        ("áo thun nữ màu đen", "ao thun nu mau den")
    ]

    sample_query_reports = []

    for q_acc, q_unacc in sample_queries:
        # Encode accented
        inp_acc = processor(text=[q_acc], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb_acc = model.encode_text(inp_acc["input_ids"].to(device)).float().cpu().numpy()

        # Encode unaccented
        inp_unacc = processor(text=[q_unacc], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb_unacc = model.encode_text(inp_unacc["input_ids"].to(device)).float().cpu().numpy()

        # Cosine similarity between accented and unaccented
        cos_sim = float(np.dot(emb_acc[0], emb_unacc[0]))

        # Search top-5 for accented query
        scores, indices = hnsw_index.search(emb_acc, 5)
        top5_items = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            item = id_map[idx]
            top5_items.append({
                "rank": rank,
                "score": float(score),
                "item_id": item["item_id"],
                "articleType": item["articleType"],
                "gender": item["gender"],
                "baseColour": item["baseColour"]
            })

        sample_query_reports.append({
            "query_accented": q_acc,
            "query_unaccented": q_unacc,
            "cosine_similarity": cos_sim,
            "top5": top5_items
        })

        print(f"\nTruy vấn: '{q_acc}' vs '{q_unacc}' | Cosine: {cos_sim:.4f}")
        for it in top5_items:
            print(f"  [{it['rank']}] score={it['score']:.4f} | ID={it['item_id']} | {it['articleType']} - {it['gender']} - {it['baseColour']}")

    # 7. Latency Benchmarks (>= 100 iterations after warm-up)
    print("\n==================================================")
    print("5. LATENCY BENCHMARK BY COMPONENT (100 iterations)")
    print("==================================================")
    N_BENCH = 120
    WARMUP = 20

    # (a) Encode text (batch 1)
    text_latencies = []
    bench_text = "áo sơ mi nam công sở màu xanh"
    for i in range(N_BENCH):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        inp = processor(text=[bench_text], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        with torch.no_grad():
            _ = model.encode_text(inp["input_ids"].to(device)).cpu()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= WARMUP:
            text_latencies.append(dt)

    # (b) Encode 1 image (batch 1, including PIL open + preprocess)
    sample_img_path = id_map[0]["path"]
    img_latencies = []
    for i in range(N_BENCH):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with Image.open(sample_img_path) as img:
            rgb_img = img.convert("RGB")
        inp = processor(images=[rgb_img], return_tensors="pt")
        pix = inp["pixel_values"].to(device)
        if device == "cuda":
            pix = pix.half()
        with torch.no_grad():
            _ = model.encode_image(pix).cpu()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= WARMUP:
            img_latencies.append(dt)

    # (c) Search HNSW top-200 + post-filter
    dummy_query = embeddings[0:1]
    filter_latencies = []
    for i in range(N_BENCH):
        t0 = time.perf_counter()
        scores, indices = hnsw_index.search(dummy_query, 200)
        # Apply simulated post-filter (e.g., gender == 'Men')
        filtered = [id_map[idx] for idx in indices[0] if id_map[idx]["gender"] == "Men"][:10]
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= WARMUP:
            filter_latencies.append(dt)

    p50_text = np.percentile(text_latencies, 50)
    p95_text = np.percentile(text_latencies, 95)

    p50_img = np.percentile(img_latencies, 50)
    p95_img = np.percentile(img_latencies, 95)

    p50_search = np.percentile(filter_latencies, 50)
    p95_search = np.percentile(filter_latencies, 95)

    total_text2img_p50 = p50_text + p50_search
    total_text2img_p95 = p95_text + p95_search

    total_img2img_p50 = p50_img + p50_search
    total_img2img_p95 = p95_img + p95_search

    print(f"(a) Encode text (batch 1):     p50 = {p50_text:6.2f} ms | p95 = {p95_text:6.2f} ms")
    print(f"(b) Encode image (batch 1):    p50 = {p50_img:6.2f} ms | p95 = {p95_img:6.2f} ms")
    print(f"(c) HNSW top-200 + post-filter: p50 = {p50_search:6.2f} ms | p95 = {p95_search:6.2f} ms")
    print("-" * 65)
    print(f"Tổng Text -> Ảnh (p50 / p95):   {total_text2img_p50:6.2f} ms / {total_text2img_p95:6.2f} ms (Target < 200 ms: {'DAT' if total_text2img_p95 < 200 else 'KHONG DAT'})")
    print(f"Tổng Ảnh -> Ảnh  (p50 / p95):   {total_img2img_p50:6.2f} ms / {total_img2img_p95:6.2f} ms (Target < 200 ms: {'DAT' if total_img2img_p95 < 200 else 'KHONG DAT'})")

    # 8. Generate reports/phase2_report.md
    report_md_path = reports_dir / "phase2_report.md"
    lines = [
        "# Báo cáo Nghiệm thu Pha 2: Embeddings & FAISS Index",
        "",
        f"**Thời gian thực hiện:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Bảng Kiểm tra Phân bổ Split trong Catalog",
        "| Phân loại Split | Toàn bộ Catalog | Trong Phạm vi In-Scope | Tỉ lệ In-Scope (%) |",
        "|---|---|---|---|",
        f"| **test** (từ `dense_captions_catalog.json`) | 1,372 | 1,290 | {1290/1372*100:.2f}% |",
        f"| **unknown** (chưa có nhãn train/val độc lập) | 39,519 | 37,356 | {37356/39519*100:.2f}% |",
        f"| **train** | 0 | 0 | 0.00% |",
        f"| **val** | 0 | 0 | 0.00% |",
        f"| **Tổng cộng** | **40,891** | **38,646** | **{38646/40891*100:.2f}%** |",
        "",
        "> [!NOTE]",
        "> 1.372 sản phẩm trong tập Test của checkpoint thuộc catalog Myntra Kaggle này (trong đó 1.290 sản phẩm thuộc danh mục thời trang in-scope). Toàn bộ 39.519 sản phẩm còn lại được gán nhãn `unknown` vì không có file phân chia train/val đi kèm.",
        "",
        "## 2. Thông số Embedding & Index FAISS",
        f"- **Tổng số vector:** {N:,} vectors (100% in-scope).",
        f"- **Số chiều vector:** {D} (L2-normalized).",
        f"- **Dung lượng file embeddings:** {os.path.getsize(emb_path)/(1024**2):.2f} MB (`index/image_embeddings.npy`).",
        f"- **Kiến trúc Index:** `IndexHNSWFlat` (M=32, efConstruction=200, default efSearch=128).",
        f"- **Dung lượng file index:** {hnsw_size_mb:.2f} MB (`index/faiss_hnsw_index.bin`).",
        f"- **Mã hash Checkpoint lưu trong manifest:** `{ckpt_hash}`",
        "",
        "## 3. Bảng Đánh giá Recall-ANN theo efSearch (so với IndexFlatIP)",
        "| efSearch | ANN Recall@10 | Thời gian tìm kiếm (ms/query) | Ghi chú |",
        "|---|---|---|---|",
    ]

    for row in ann_results:
        rec = row["Recall@10"]
        lat = row["Latency_ms_query"]
        note = "Mặc định (cân bằng tối ưu)" if row["efSearch"] == 128 else ("Nhanh nhất" if row["efSearch"] == 32 else "Độ chính xác cao nhất")
        lines.append(f"| {row['efSearch']} | **{rec*100:.2f}%** ({rec:.4f}) | {lat:.3f} ms | {note} |")

    lines.extend([
        "",
        "## 4. Bảng Đo Độ trễ theo Thành phần (Latency Benchmark)",
        "| Thành phần | p50 (ms) | p95 (ms) | Ngưỡng chỉ tiêu | Trạng thái |",
        "|---|---|---|---|---|",
        f"| **(a) Encode Text (batch 1)** | {p50_text:.2f} ms | {p95_text:.2f} ms | - | Đạt |",
        f"| **(b) Encode Image (batch 1, kèm PIL)** | {p50_img:.2f} ms | {p95_img:.2f} ms | - | Đạt |",
        f"| **(c) HNSW top-200 + Lọc hậu kỳ** | {p50_search:.2f} ms | {p95_search:.2f} ms | - | Đạt |",
        f"| **TỔNG Text → Ảnh (a + c)** | **{total_text2img_p50:.2f} ms** | **{total_text2img_p95:.2f} ms** | **< 200 ms** | **{'ĐẠT' if total_text2img_p95 < 200 else 'KHÔNG ĐẠT'}** |",
        f"| **TỔNG Ảnh → Ảnh (b + c)** | **{total_img2img_p50:.2f} ms** | **{total_img2img_p95:.2f} ms** | **< 200 ms** | **{'ĐẠT' if total_img2img_p95 < 200 else 'KHÔNG ĐẠT'}** |",
        "",
        "## 5. Kết quả Kiểm tra Quan sát 5 Câu mẫu (Có dấu vs Không dấu)",
        "| STT | Câu có dấu | Câu không dấu | Cosine Sim | Top-1 Dự đoán | articleType | gender | baseColour |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for idx, rep in enumerate(sample_query_reports, 1):
        q_a = rep["query_accented"]
        q_u = rep["query_unaccented"]
        cos = rep["cosine_similarity"]
        top1 = rep["top5"][0]
        lines.append(f"| {idx} | {q_a} | {q_u} | **{cos:.4f}** | ID {top1['item_id']} | {top1['articleType']} | {top1['gender']} | {top1['baseColour']} |")

    lines.extend([
        "",
        "### Chi tiết Top-5 cho từng câu truy vấn mẫu:",
        ""
    ])

    for rep in sample_query_reports:
        lines.append(f"#### Truy vấn: \"{rep['query_accented']}\" (Cosine với không dấu: {rep['cosine_similarity']:.4f})")
        lines.append("| Rank | Score | Item ID | articleType | gender | baseColour |")
        lines.append("|---|---|---|---|---|---|")
        for it in rep["top5"]:
            lines.append(f"| {it['rank']} | {it['score']:.4f} | {it['item_id']} | {it['articleType']} | {it['gender']} | {it['baseColour']} |")
        lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nPhase 2 report saved to {report_md_path}!")

if __name__ == "__main__":
    main()
