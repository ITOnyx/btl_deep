# Báo cáo Nghiệm thu Pha 2: Embeddings & FAISS Index

**Thời gian thực hiện:** 2026-11-08 22:38:23

## 1. Bảng Kiểm tra Phân bổ Split trong Catalog
| Phân loại Split | Toàn bộ Catalog | Trong Phạm vi In-Scope | Tỉ lệ In-Scope (%) |
|---|---|---|---|
| **test** (từ `dense_captions_catalog.json`) | 1,372 | 1,290 | 94.02% |
| **unknown** (chưa có nhãn train/val độc lập) | 39,519 | 37,356 | 94.53% |
| **train** | 0 | 0 | 0.00% |
| **val** | 0 | 0 | 0.00% |
| **Tổng cộng** | **40,891** | **38,646** | **94.51%** |

> [!NOTE]
> 1.372 sản phẩm trong tập Test của checkpoint thuộc catalog Myntra Kaggle này (trong đó 1.290 sản phẩm thuộc danh mục thời trang in-scope). Toàn bộ 39.519 sản phẩm còn lại được gán nhãn `unknown` vì không có file phân chia train/val đi kèm.

## 2. Thông số Embedding & Index FAISS
- **Tổng số vector:** 38,646 vectors (100% in-scope).
- **Số chiều vector:** 1152 (L2-normalized).
- **Dung lượng file embeddings:** 169.83 MB (`index/image_embeddings.npy`).
- **Kiến trúc Index:** `IndexHNSWFlat` (M=32, efConstruction=200, default efSearch=128).
- **Dung lượng file index:** 179.86 MB (`index/faiss_hnsw_index.bin`).
- **Mã hash Checkpoint lưu trong manifest:** `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`

## 3. Bảng Đánh giá Recall-ANN theo efSearch (so với IndexFlatIP)
| efSearch | ANN Recall@10 | Thời gian tìm kiếm (ms/query) | Ghi chú |
|---|---|---|---|
| 32 | **99.35%** (0.9935) | 0.070 ms | Nhanh nhất |
| 64 | **99.95%** (0.9995) | 0.069 ms | Độ chính xác cao nhất |
| 128 | **99.95%** (0.9995) | 0.113 ms | Mặc định (cân bằng tối ưu) |
| 256 | **99.95%** (0.9995) | 0.166 ms | Độ chính xác cao nhất |

## 4. Bảng Đo Độ trễ theo Thành phần (Latency Benchmark)
| Thành phần | p50 (ms) | p95 (ms) | Ngưỡng chỉ tiêu | Trạng thái |
|---|---|---|---|---|
| **(a) Encode Text (batch 1)** | 22.14 ms | 37.61 ms | - | Đạt |
| **(b) Encode Image (batch 1, kèm PIL)** | 82.98 ms | 130.68 ms | - | Đạt |
| **(c) HNSW top-200 + Lọc hậu kỳ** | 0.25 ms | 0.29 ms | - | Đạt |
| **TỔNG Text → Ảnh (a + c)** | **22.39 ms** | **37.90 ms** | **< 200 ms** | **ĐẠT** |
| **TỔNG Ảnh → Ảnh (b + c)** | **83.23 ms** | **130.98 ms** | **< 200 ms** | **ĐẠT** |

## 5. Kết quả Kiểm tra Quan sát 5 Câu mẫu (Có dấu vs Không dấu)
| STT | Câu có dấu | Câu không dấu | Cosine Sim | Top-1 Dự đoán | articleType | gender | baseColour |
|---|---|---|---|---|---|---|---|
| 1 | áo sơ mi nam trắng | ao so mi nam trang | **0.2798** | ID 33233 | Shirts | Men | White |
| 2 | giày thể thao nữ màu hồng | giay the thao nu mau hong | **0.5043** | ID 40483 | Sports Shoes | Women | Pink |
| 3 | túi xách nữ màu nâu | tui xach nu mau nau | **0.4079** | ID 38470 | Handbags | Women | Brown |
| 4 | đồng hồ nam dây da | dong ho nam day da | **0.2990** | ID 46318 | Watches | Men | Brown |
| 5 | áo thun nữ màu đen | ao thun nu mau den | **0.4123** | ID 45959 | Tops | Women | Black |

### Chi tiết Top-5 cho từng câu truy vấn mẫu:

#### Truy vấn: "áo sơ mi nam trắng" (Cosine với không dấu: 0.2798)
| Rank | Score | Item ID | articleType | gender | baseColour |
|---|---|---|---|---|---|
| 1 | 0.7618 | 33233 | Shirts | Men | White |
| 2 | 0.7570 | 14750 | Shirts | Men | White |
| 3 | 0.7475 | 54252 | Shirts | Men | White |
| 4 | 0.7427 | 23036 | Shirts | Men | White |
| 5 | 0.7317 | 19193 | Shirts | Men | White |

#### Truy vấn: "giày thể thao nữ màu hồng" (Cosine với không dấu: 0.5043)
| Rank | Score | Item ID | articleType | gender | baseColour |
|---|---|---|---|---|---|
| 1 | 0.7960 | 40483 | Sports Shoes | Women | Pink |
| 2 | 0.7925 | 4855 | Casual Shoes | Unisex | Pink |
| 3 | 0.7678 | 36418 | Sports Shoes | Women | Pink |
| 4 | 0.7619 | 36331 | Casual Shoes | Women | Pink |
| 5 | 0.7444 | 15270 | Casual Shoes | Women | Pink |

#### Truy vấn: "túi xách nữ màu nâu" (Cosine với không dấu: 0.4079)
| Rank | Score | Item ID | articleType | gender | baseColour |
|---|---|---|---|---|---|
| 1 | 0.7619 | 38470 | Handbags | Women | Brown |
| 2 | 0.7562 | 33404 | Handbags | Women | Brown |
| 3 | 0.7519 | 20490 | Handbags | Women | Brown |
| 4 | 0.7509 | 53070 | Handbags | Women | Brown |
| 5 | 0.7481 | 57261 | Handbags | Women | Brown |

#### Truy vấn: "đồng hồ nam dây da" (Cosine với không dấu: 0.2990)
| Rank | Score | Item ID | articleType | gender | baseColour |
|---|---|---|---|---|---|
| 1 | 0.7820 | 46318 | Watches | Men | Brown |
| 2 | 0.7777 | 36893 | Watches | Men | Brown |
| 3 | 0.7728 | 51553 | Watches | Men | Brown |
| 4 | 0.7643 | 29447 | Watches | Men | Bronze |
| 5 | 0.7510 | 54630 | Watches | Men | Coffee Brown |

#### Truy vấn: "áo thun nữ màu đen" (Cosine với không dấu: 0.4123)
| Rank | Score | Item ID | articleType | gender | baseColour |
|---|---|---|---|---|---|
| 1 | 0.7814 | 45959 | Tops | Women | Black |
| 2 | 0.7747 | 53123 | Tshirts | Women | Black |
| 3 | 0.7654 | 31297 | Tshirts | Women | Black |
| 4 | 0.7547 | 1989 | Tshirts | Women | Black |
| 5 | 0.7470 | 32450 | Tshirts | Women | Black |
