# Hướng dẫn Chạy Ứng dụng Streamlit (Pha 6)

> *Ghi chú:* Đây là tài liệu hướng dẫn nhanh cho phần khởi chạy ứng dụng (Pha 6). Tài liệu đầy đủ toàn diện của toàn bộ dự án sẽ được tổng hợp ở Pha 7 theo đề cương.

---

## 1. Lệnh Khởi chạy Nhanh

Khởi chạy ứng dụng Streamlit cục bộ tại máy chủ nội bộ:

```bash
# Đứng tại thư mục gốc của dự án:
streamlit run app/main.py --server.port 8501 --server.address 127.0.0.1
```

Mở trình duyệt: `http://127.0.0.1:8501`

### Chạy như app Windows (cửa sổ riêng, không cần trình duyệt)

Cần thêm `pywebview` (đã thử với 5.4) và Microsoft Edge WebView2 Runtime (có sẵn trên Windows 10/11):

```powershell
pip install pywebview==5.4
# Tạo shortcut "Tra cứu thời trang" trên Desktop và Start Menu (chạy một lần):
powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1
```

- Bấm đúp shortcut để mở app. Server Streamlit chạy ẩn trên một cổng trống của `127.0.0.1`
  và tự tắt khi đóng cửa sổ. Mỗi lúc chỉ chạy được một phiên.
- Không có shortcut, hoặc cần xem lỗi trong console: chạy `Chay_ung_dung.bat` ở thư mục gốc.
- Log nằm ở `%LOCALAPPDATA%\FashionSearch\logs\` (`desktop.log` của trình khởi chạy, `streamlit.log` của server).
- Đổi icon: sửa `scripts/make_app_icon.py`, chạy lại nó, rồi chạy lại script tạo shortcut.

---

## 2. Các Luồng Nghiệp vụ Chính

1. **Luồng 1 (Text → Ảnh):** Hỗ trợ truy vấn tự nhiên tiếng Việt có dấu và không dấu (ví dụ: `áo sơ mi nam màu xanh`, `ao so mi nam mau xanh`, `váy hoa nữ`, `giay the thao nam`).
2. **Luồng 2 (Ảnh → Ảnh):** Tải ảnh trực tiếp lên bộ nhớ RAM để tìm sản phẩm tương đồng; hỗ trợ kiểm tra Sanity Check tự truy xuất (Self-Retrieval) đạt Rank 1 với Cosine ≈ 1.0000.
3. **Luồng 3 (CIR Catalog):** Hòa trộn vector sản phẩm catalog có sẵn với câu chỉnh sửa văn bản (Modifier).
4. **Luồng 4 (CIR Upload):** Hòa trộn ảnh người dùng tải lên với câu chỉnh sửa văn bản.

---

## 3. Cấu hình Mặc định & Lựa chọn Thanh bên

- **Gallery mặc định:** `test_gallery` ($N=1,290$ sản phẩm test split chuẩn). Lựa chọn phụ `reference_gallery` ($N=38,646$) mang nhãn *ƯỚC LƯỢNG LẠC QUAN* do chứa ảnh tập train.
- **Cấu hình Text → Ảnh:**
  - `cfg_v5_10` (Mặc định trên giao diện): Phân loại 3 tầng + chặn lệch lĩnh vực (mở rộng sau Pha 6, chưa chạy trên tập đóng băng `held-out v2`).
  - `cfg_v5_11` (DEV SET, chọn thủ công): cfg_v5_10 + (1) khôi phục dấu trước khi encode, (2) parser v3 (`src/serve/parser_v3.py`: từ vựng bổ sung, khớp lại trên chuỗi bỏ dấu, khớp mờ sai 1 ký tự, "ví cầm tay" → Clutches), (3) bổ sung 30 ứng viên cùng loại theo cosine khi truy vấn nêu rõ loại (loại hiếm như jumpsuit), (4) trọng số giới tính ×3 khi truy vấn nêu nam/nữ. Mọi thay đổi chỉ ở bước suy luận; checkpoint và các cấu hình cũ giữ nguyên. Từ vựng/trọng số rút ra sau khi xem lỗi trên tập dev nên chỉ là số DEV, chưa chạy trên held-out. Đo thật bằng app (`reports/dev_v6/run3_all_queries`, reference_gallery, lạc quan): R@10 hand 0.970, val_a 1.000, val_b 0.898, val_h 0.978; p95 ≤ 81 ms.
  - `cir_v1_07` (DEV SET, chọn thủ công): dùng vector ảnh tham chiếu (bỏ qua combiner chưa huấn luyện hiệu quả) + giữ loại/màu/giới tính gốc, đổi theo modifier (giới tính tường minh ×5), loại ảnh gốc khỏi kết quả, parser v3. Offline trên cir_eval_val (val_gallery): hit@10 0.975 so với 0.775 của cir_v1_06; ground truth CIR dựa trên thuộc tính nên có tính vòng tròn.
  - `cfg_v5_09` (Chính thức Pha 5c): Cấu hình chiến thắng ($\gamma=1.0$, tier_soft bonus).
  - `cfg_v5_01` (An toàn): Cấu hình an toàn baseline ($\gamma=1.5$).
  - `model_only` (Tham chiếu): Model thuần SigLIP ($\gamma=0.0$, cosine thuần).
- **Cấu hình CIR:**
  - `cir_v1_05` (Mặc định): Hệ thống CIR (+ C3 Attribute Bonus).
  - `cir_v1_01` (Tham chiếu): Baseline bộ nối thuần tiếng Anh.
- **Bảo mật Hash Guard (Luật 6):** Tự động đối chiếu SHA-256 của checkpoint trong `config.yaml` (`e2f82fee...`) với `manifest.json`. Lệch sẽ từ chối nạp và hiển thị thông báo lỗi.
