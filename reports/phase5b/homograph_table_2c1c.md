# Bảng Đồng âm Không dấu (Homograph Table) — Bước 2c.1c

- **Nguồn:** Hệ thống âm tiết tiếng Việt chuẩn (thanh điệu 6 thanh) và từ điển mở tiếng Việt Hồ Ngọc Đức (`Viet74K` / âm tiết hợp lệ tiếng Việt). **Không** trích từ DEV, Tập A/B-synth/C hay B-human.
- **Mục đích:** Liệt kê các âm tiết tiếng Việt phổ biến có **cùng dạng bỏ dấu** với từ khóa trong lexicon `parser_v1.py`, giải thích nguyên nhân dương tính giả khi parser bỏ dấu toàn bộ trước khi khớp.

## 1. Từ khóa Màu sắc (`COLOUR_LEXICON`)

| Dạng bỏ dấu (v1) | Nhãn màu | Nghĩa màu (có dấu) | Âm tiết tiếng Việt khác cùng dạng bỏ dấu (ví dụ) |
|---|---|---|---|
| `tim` | `Purple` | tím | **tìm** (tìm mua, cần tìm), **tim** (trái tim, màu tim), **tỉm** (tủm tỉm) |
| `do` | `Red` | đỏ | **đồ** (đồ ngủ, đựng đồ, đồ chơi), **độ** (độ bền, nhiệt độ), **đỗ** (đỗ xe), **do** (do đó), **dở**, **đỡ** |
| `cam` | `Orange` | cam | **cầm** (cầm tay), **cảm** (cảm giác), **cấm**, **cám**, **căm** |
| `be` | `Beige` | be (màu be) | **bé** (bé trai, bé gái, nhỏ bé), **bẻ**, **bề** (bề mặt), **bế**, **bể**, **bệ**, **búp bê** (`be`) |
| `den` | `Black` | đen | **đến** (đi đến), **đèn**, **đền**, **đến nỗi** |
| `trang` | `White` | trắng | **trang** (thời trang, trang phục, trang điểm), **trạng**, **trăng**, **trảng** |
| `hong` | `Pink` | hồng | **hỏng** (bị hỏng), **hóng**, **họng**, **hông** (đeo hông) |
| `vang` | `Yellow` | vàng | **vang** (vang dội), **vắng** (vắng mặt), **văng**, **vạng** |
| `bac` | `Silver` | bạc | **bác**, **bắc**, **bặc**, **bấc** |
| `xam` | `Grey` | xám | **xâm** (xâm nhập), **xàm**, **xẩm** |
| `nau` | `Brown` | nâu | **náu**, **nấu** |
| `xanh` | `Blue` | xanh | **xảnh** |
| `kem` | `Off White` | kem | **kém** (kém chất lượng), **kèm** (đi kèm), **kẽm** |
| `ghi` | `Grey` | ghi | **ghi** (ghi chép, ghi nhớ), **ghì** |

## 2. Từ khóa Giới tính (`GENDER_LEXICON`)

| Dạng bỏ dấu (v1) | Nhãn giới tính | Nghĩa đúng | Âm tiết tiếng Việt khác cùng dạng bỏ dấu |
|---|---|---|---|
| `nam` | `Men` | nam | **năm** (5 năm, hàng năm), **nằm**, **nắm**, **nấm** |
| `nu` | `Women` | nữ | **nụ** (nụ hoa), **nu** |
| `be trai` / `be` | `Boys` | bé trai | `be` trùng `bé`, `bề`, `bể` (nhưng cụm `be trai` ít đồng âm hơn đơn âm `be`) |
| `be gai` / `gai` | `Girls` | bé gái | `gai` trùng **gái**, **gãi**, **gài**, **gai** (gai góc) |

## 3. Từ khóa Loại sản phẩm (`ARTICLE_TYPE_LEXICON`, đơn âm & cụm ngắn)

| Dạng bỏ dấu (v1) | Nhãn sản phẩm | Nghĩa đúng | Âm tiết tiếng Việt khác cùng dạng bỏ dấu |
|---|---|---|---|
| `dam` | `Dresses` | đầm | **đậm** (đậm màu, xanh đậm), **đám** (đám cưới), **đâm**, **dám**, **dầm**, **dăm** |
| `vay` | `Dresses` | váy | **vậy** (như vậy), **vay**, **vẫy**, **vấy**, **vảy** |
| `mu` | `Caps` | mũ | **mù**, **mụ**, **mu**, **mủ** |
| `non` | `Caps` | nón | **non** (màu xanh non, non nớt), **nôn**, **nồn** |
| `vi` | `Wallets` | ví | **vì** (bởi vì), **vị** (vị trí), **vi**, **vĩ** |
| `bop` | `Wallets` | bóp | **bóp** (động từ bóp), **bọp**, **bốp** |
| `tat` | `Socks` | tất | **tất** (tất cả, tất nhiên), **tát**, **tắt**, **tật** |
| `vo` | `Socks` | vớ | **vợ**, **vỏ**, **vỡ**, **vò**, **vô** (vô cùng) |
| `ao` | `Tshirts`/prefix | áo | **ảo**, **ao** (ao hồ), **ào** |
| `quan` | `Jeans`/prefix | quần | **quán**, **quân**, **quấn** (váy quấn), **quận**, **quan** |
| `giay` | `Shoes`/prefix | giày | **giấy** (giấy tờ), **giãy**, **giày** (dày dặn dễ gõ nhầm) |
| `dep` | `Sandals`/`Flip Flops` | dép | **đẹp** (rất đẹp, đẹp mắt), **dẹp** |
| `tui` | `Handbags`/prefix | túi | **tủi**, **tụi**, **tùi** |
| `kinh` | `Sunglasses` | kính | **kinh** (kinh điển, kinh doanh), **kính** (kính trọng) |
| `da` | (trong `ao da`, `vi da`) | da | **dạ** (dạ hội, áo dạ), **đá** (đá bóng, đính đá), **đã** (đã qua), **đa** (đa năng) |
