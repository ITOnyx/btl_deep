# Changelog Sửa Tập A và Tập B-synth (Bước 2c.4)

## 1. Tập A (`eval/val_queries_a_synthetic.json`)

- `val_a_142`: `áo dài truyền thống sari mau xanh la cho nu` -> `sari an do mau xanh la cho nu` (bỏ cụm 'áo dài' gây nhầm văn hóa)
- `val_a_162`: `áo dài truyền thống sari cho nữ` -> `sari Ấn Độ cho nữ` (bỏ cụm 'áo dài' gây nhầm văn hóa)
- `val_a_209`: `áo dài truyền thống sari cho nu` -> `sari an do cho nu` (bỏ cụm 'áo dài' gây nhầm văn hóa)
- `val_a_229`: `áo dài truyền thống sari nu mau xanh la` -> `sari an do nu mau xanh la` (bỏ cụm 'áo dài' gây nhầm văn hóa)

Tổng số câu sửa ở Tập A: **4** | SHA-256 mới: `4fd30c2028a2b8c63ea2fac0d0147297e4d11f8b1eb40ee4ff2f1f8a6cf4f2b4`

## 2. Tập B-synth (`eval/val_queries_b_synth.json`) — Kiểm toán nhãn đa nghĩa (2c.4a)

- `val_b_synth_007` (`cho mình hỏi ba lô laptop unisex mẫu mới nhất`): `articleType` `['Backpacks']` -> `['Backpacks', 'Laptop Bag']` (Đa nghĩa Backpacks/Laptop Bag (khớp DEV h066/h067); `n_relevant=29`)
- `val_b_synth_008` (`giày mọi chị em đen tuyền giá tốt`): `articleType` `['Flats']` -> `['Flats', 'Casual Shoes']` (Đa nghĩa Flats/Casual Shoes (giày lười đế bệt); `n_relevant=13`)
- `val_b_synth_017` (`tim mua giay sneaker thuong cua nam xanh bien gia tot`): `articleType` `['Casual Shoes']` -> `['Casual Shoes', 'Sports Shoes']` (Đa nghĩa Casual/Sports Shoes (khớp DEV h048); `n_relevant=11`)
- `val_b_synth_099` (`giay sneaker thuong nau dat gia tot`): `articleType` `['Casual Shoes']` -> `['Casual Shoes', 'Sports Shoes']` (Đa nghĩa Casual/Sports Shoes (khớp DEV h048); `n_relevant=14`)
- `val_b_synth_100` (`shop ơi có áo ấm đen tuyền đi làm`): `articleType` `['Jackets']` -> `['Jackets', 'Sweaters', 'Sweatshirts']` (Đa nghĩa Jackets/Sweaters/Sweatshirts; `n_relevant=11`)
- `val_b_synth_121` (`giay the thao nhe trang mau moi nhat`): `articleType` `['Casual Shoes']` -> `['Casual Shoes', 'Sports Shoes']` (Đa nghĩa Casual/Sports Shoes (khớp DEV h048); `n_relevant=64`)
- `val_b_synth_123` (`ao sweater ni cua nam di choi`): `articleType` `['Sweatshirts']` -> `['Sweatshirts', 'Sweaters']` (Đa nghĩa Sweatshirts/Sweaters; `n_relevant=11`)
- `val_b_synth_124` (`giày thể thao nhẹ nam nữ đều mang được đi chơi`): `articleType` `['Casual Shoes']` -> `['Casual Shoes', 'Sports Shoes']` (Đa nghĩa Casual/Sports Shoes (khớp DEV h048); `n_relevant=18`)
- `val_b_synth_125` (`tim mua giay mọi đen đi chơi`): `articleType` `['Flats']` -> `['Flats', 'Casual Shoes']` (Đa nghĩa Flats/Casual Shoes (giày lười đế bệt); `n_relevant=50`)
- `val_b_synth_136` (`can tim ao am cho nam`): `articleType` `['Jackets']` -> `['Jackets', 'Sweaters', 'Sweatshirts']` (Đa nghĩa Jackets/Sweaters/Sweatshirts; `n_relevant=24`)
- `val_b_synth_147` (`cho mình hỏi áo ấm nam giới đen mẫu mới nhất`): `articleType` `['Jackets']` -> `['Jackets', 'Sweaters', 'Sweatshirts']` (Đa nghĩa Jackets/Sweaters/Sweatshirts; `n_relevant=8`)
- `val_b_synth_165` (`typo_type=wrong_tone`): `dia_class` `full` -> `partial`
- `val_b_synth_171` (`cho mình hỏi ba lô laptop uisex mẫu mới nhất`): `articleType` `['Backpacks']` -> `['Backpacks', 'Laptop Bag']` (Đa nghĩa Backpacks/Laptop Bag (khớp DEV h066/h067); `n_relevant=29`)
- `val_b_synth_172` (`giày mọi chị em đenn tuyền giá tốt`): `articleType` `['Flats']` -> `['Flats', 'Casual Shoes']` (Đa nghĩa Flats/Casual Shoes (giày lười đế bệt); `n_relevant=13`)

## 3. Tập B-synth — Bổ sung 35 câu typo Telex/VNI/dính phím/sai thanh (2c.4b)

- `val_b_synth_210` (`telex_raw`, `dia_class=partial`): `tim mua ao coc tay nữ giới đi làm` -> `tim mua ao coc tay nux giới đi làm`
- `val_b_synth_211` (`vni_raw`, `dia_class=full`): `cần tìm sơ mi công sở của nam lông chuột đi làm` -> `cần1 tìm sơ mi công sở của nam lông chuột đi làm`
- `val_b_synth_212` (`sticky_keys`, `dia_class=partial`): `tư vấn giúp ba lô đi học xanh da trời đẹp xịn` -> `tưvấn giúp ba lô đi học xanh da trời đẹp xịn`
- `val_b_synth_213` (`wrong_tone`, `dia_class=partial`): `shop ơi có áo phông vàng đi làm` -> `shop ơi có ảo phông vàng đi làm`
- `val_b_synth_214` (`telex_raw`, `dia_class=partial`): `shop ơi có đồng hồ thời trang chị em` -> `shop ơi có ddoongf hồ thời trang chị em`
- `val_b_synth_215` (`vni_raw`, `dia_class=full`): `hoa tai nữ giá tốt` -> `hoa tai nu4 giá tốt`
- `val_b_synth_216` (`sticky_keys`, `dia_class=partial`): `cho minh hoi non luoi trai nam nữ đều mang được giá tốt` -> `cho minh hoi non luoi trai nam nữ đều mang được giátốt`
- `val_b_synth_217` (`wrong_tone`, `dia_class=partial`): `đồng hồ kim cho nữ trắng giá tốt` -> `đồng hồ kim cho nữ trẳng giá tốt`
- `val_b_synth_218` (`telex_raw`, `dia_class=partial`): `tìm mua áo sơ mi nam xanh biển giá tốt` -> `tìm mua aso sơ mi nam xanh biển giá tốt`
- `val_b_synth_219` (`vni_raw`, `dia_class=full`): `áo thun cotton bé trai đi chơi` -> `ao1 thun cotton bé trai đi chơi`
- `val_b_synth_220` (`sticky_keys`, `dia_class=none`): `tim mua dong ho deo tay cua nam mau bac mau moi nhat` -> `tim mua dong ho deotay cua nam mau bac mau moi nhat`
- `val_b_synth_221` (`wrong_tone`, `dia_class=partial`): `tu van giup túi đeo vai unisex` -> `tu van giup tủi đeo vai unisex`
- `val_b_synth_222` (`telex_raw`, `dia_class=full`): `giày vải chị em trắng tinh đi chơi` -> `giayf vải chị em trắng tinh đi chơi`
- `val_b_synth_223` (`vni_raw`, `dia_class=none`): `giay tap gym nam đen` -> `giay tap gym nam d9en`
- `val_b_synth_224` (`sticky_keys`, `dia_class=partial`): `ao phong co tron hồng phấn đi chơi` -> `ao phongco tron hồng phấn đi chơi`
- `val_b_synth_225` (`wrong_tone`, `dia_class=partial`): `túi công sở màu đen` -> `tủi công sở màu đen`
- `val_b_synth_226` (`telex_raw`, `dia_class=full`): `cần tìm dây lưng da của nam màu nâu đi làm` -> `cần tìm dây lưng da của nam màu naau đi làm`
- `val_b_synth_227` (`vni_raw`, `dia_class=partial`): `tim mua ao coc tay trắng tinh mặc hàng ngày` -> `tim mua ao coc tay tra8ng1 tinh mặc hàng ngày`
- `val_b_synth_228` (`sticky_keys`, `dia_class=partial`): `cho minh hoi quan bò chị em đi làm` -> `cho minh hoi quanbò chị em đi làm`
- `val_b_synth_229` (`wrong_tone`, `dia_class=partial`): `tim mua tui cong so chị em màu nâu đẹp xịn` -> `tim mua tui cong so chị em màu nẩu đẹp xịn`
- `val_b_synth_230` (`telex_raw`, `dia_class=full`): `tìm mua áo phông nữ màu trắng` -> `tìm mua aso phông nữ màu trắng`
- `val_b_synth_231` (`vni_raw`, `dia_class=partial`): `tu van giup ao thun cotton màu xám đẹp xịn` -> `tu van giup ao thun cotton mau2 xám đẹp xịn`
- `val_b_synth_232` (`sticky_keys`, `dia_class=partial`): `tim mua giay mọi đen đi chơi` -> `tim muagiay mọi đen đi chơi`
- `val_b_synth_233` (`wrong_tone`, `dia_class=partial`): `cần tìm áo phông cổ tròn cho nam màu trắng giá tốt` -> `cần tìm áo phông cổ tròn cho nam màu trẳng giá tốt`
- `val_b_synth_234` (`telex_raw`, `dia_class=full`): `giày gót nhọn nữ giá tốt` -> `giayf gót nhọn nữ giá tốt`
- `val_b_synth_235` (`vni_raw`, `dia_class=partial`): `tìm mua vớ cổ ngắn cho nam trắng tinh giá tốt` -> `tìm mua vớ cổ ngắn cho nam tra8ng1 tinh giá tốt`
- `val_b_synth_236` (`sticky_keys`, `dia_class=full`): `dép quai hậu nâu bò đi chơi` -> `dép quai hậu nâu bò đichơi`
- `val_b_synth_237` (`wrong_tone`, `dia_class=partial`): `shop ơi có bóp da đen đẹp xịn` -> `shop oi có bóp da đen đẹp xịn`
- `val_b_synth_238` (`telex_raw`, `dia_class=full`): `cần tìm giày tập gym nữ màu trắng đi làm` -> `cần tìm giayf tập gym nữ màu trắng đi làm`
- `val_b_synth_239` (`vni_raw`, `dia_class=partial`): `cho mình hỏi bóp da của nữ nâu đất đi chơi` -> `cho mình hỏi bóp da của nữ na6u đất đi chơi`
- `val_b_synth_240` (`sticky_keys`, `dia_class=full`): `giày vải của nam đi chơi` -> `giày vải của nam đichơi`
- `val_b_synth_241` (`wrong_tone`, `dia_class=partial`): `cho mình hỏi dép quai hậu của nam mặc hàng ngày` -> `cho mình hỏi dẹp quai hậu của nam mặc hàng ngày`
- `val_b_synth_242` (`telex_raw`, `dia_class=partial`): `cho mình hỏi dép lào màu đen đi chơi` -> `cho mình hỏi dép lào màu dden đi chơi`
- `val_b_synth_243` (`vni_raw`, `dia_class=partial`): `túi công sở chị em xanh da trời` -> `tui1 công sở chị em xanh da trời`
- `val_b_synth_244` (`sticky_keys`, `dia_class=none`): `cho minh hoi so mi cong so mau hong mau moi nhat` -> `cho minhhoi so mi cong so mau hong mau moi nhat`

Tổng kết B-synth: **245** câu (**165** sạch, **80** typo), **166** `spec_cluster_id` (**165** trên câu sạch), sửa nhãn đa nghĩa: **13**, sửa `dia_class`: **1**, thêm typo mới: **35** | SHA-256: `41739174392fd1fe4cb14c76bddefc33ff23cf7dd7e885d9e999987db3bea6a1`