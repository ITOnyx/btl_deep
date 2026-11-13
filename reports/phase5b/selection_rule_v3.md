# Quy tắc Chọn Cấu hình v3 (Khóa trước Bước 3)

1. **Lưới cấu hình tường minh (22 cấu hình $\le 24$):** Liệt kê đầy đủ trong [selection_rule_v3.json](selection_rule_v3.json), theo thứ tự độ đơn giản tăng dần:
   `mean-centering tắt < BM25 tắt < dual-encode tắt < query-rewrite tắt < bonus đều < γ nhỏ hơn`.
2. **Định nghĩa đánh giá:** `mAP` và `R@10` tính bằng **exact search toàn gallery** (không cắt top-100), áp dụng `tie_rule="conservative"` ($\epsilon = 10^{-5}$). `"mAP theo cụm"` = trung bình `mAP` trong từng `spec_cluster_id`, sau đó lấy trung bình giữa các cụm.
3. **Chọn cấu hình bằng 5-fold CV theo cụm trên A + B-synth:**
   $$\text{Score} = 0.4 \cdot \text{mAP}_{A\text{ (theo cụm)}} + 0.6 \cdot \text{mAP}_{B\text{-synth sạch (theo cụm)}}$$
   *(Phần typo của B-synth báo cáo riêng; B-human KHÔNG nằm trong hàm mục tiêu.)*
4. **Cổng xác nhận B-human:** Cấu hình thắng không được kém baseline (`parser_v1`, $\gamma=0.3$ đều) quá `0.02` mAP ở cận dưới 95% bootstrap CI theo cụm, và tỉ lệ tụt thảm họa $\le 5\%$.
5. **Ràng buộc bắt buộc:**
   - **(i) Tác hại trên C lồng nhau mức 20% (lỗi thực):** $\text{mAP} \ge \text{model\_only} - 0.03$ và $\ge \text{baseline} - 0.05$.
   - **(ii) Tụt thảm họa trên B-synth:** $\le 5\%$.
   - **(iii) R@10 trên A + B-synth:** không kém baseline quá `0.01`.
   - **(iv) Độ trễ Text→Image p95:** tăng $\le 20\text{ ms}$ và $< 200\text{ ms}$.
   - **(v) Dương tính giả parser:** tỉ lệ slot dương tính giả không cao hơn `parser_v1`.
6. **Phá hòa theo CI:** Nếu chênh lệch Score nằm trong 95% paired cluster bootstrap CI, chọn cấu hình đơn giản hơn theo thứ tự mục 1.
7. **Báo cáo song song:** Báo cả cấu hình **Tốt nhất theo Score** và cấu hình **An toàn** (thỏa 5(i)(ii) với $\gamma$ thấp nhất).
8. **Ablation bắt buộc:** Luôn báo hàng **`cfg_02` (`parser_v2 + γ=0.3 đều`)** để tách hiệu quả sửa parser khỏi hiệu quả $\gamma$/trọng số slot.
