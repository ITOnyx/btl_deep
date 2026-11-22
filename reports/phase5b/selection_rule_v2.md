# Quy tắc Chọn Cấu hình v2 (Khóa trước Bước 3)

1. **Mục tiêu tối đa hóa:**
   $$\text{Score} = 0.5 \cdot \text{mAP}_{\text{B-human}} + 0.3 \cdot \text{mAP}_{\text{B-synth (theo cụm)}} + 0.2 \cdot \text{mAP}_{\text{A (theo cụm)}}$$
2. **Ràng buộc tác hại trên Tập C mới (tiêm lỗi 20%):**
   - $\text{mAP}_{C,20\%} \ge \text{mAP}_{\text{model\_only}} - 0.03$
   - $\text{mAP}_{C,20\%} \ge \text{mAP}_{\text{baseline } (\gamma=0.3)} - 0.05$
3. **Ràng buộc trường hợp xấu nhất (Catastrophic Drop Rate):**
   - Tỉ lệ truy vấn có ảnh đúng trong Top-10 ở baseline nhưng **rớt khỏi Top-10** ở cấu hình mới $\le 5\%$ trên cả **B-human** và **B-synth**.
4. **Hạn chế winner's curse:**
   - Lưới $\le 24$ cấu hình định nghĩa trước; chọn bằng **5-fold CV theo cụm (`spec_cluster_id`)** trên A + B-synth, xác nhận trên B-human kèm bootstrap CI.
5. **Ưu tiên đơn giản:** Nếu chênh lệch Score nằm trong 95% paired bootstrap CI (theo cụm), chọn cấu hình đơn giản/nhanh hơn ($\gamma$ nhỏ, không mean-centering, không BM25).
6. **Báo cáo song song:** Báo cả cấu hình **Tốt nhất theo Score** và cấu hình **An toàn** (thỏa ràng buộc 2 & 3 với $\gamma$ thấp nhất).
