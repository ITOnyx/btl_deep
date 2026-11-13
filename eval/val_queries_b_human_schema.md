# Hướng dẫn Schema cho `eval/val_queries_b_human.json` (>= 40 câu do người dùng viết)

Mỗi phần tử trong mảng JSON có định dạng:
```json
{
  "id": "val_b_human_000",
  "query": "câu truy vấn tự nhiên của bạn",
  "has_diacritics": true,
  "dia_class": "full | none | partial",
  "typo_type": null,
  "spec": {
    "articleType": [
      "Tshirts"
    ],
    "gender": "Men",
    "baseColour": [
      "Black"
    ]
  },
  "source": "human_written"
}
```

- Lưu ý: Chỉ ghi các thuộc tính **nêu rõ** trong câu vào `spec` (`gender: null` nếu không nêu giới tính, `baseColour: []` nếu không nêu màu).
- Sau khi bạn điền xong, hệ thống sẽ kiểm tra `n_relevant >= 5` trên `index/val_gallery/` và gắn `spec_cluster_id` rồi khóa SHA-256.
