# E2E Test Infra: Multimodal Visual Search

## Test Philosophy
- Requirement-driven, opaque-box testing based on `ORIGINAL_REQUEST.md`.
- Verification of functional retrieval, metric targets (Recall@10 >= 0.85, mAP >= 0.75, Latency < 200ms), and UI stability.

## Feature Inventory
| # | Feature | Source | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|--------|:------:|:------:|:------:|:------:|
| 1 | Checkpoint & Model Loading | R1 | 5 | 5 | ✓ | ✓ |
| 2 | FAISS HNSW Real Index & Metadata | R1 | 5 | 5 | ✓ | ✓ |
| 3 | Vietnamese Query Preprocessing | R2 | 5 | 5 | ✓ | ✓ |
| 4 | Dense + BM25 + RRF Hybrid Retrieval | R2 | 5 | 5 | ✓ | ✓ |
| 5 | Recall@10 & mAP Benchmark Target | AC1, AC2 | 5 | 5 | ✓ | ✓ |
| 6 | Latency Performance (<200ms) | AC3 | 5 | 5 | ✓ | ✓ |
| 7 | Streamlit UI (Text, Image, CIR) | R3, AC4 | 5 | 5 | ✓ | ✓ |

## Test Architecture
- Test Runner: `pytest tests/` or `python -m unittest discover -s tests`
- Evaluation Suite: `scripts/evaluation/evaluate_retrieval.py` & `scripts/evaluation/evaluate_test_and_plot.py`
- Pass/Fail Semantics:
  - Recall@10 >= 0.85 on test split of `golden_queries_vi.jsonl`
  - mAP >= 0.75 on test split of `golden_queries_vi.jsonl`
  - Latency < 200ms per query on GPU
  - 100% of rendered images exist on disk (no broken image paths)
  - Streamlit imports & initialization cleanly without mock vectors

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Acceptance Target |
|---|----------|--------------------|-------------------|
| 1 | Text Search on 200 Golden Queries | F1, F2, F3, F4 | Recall@10 >= 0.85, mAP >= 0.75 |
| 2 | Query Latency on RTX 2050 | F1, F2, F4 | Mean latency < 200ms |
| 3 | Image-to-Image Visual Search | F1, F2 | Top-10 relevant items returned |
| 4 | Composed Image Retrieval (CIR) | F1, F2, F4 | Modified query attributes retrieved |
| 5 | Streamlit Full Workflow Smoke Test | F1, F2, F5, F7 | Clean start, no exceptions, real images rendered |
