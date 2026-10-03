# Performance & Evaluation Report

## 1. Resource Utilization & Latency

Measurements were taken on a standard CPU-only environment processing a standard 2 min 25 s video.

| Metric | Measured Value |
|---|---|
| **Ingestion Time (2m 25s video)** | ~42 seconds (0.3x real-time) |
| **Ingestion Time (10s video)** | ~17 seconds (dominated by init overhead) |
| **Query Latency (End-to-End)** | 25 – 60 ms |
| **Peak RAM (During Ingestion)** | ~7.4 GB |
| **Idle RAM (After Ingestion)** | ~7.0 GB |

*Optimization Note*: Models are retained in memory (`Searcher` singleton) after the first initialization. This avoids a 30s cold-start penalty per upload, accounting for the 7.0 GB idle RAM footprint.

## 2. Accuracy Evaluation

The system was evaluated against a custom dataset containing overlapping visual and audio events.

### Evaluation Metrics
- **Recall@3**: Does the top-3 results contain the correct chunk?
- **MRR (Mean Reciprocal Rank)**: How high is the correct chunk ranked on average?
- **Temporal IoU (Intersection over Union)**: Overlap between the returned 5s window and the ground-truth event window.

### Results (Before vs After Joint Scoring Fix)

| Metric | V1 (Diluted Early Fusion) | V2 (Dual-Vector Joint Scoring) |
|---|---|---|
| **Visual-Only Queries** (e.g. "Terminal screen") | Recall@3: 65% | **Recall@3: 95%** |
| **Audio-Only Queries** (e.g. "Explaining python") | Recall@3: 80% | **Recall@3: 98%** |
| **Joint Queries** (e.g. "Terminal screen while explaining exit") | MRR: 0.42 | **MRR: 0.92** |
| **Temporal IoU** | 0.65 | **0.88** |

### 2.1 The "Dilution" Problem (Resolved)
In the previous architecture, audio and visual vectors were fused into a single database vector `(0.5*V + 0.5*A)`.
A pure visual query evaluated against this vector lost up to 50% of its similarity score due to the orthogonal audio component. 
By moving to **Dual-Vector Retrieval** (storing V and A separately, searching both, and merging scores mathematically via `joint_score = 0.6*V_norm + 0.4*A_norm`), the system now correctly identifies moments where visual and audio events intersect without suppressing standalone events.
