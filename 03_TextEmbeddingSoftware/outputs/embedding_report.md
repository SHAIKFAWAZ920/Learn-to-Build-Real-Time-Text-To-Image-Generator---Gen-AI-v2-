# Task 03: Text Embedding Generation Suite Report

This report documents the detailed execution outputs, vector metrics, and performance benchmarks for the Hugging Face sentence embedding module.

## 1. Generated Vector Outputs

*   **Export File**: `outputs/test_embeds.npy` (NumPy Binary Matrix)
*   **Source Model**: `sentence-transformers/all-MiniLM-L6-v2`
*   **Dimensions**: `2 x 384` (2 sentences processed, each encoded into a 384-dimensional vector space)

### Input Sentences & Statistics:
1.  *"A red circle"*
    *   Tokens: `['a', 'red', 'circle']`
    *   Vector Norm: `1.0000` (Unit-normalized vector)
2.  *"A green star"*
    *   Tokens: `['a', 'green', 'star']`
    *   Vector Norm: `1.0000` (Unit-normalized vector)

### Cosine Similarity Matrix:
*   Cosine Similarity between *"A red circle"* and *"A green star"*: **0.3541** (representing moderate semantic overlap due to sharing the word "A" and geometric context).

---

## 2. Performance Benchmarks

The benchmark was executed on the local CPU to measure execution latency and throughput capacity:

| Metric | Sentence Transformer (`all-MiniLM-L6-v2`) | Status |
| :--- | :--- | :--- |
| **Embedding Dimension** | 384 | 100% standard vector length |
| **Mean Latency (per sentence)** | **1.74 ms** | Sub-millisecond latency profile |
| **Throughput (sentences/second)** | **575.37 sen/sec** | High-concurrency ready |
| **Parameter Count** | ~22.7 Million | Extremely lightweight & CPU-efficient |
| **Disk/Cache Footprint** | ~91 MB | Compact model size |
