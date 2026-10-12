# LFM2.5-VL-3B + DSpark — 64K Benchmark (Jetson Orin Nano 8GB)

**Tags:** `llm` `benchmark` `llama.cpp` `Jetson` `liquid-ai` `lfm` `speculative-decoding` `long-context` `needle-recall` `edge-ai`

Benchmarks **LFM2.5-VL-3B** (Q4_K_M, 1.56 GB) with the **DSpark** speculative drafter (F16, 0.53 GB) at **64K context with an f16 KV cache** on a Jetson Orin Nano 8 GB, using YaRN `--rope-scale 2.0` to stretch past the model's 32K native context (`n_ctx_seq 65536 > n_ctx_train 32768` — deep-context quality is a measured finding, not a given).

## Key results (34-prompt suite A–D, temp 0.3 / top_p 0.95 / top_k 40)

- **35.2 tok/s** average generation (peak **51.7**), one prompt per fresh server load
- **36% fewer tokens** than Qwen3.5-2B on identical prompts
- Reasoning traps **5/5** correct; tool calls **5/6**
- Deep needle **PINECONE-42**: exact recall at **30.6K** depth (inside native 32K); degrades to "PINECONE" at **49.3K** under greedy decoding — exact-token recall beyond the training window is not trustworthy, semantic retrieval survives
- RSS steady at 4.84 GB; 1,349 MB minimum free headroom, zero OOM events

## Files

- `lfm_bench.py` — harness: kill → launch → health-wait → one prompt → capture tok/s + RSS → kill → JSONL (resume-safe)
- `needle_test.py` — deep-context needle-recall probes (incl. greedy reruns)
- `gguf_ctx.py` — context/KV cache feasibility math
- `prompts_abcd.json` — the 34-prompt A–D suite (shared with the qwen35-2b-benchmark campaign)
- `results/` — per-prompt JSONL, needle results, and `LFM25-3B_64K_Benchmark_Report.md` (full report)

Run the report first; the summary metrics and per-suite tables live there.