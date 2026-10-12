# LFM2.5-VL-3B + DSpark — 64K f16 KV Benchmark — 2026-10-11

**Model:** LFM2.5-VL-3B (Q4_K_M, 1.56 GB) + DSpark speculative drafter (F16, 0.53 GB)
**Hardware:** Jetson Orin Nano 8 GB, GUI off, Hermes gateway resident
**Server:** llama.cpp llama-server, `-c 65536` **f16 KV cache** (no -ctk/-ctv),
YaRN `--rope-scaling yarn --rope-scale 2.0` — extends past the model's 32K
native context (log: `n_ctx_seq (65536) > n_ctx_train (32768)`), so deep-context
quality is a measured finding, not a given.
**Method:** one prompt per FRESH server load — kill → launch → wait health →
run → kill — 34 cycles, per the suite from `qwen35-2b-benchmark` (suites A–D,
identical prompts/sampling: temp 0.3, top_p 0.95, top_k 40). Production Qwen
server stopped and keepalive cron paused during the campaign; both restored after.

Raw data: `results/bench_lfm_20261011.jsonl` (per-prompt metrics + server facts + full text)
Summary: `results/summary_metrics.txt`

---

## Speed (generation tok/s, streaming, DSpark active)

| Suite | n | Avg gen tok/s | Range |
|---|---|---|---|
| A — general (400-tok caps) | 5 | 34.6 | 21.7–48.1 |
| B — one-shot coding (2000-tok) | 5 | **44.6** | 40.8–47.5 |
| C — zoo/ornith | 12 | 29.1 | 10.8–51.7 |
| D — reasoning traps | 12 | 37.6 | 25.0–51.0 |
| **Overall** | **34** | **35.2** | peak **51.7** |

- Prompt processing: avg 106.5 t/s (cold cache every test — the per-test restarts mean zero prompt-cache reuse)
- Note: the C-suite average is dragged down by 10–20 tok/s short tool-call
  bursts (13–37 tokens); its quality prompts averaged **41.7 tok/s**.

## Server cycles (one per test, by design)

- 34 launches, **avg 9.4 s to healthy**, all first-try
- RSS steady at **4.84 GB** (64K f16 KV ≈ 1.25 GB + 1.56 GB weights + 0.53 GB drafter + buffers)
- Minimum free memory after load: **1,349 MB** — comfortable margin, no OOM, no stalls

## Token efficiency (vs Qwen3.5-2B on the identical suite)

| Metric | LFM2.5-VL-3B | Qwen3.5-2B |
|---|---|---|
| Total completion tokens | **12,973** | 20,138 |
| Truncated (finish=length) | **3** | 7 |
| Clean stops in suites B+C | **11/11** | 9/11 (b_html, c_html_game hit cap) |

LFM emits ~36% fewer tokens for the same prompts and finished the big coding
prompts well under budget where Qwen hit the cap (b_html 1590 clean vs 2000
truncated; c_html_game 606 clean vs 2048 truncated).

## Head-to-head (same prompts, same sampling)

- Per-prompt gen-speed wins: Qwen3.5-2B 19, LFM 15 (Qwen's MTP edges it on
  small outputs; LFM's DSpark wins on long code: 44.6 vs 41.8 on suite B)
- 64K memory: LFM wins decisively — hybrid KV makes 64K f16 cost ~1.25 GB
  (Qwen needed q8_0 KV to fit; f16 would be ~2.5 GB)

## Objective auto-verdicts

- **Reasoning traps: 5/5 correct** — strawberry 3 ✓, elephant 2 ✓, sheep 9 ✓,
  bat & ball $0.05 ✓ (full algebra), sum 1–100 = 5050 ✓ — all with concise,
  correct breakdowns (elephant's enumeration is actually cleaner than Qwen's,
  which hallucinated a phantom 'l')
- **Tool calls: 5/6 tool-name PASS, all first-turn, well-formed JSON args**
  (web_search, df -h /, ~/todo.txt with items, /etc/hostname) —
  ⚠️ **c_func_sqlite is a partial fail**: `terminal` with
  `sql -U root -d clinic.db SELECT name, visit_date FROM patients WHERE visit_date >= CURRENT_DATE - INTERVAL '30 days'`
  — right table/columns/date-filter concept, but that is **psql syntax, not
  sqlite3** (`sql` binary doesn't exist; SQLite has no `-U` flag or `INTERVAL`).
  Qwen got this one right with `sqlite3 clinic.db "SELECT..."`.

## Open items

- Full 1–10 rubric grading of all 34 responses (raw text saved in the JSONL) — not yet done
- Suite A at 6000 max tokens (3 of 5 A-prompts hit the historical 400 cap —
  same truncation artifact as Qwen's run)
- A deep-context needle test at ≥48K depth would directly validate the YaRN
  extension (the A–D prompts here are all short-prompt)