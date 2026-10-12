#!/usr/bin/env python3
"""summarize_lfm.py — metrics tables + objective auto-verdicts for the LFM run."""
import json

HERE = "/home/walker/projects/lfm25-3b-64k-benchmark"
rows = [json.loads(x) for x in open(f"{HERE}/results/bench_lfm_20261011.jsonl") if x.strip()]
qrows = {r["id"]: r for r in (json.loads(x) for x in
          open("/home/walker/projects/qwen35-2b-benchmark/results/bench_20261011.jsonl"))}

print(f"{len(rows)} rows")

# ---- speed per suite ----
print("\n== SPEED (gen tok/s) ==")
suites = {}
for r in rows:
    if r.get("gen_tok_s"):
        suites.setdefault(r["suite"], []).append(r)
all_gen = []
for s in sorted(suites):
    rs = suites[s]
    gens = [r["gen_tok_s"] for r in rs]
    all_gen += gens
    print(f"  {s}: n={len(rs)}  avg={sum(gens)/len(gens):.2f}  min={min(gens):.1f}  max={max(gens):.1f}")
print(f"  OVERALL: {len(all_gen)} prompts, avg={sum(all_gen)/len(all_gen):.2f} tok/s, "
      f"peak={max(all_gen):.2f}")

# load metrics
loads = [r["server"]["load_s"] for r in rows if r.get("server", {}).get("load_s")]
rss = [r["server"]["rss_gb"] for r in rows if r.get("server", {}).get("rss_gb")]
freeal = [r["server"]["free_after_load_mb"] for r in rows if r.get("server", {}).get("free_after_load_mb")]
print(f"\n== SERVER CYCLES ==\n  loads: n={len(loads)} avg={sum(loads)/len(loads):.1f}s  "
      f"RSS avg={sum(rss)/len(rss):.2f} GB  free-after-load min={min(freeal)} MB")

# truncation + token totals
trunc = [r["id"] for r in rows if r.get("finish_reason") == "length"]
tot_tok = sum(r.get("completion_tokens") or 0 for r in rows)
qtot = sum(qrows[r["id"]].get("completion_tokens") or 0 for r in rows if r["id"] in qrows)
print(f"\n== TOKENS ==\n  total completion tokens: {tot_tok} (qwen3.5-2b same suite: {qtot})")
print(f"  truncated (finish=length): {len(trunc)} -> {', '.join(trunc)}")

# prompt processing
pps = [r["pp_tok_s"] for r in rows if r.get("pp_tok_s")]
print(f"  pp_tok_s: n={len(pps)} avg={sum(pps)/len(pps):.1f}")

# ---- objective auto-verdicts ----
print("\n== D TRAP ANSWERS (auto-checked) ==")
TRAPS = {
    "d_r_strawberry": ["3"], "d_e_elephant": ["2"], "d_sheep": ["9"],
    "d_batball": ["$0.05", "0.05"], "d_sum100": ["5050"],
}
for tid, keys in TRAPS.items():
    r = next(x for x in rows if x["id"] == tid)
    t = r.get("text", "")
    ok = any(k in t for k in keys)
    print(f"  {'PASS' if ok else 'FAIL'} {tid}: {t[:90]!r}")

print("\n== TOOL CALLS (auto-checked) ==")
EXPECT = {"c_func_web_search": ("web_search", None), "c_func_terminal": ("terminal", "df"),
          "c_func_write_file": ("write_file", "todo.txt"), "c_func_read_file": ("read_file", "/etc/hostname"),
          "c_func_sqlite": ("terminal", None)}
for r in rows:
    tid = r["id"]
    if tid not in EXPECT:
        continue
    exp_tool, hint = EXPECT[tid]
    tcs = r.get("tool_calls") or []
    got = tcs[0]["name"] if tcs else None
    args = tcs[0]["arguments"] if tcs else ""
    name_ok = got == exp_tool
    hint_ok = (hint is None) or (hint in args)
    print(f"  {'PASS' if name_ok and hint_ok else 'CHECK'} {tid}: tool={got} "
          f"{'(right tool)' if name_ok else '(WRONG TOOL)'} args={args[:80]!r}")
    if tid == "c_func_sqlite":
        print(f"        full sqlite args: {args!r}")

# compare speed head-to-head where both runs exist
print("\n== HEAD-TO-HEAD gen tok/s (same prompts, qwen3.5-2b vs lfm2.5-vl-3b) ==")
wins = {"qwen": 0, "lfm": 0}
for r in rows:
    q = qrows.get(r["id"])
    if q and q.get("gen_tok_s") and r.get("gen_tok_s"):
        w = "lfm" if r["gen_tok_s"] > q["gen_tok_s"] else "qwen"
        wins[w] += 1
print(f"  prompt wins: lfm={wins['lfm']}  qwen={wins['qwen']}  (of {wins['lfm']+wins['qwen']})")

# per-prompt table for the report
print("\n== PER-PROMPT ==")
def fmt(v, w=6, typ="f"):
    if v is None:
        return str("-").rjust(w)
    return f"{v:>{w}}" if typ == "s" else f"{v:>{w}.1f}"
for r in rows:
    s = r.get("server", {})
    print(f"  {r['id']:20s} {fmt(r.get('completion_tokens'), 5, 's')} tok  "
          f"gen={fmt(r.get('gen_tok_s'))}  pp={fmt(r.get('pp_tok_s'))}  "
          f"load={fmt(s.get('load_s'), 5, 's')}s  finish={r.get('finish_reason')}")