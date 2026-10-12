#!/usr/bin/env python3
"""needle_test.py — deep-context needle retrieval at ~48K depth for the
LFM2.5-VL-3B 64K f16 YaRN run. Needle construction identical to the 64k
project's grade_one.py T3, so results are comparable across models.

One fresh server cycle: kill -> launch -> wait health -> send ~48K-token
prompt (needle at the very start, filler after) -> measure pp at depth ->
answer must be PINECONE-42.
"""
import json
import os
import time

from lfm_bench import (kill_server, launch_server, server_facts, stream_chat,
                       RESULTS, ALIAS, CTX)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "needle_48k.json")
LOG = "/tmp/lfm64k_needle.log"

NEEDLE = "The secret access word for the vault is PINECONE-42."
FILLER = ("The quarterly logistics review covers inventory turnover, "
          "carrier contracts, warehouse throughput and demand forecasts. ")
# Measured: LFM's tokenizer packs this line into ~17.0 tokens (not the ~26 of
# Qwen-class tokenizers) — 1800 reps only reached 30.6K depth, still inside the
# 32K native boundary. 2900 reps ~= 49.4K tokens: a true beyond-training depth.
FILLER_BIG = FILLER * 2900
PROMPT = (NEEDLE + "\n\n" + FILLER_BIG +
          "\n\nWhat is the secret access word for the vault? "
          "Answer with just the word.")


def main():
    print("launching fresh server...", flush=True)
    kill_server()
    proc, load_s, _ = launch_server(LOG)
    if proc is None:
        print("LOAD FAILED")
        raise SystemExit(1)
    facts = server_facts(proc, LOG)
    print(f"loaded in {load_s}s  RSS {facts['rss_gb']} GB  "
          f"free {facts['free_after_load_mb']} MB", flush=True)

    print(f"sending needle prompt (~{len(PROMPT)//4} chars)...", flush=True)
    t_send = time.time()
    r = stream_chat(PROMPT, max_tokens=64, timeout=600)
    r["wall_from_send_s"] = round(time.time() - t_send, 2)

    depth = r.get("prompt_tokens")
    answer = (r.get("text") or "").strip()
    ok = "PINECONE-42" in answer.upper()
    rec = {"id": "needle_48k", "model": ALIAS, "ctx": CTX, "kv_quant": "f16",
           "yarn": "rope-scale 2.0", "n_ctx_train": 32768,
           "server": {"load_s": load_s, **facts},
           "needle": "PINECONE-42", "grade": "PASS" if ok else "FAIL",
           "depth_prompt_tokens": depth, "answer": answer[:200],
           "ttft_s": r.get("ttft_s"), "total_s": r.get("total_s"),
           "pp_tok_s_at_depth": r.get("pp_tok_s"),
           "gen_tok_s": r.get("gen_tok_s"),
           "completion_tokens": r.get("completion_tokens"),
           "finish_reason": r.get("finish_reason")}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rec, open(OUT, "w"), indent=1)
    print(json.dumps(rec, indent=1))

    proc.kill()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()