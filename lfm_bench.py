#!/usr/bin/env python3
"""
lfm_bench.py — LFM2.5-VL-3B + DSpark @ 64K f16 KV, one prompt per fresh server.

Per prompt: kill any llama-server -> launch LFM (YaRN rope-scale 2.0 past its
32K native ctx, DSpark speculative drafter, f16 KV) -> wait health -> run ONE
prompt (streaming, include_usage) -> capture load/RSS/free mem + tok/s ->
kill server -> append record to JSONL (resume-safe, keyed on id).

Launches use setsid (start_new_session) from this FOREGROUND process — never
run this script through the terminal background wrapper (memcg OOM trap).

Usage:
  python3 lfm_bench.py --ids d_r_strawberry            # one or more ids
  python3 lfm_bench.py --suite a                       # whole suite
  python3 lfm_bench.py --list                          # show ids + done state
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(HERE, "prompts_abcd.json")
RESULTS = os.path.join(HERE, "results", "bench_lfm_20261011.jsonl")

SERVER = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
MODEL = os.path.expanduser("~/models/new-zoo/LFM2.5-VL-3B-Q4_K_M.gguf")
DRAFTER = os.path.expanduser("~/models/new-zoo/LFM2.5-VL-3B-DSpark-F16.gguf")
ALIAS = "lfm25-3b-64k"
PORT = 8090
BASE = f"http://127.0.0.1:{PORT}"
CTX = 65536

# Launch flags: verified DSpark command (granite4-jetson-hermes 2026-10-01) with
# -c 65536 + YaRN rope-scale 2.0 (64k-context-hermes-8g Gemma-4-E2B precedent,
# also 32K native) and NO -ctk/-ctv -> f16 KV cache.
LAUNCH_ARGS = [
    SERVER, "-m", MODEL, "-md", DRAFTER,
    "--spec-type", "draft-dspark",
    "--alias", ALIAS, "--host", "127.0.0.1", "--port", str(PORT),
    "-ngl", "99", "-c", str(CTX),
    "--rope-scaling", "yarn", "--rope-scale", "2.0",
    "-b", "512", "-ub", "512", "-fa", "on", "--jinja", "--fit", "off",
    "-np", "1", "-t", "6",
    "--temp", "0.0", "--top-k", "1", "--repeat-penalty", "1.0",
]

# Suite sampling — identical to the Qwen3.5-2B A-D run (temp 0.3, llama defaults).
SAMPLING = {"temperature": 0.3, "top_p": 0.95, "top_k": 40, "min_p": 0.0,
            "repeat_penalty": 1.0}

TOOL_SCHEMAS = {
    "web_search": {"type": "function", "function": {
        "name": "web_search", "description": "Search the web for current information",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Search query"}},
                       "required": ["query"]}}},
    "terminal": {"type": "function", "function": {
        "name": "terminal", "description": "Execute a shell command on the user's system",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string", "description": "Shell command to run"}},
                       "required": ["command"]}}},
    "write_file": {"type": "function", "function": {
        "name": "write_file", "description": "Write content to a file on disk",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "File path"},
                                      "content": {"type": "string", "description": "File content"}},
                       "required": ["path", "content"]}}},
    "read_file": {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from disk",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "File path to read"}},
                       "required": ["path"]}}},
}


def free_mb():
    """Available MB from free(1), parsed in Python (no shell)."""
    r = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=10)
    parts = r.stdout.splitlines()[1].split()
    return parts[6] if len(parts) > 6 else None


def server_running():
    r = subprocess.run(["pgrep", "-f", "[l]lama-server -m"],
                       capture_output=True, text=True, timeout=10)
    return r.stdout.strip() != ""


def log_tail(path, n=5):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip()
    except OSError:
        return ""


def http_ok(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def kill_server():
    subprocess.run(["pkill", "-9", "-f", "[l]lama-server -m"], capture_output=True)
    for _ in range(20):
        if not server_running():
            break
        time.sleep(0.5)
    # wait for the port to actually free
    for _ in range(10):
        if not http_ok(f"{BASE}/health", timeout=1):
            return
        time.sleep(1)


def launch_server(log_path):
    env = dict(os.environ)
    env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    env["GGML_CUDA_NO_PINNED"] = "1"
    logf = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(LAUNCH_ARGS, stdout=logf, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    deadline = time.time() + 150
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                if b"ok" in r.read():
                    return proc, round(time.time() - t0, 1), log_path
        except Exception:
            pass
        if proc.poll() is not None:
            return None, round(time.time() - t0, 1), log_path
        time.sleep(2)
    proc.kill()
    return None, round(time.time() - t0, 1), log_path


def server_facts(proc, log_path):
    facts = {"rss_gb": None, "free_after_load_mb": int(free_mb() or -1),
             "kv_line": None, "n_ctx_train": None}
    try:
        with open(f"/proc/{proc.pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    facts["rss_gb"] = round(int(line.split()[1]) / 1048576, 2)
    except OSError:
        pass
    try:
        log = open(log_path, errors="replace").read()
        m = re.search(r"KV self size *= *([\d.]+ *\w+)", log)
        if m:
            facts["kv_line"] = m.group(1)
        m = re.search(r"n_ctx_train *= *(\d+)", log)
        if m:
            facts["n_ctx_train"] = int(m.group(1))
    except OSError:
        pass
    return facts


def stream_chat(prompt, tools=None, max_tokens=1024, timeout=300):
    body = {"model": ALIAS,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, **SAMPLING}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text_parts, tool_calls = [], {}
    usage, finish = None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    text_parts.append(d["content"])
                for tc in d.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    total = time.time() - t0
    gen_time = (total - ttft) if ttft else total
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens")
    return {"text": "".join(text_parts),
            "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
            "finish_reason": finish, "prompt_tokens": pt, "completion_tokens": ct,
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "total_s": round(total, 2),
            "pp_tok_s": round(pt / ttft, 1) if (pt and ttft) else None,
            "gen_tok_s": round(ct / gen_time, 2) if (ct and gen_time > 0) else None,
            "eff_tok_s": round(ct / total, 2) if (ct and total > 0) else None}


def run_ids(targets, spec):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    out_f = open(RESULTS, "a")
    for p in targets:
        pid_ = p["id"]
        log_path = f"/tmp/lfm64k_{pid_}.log"
        print(f"\n=== {pid_} ({p['label']}) ===", flush=True)
        kill_server()
        proc, load_s, log_path = launch_server(log_path)
        if proc is None:
            tail = log_tail(log_path)
            rec = {"id": pid_, "error": f"SERVER LOAD FAILED after {load_s}s", "log_tail": tail}
            out_f.write(json.dumps(rec) + "\n"); out_f.flush()
            print(f"  LOAD FAILED after {load_s}s\n  {tail}")
            continue
        facts = server_facts(proc, log_path)
        print(f"  loaded in {load_s}s  RSS {facts['rss_gb']} GB  KV {facts['kv_line']}  "
              f"free {facts['free_after_load_mb']} MB", flush=True)
        tools = [TOOL_SCHEMAS[p["tool"]]] if p.get("tool") in TOOL_SCHEMAS else None
        try:
            r = stream_chat(p["prompt"], tools=tools, max_tokens=p["max_tokens"])
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"}
        rec = {"id": pid_, "suite": p["suite"], "category": p["category"],
               "label": p["label"], "max_tokens": p["max_tokens"],
               "expected_tool": p.get("expected_tool"),
               "server": {"load_s": load_s, "alias": ALIAS, "ctx": CTX,
                          "kv_quant": "f16", "yarn": "rope-scale 2.0",
                          **facts}, **r}
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        if "error" in r:
            print(f"  REQUEST ERROR: {r['error']}")
        else:
            tc = r["tool_calls"]
            tc_s = f" tool={tc[0]['name']}({tc[0]['arguments'][:50]})" if tc else ""
            print(f"  {r['completion_tokens']} tok in {r['total_s']}s  gen={r['gen_tok_s']} t/s  "
                  f"pp={r['pp_tok_s']} t/s  finish={r['finish_reason']}{tc_s}", flush=True)
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    out_f.close()
    summarize()


def summarize():
    rows = [json.loads(x) for x in open(RESULTS) if x.strip()]
    print(f"\n{len(rows)} rows in {RESULTS}")
    by = {}
    for row in rows:
        if row.get("gen_tok_s"):
            by.setdefault(row.get("suite", "?"), []).append(row["gen_tok_s"])
    for s, v in sorted(by.items()):
        print(f"  suite {s}: n={len(v)}  avg gen {sum(v)/len(v):.2f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma-separated prompt ids")
    ap.add_argument("--suite", choices=["a", "b", "c", "d"])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    spec = json.load(open(PROMPTS_FILE))
    prompts = spec["prompts"]
    done = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass

    if args.list:
        for p in prompts:
            mark = "DONE " if p["id"] in done else "     "
            print(f"{mark}{p['id']:20s} {p['suite']}/{p['category']}")
        return

    if args.ids:
        want = {x.strip() for x in args.ids.split(",")}
        targets = [p for p in prompts if p["id"] in want]
    elif args.suite:
        targets = [p for p in prompts if p["suite"].lower() == args.suite]
    else:
        ap.error("need --ids, --suite, or --list")

    missing = [t["id"] for t in targets if t["id"] in done]
    if missing:
        print(f"SKIP (already recorded): {', '.join(missing)}")
    targets = [t for t in targets if t["id"] not in done]
    if not targets:
        print("nothing to run")
        return
    run_ids(targets, spec)


if __name__ == "__main__":
    main()