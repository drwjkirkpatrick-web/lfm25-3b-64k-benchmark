#!/usr/bin/env python3
"""gguf_ctx.py — read architecture + native context length straight from GGUF
headers (no model load). Usage: python3 gguf_ctx.py file1.gguf [file2.gguf ...]"""
import struct
import sys

# GGUF v3 value-type enum (spec): 0=uint8 1=int8 2=uint16 3=int16 4=uint32
# 5=int32 6=float32 7=bool(1B) 8=string 9=array 10=uint64 11=int64 12=float64
TYPES = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
         5: ("<i", 4), 6: ("<f", 4), 7: "bool", 8: "str", 9: "arr",
         10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}

def read_str(f, v1=False):
    n = struct.unpack("<I" if v1 else "<Q", f.read(4 if v1 else 8))[0]
    if n > 1_000_000:
        raise ValueError(f"implausible string length {n} — wrong version?")
    return f.read(n).decode("utf-8", "replace")

def read_kv(f, v1=False):
    key = read_str(f, v1)
    vtype = struct.unpack("<I", f.read(4))[0]
    if vtype == 8:
        val = read_str(f, v1)
    elif vtype == 7:
        val = struct.unpack("<B", f.read(1))[0]
    elif vtype == 9:
        etype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<I" if v1 else "<Q", f.read(4 if v1 else 8))[0]
        if etype == 8:
            val = [read_str(f, v1) for _ in range(n)]
        elif etype == 7:
            val = [struct.unpack("<B", f.read(1))[0] for _ in range(n)]
        else:
            fmt, sz = TYPES[etype]
            val = [struct.unpack(fmt, f.read(sz))[0] for _ in range(n)]
    elif vtype in (0, 1, 2, 3, 4, 5, 6, 10, 11, 12):
        fmt, sz = TYPES[vtype]
        val = struct.unpack(fmt, f.read(sz))[0]
    else:
        raise ValueError(f"unhandled gguf value type {vtype} for key {key}")
    return key, val

def header(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"{path}: not a GGUF"
        version = struct.unpack("<I", f.read(4))[0]
        v1 = (version == 1)
        struct.unpack("<I" if v1 else "<Q", f.read(4 if v1 else 8))  # n_tensors
        n_kv = struct.unpack("<I" if v1 else "<Q", f.read(4 if v1 else 8))[0]
        kv = dict(read_kv(f, v1) for _ in range(n_kv))
    return kv

for path in sys.argv[1:]:
    kv = header(path)
    arch = kv.get("general.architecture", "?")
    ctx = kv.get(f"{arch}.context_length")
    blocks = kv.get(f"{arch}.block_count")
    kvh = kv.get(f"{arch}.attention.head_count_kv")
    klen = kv.get(f"{arch}.attention.key_length")
    vlen = kv.get(f"{arch}.attention.value_length")
    name = kv.get("general.name", path.split("/")[-1])
    # f16 KV bytes/token = layers * kvh * (klen + vlen) * 2
    if all(v is not None for v in (blocks, kvh, klen, vlen)):
        kv_tok = blocks * kvh * (klen + vlen) * 2
        kv64k_mb = kv_tok * 65536 / 1e6
    else:
        kv_tok = kv64k_mb = None
    print(f"{name:26s} arch={arch:12s} native_ctx={ctx:>7} "
          f"layers={blocks} kvh={kvh} kv_f16_tok={kv_tok} "
          f"kv64k_f16={kv64k_mb and int(kv64k_mb)}MB q8={kv64k_mb and int(kv64k_mb/2)}MB")