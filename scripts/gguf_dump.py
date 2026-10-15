#!/usr/bin/env python3
"""Dump GGUF metadata + tensor inventory. Pure stdlib, no deps."""
import struct, sys, json

PATH = sys.argv[1] if len(sys.argv) > 1 else "refs/harrier-oss-v1-270m-gguf-q8_0/harrier-oss-v1-270m-Q8_0.gguf"
f = open(PATH, "rb")

def r(fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

assert f.read(4) == b"GGUF", "not a GGUF file"
ver = r("<I"); n_tensors = r("<Q"); n_kv = r("<Q")

def read_str():
    n = r("<Q")
    return f.read(n).decode("utf-8", errors="replace")

TYPES = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",8:"str",9:"arr",10:"u64",11:"i64",12:"f64"}

def read_val(t, depth=0):
    if t == 8: return read_str()
    m = {0:("<B"),1:("<b"),2:("<H"),3:("<h"),4:("<I"),5:("<i"),6:("<f"),7:("<B"),10:("<Q"),11:("<q"),12:("<d")}
    if t in m: return r(m[t])
    if t == 9:
        et = r("<I"); n = r("<Q")
        # skip array contents (tokenizer blobs) — record type+count only
        if et == 8:
            # need real size to skip: strings are len-prefixed, must walk
            total = 0
            samples = []
            for _ in range(n):
                ln = r("<Q"); f.read(ln)
                if not samples:
                    f_prev = f.tell(); pass
            # too slow for 262k; instead re-walk with capture of first few
            return f"<arr str x{n}>"
        item_sz = {"u8":1,"i8":1,"u16":2,"i16":2,"u32":4,"i32":4,"f32":4,"u64":8,"i64":8,"f64":8}.get(TYPES.get(et), 4)
        f.seek(n * item_sz, 1)
        return f"<arr {TYPES.get(et)} x{n}>"
    raise ValueError(t)

# string arrays: walk properly (needed for vocab sampling later)
def read_val_full(t):
    if t == 9:
        et = r("<I"); n = r("<Q")
        vals = [read_val_full(et) for _ in range(n)]
        return vals
    if t == 8: return read_str()
    return read_val(t)

meta = {}
for _ in range(n_kv):
    k = read_str(); t = r("<I")
    if k in ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type", "tokenizer.ggml.mergs"):
        meta[k] = f"<skipped-array>"
        meta[k] = read_val_full(t) if k.endswith("tokens") and False else read_val(t)
    else:
        meta[k] = read_val_full(t)

tensors = []
for _ in range(n_tensors):
    name = read_str(); nd = r("<I")
    dims = [r("<Q") for _ in range(nd)]
    tt = r("<I"); off = r("<Q")
    tensors.append((name, dims, tt, off))

GGML = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",24:"I8?",30:"BF16"}

print(f"version={ver} n_tensors={n_tensors} n_kv={n_kv}")
print("--- metadata (non-tokenizer) ---")
for k, v in meta.items():
    if isinstance(v, list) and len(v) > 6:
        print(f"{k} = <list x{len(v)}> head={v[:4]}")
    elif isinstance(v, str) and len(v) > 80:
        print(f"{k} = {v[:80]}...")
    else:
        print(f"{k} = {v}")
print("--- tensors ---")
for name, dims, tt, off in tensors:
    print(f"{name}  dims={dims}  type={GGML.get(tt, tt)}  off={off}")
