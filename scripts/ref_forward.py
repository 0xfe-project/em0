#!/usr/bin/env python3
"""Numpy reference forward for harrier weights.bin, with variant switches.
Usage: ref_forward.py <weights.bin> <ids comma-sep> [--variant flags...]
Compares against /tmp/oracle.txt if present."""
import numpy as np, sys, struct, math

WB = sys.argv[1]
IDS = [int(x) for x in sys.argv[2].split(",")]
V = set(sys.argv[3:])

# variant flags:
#  norm+1        : add 1.0 to all rmsnorm weights before use
#  rope-halves   : rope pairs (x[i], x[i+128]) instead of adjacent
#  no-attnscale  : attention scale 1.0 (skip 1/sqrt(256) on Q)
#  scale-on-scores: apply 1/sqrt(256) to scores instead of Q
#  gelu-erf      : exact gelu instead of tanh approx
#  causal        : causal mask instead of bidirectional
#  no-embscale   : skip * sqrt(640) on embeddings
#  qknorm-after-rope
#  q-then-rope-only-k? (not needed)
#  norm-half     : norms use weight/2 (? speculative off)

f = open(WB, "rb")
def hdr():
    f.seek(0); b = f.read(128)
    return struct.unpack("<4sI7I3I5fQ2d", b[:92])
(magic, ver, n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff, n_vocab,
 n_swa, swa_pat, _pad, rope_f, rope_s, eps, embscale_f, attnscale_f, total,
 log2f, log2s) = hdr()
assert magic == b"HRW1"

# tensor offsets: replicate the fixed walk
off = {}
p = 128
def take(name, nbytes):
    global p
    p = (p + 15) & ~15
    off[name] = p
    p += nbytes
take("tok_embd", n_vocab*680)
for il in range(n_layer):
    take(f"b{il}.attn_norm", 640*4)
    take(f"b{il}.wq", 1024*680)
    take(f"b{il}.q_norm", 256*4)
    take(f"b{il}.wk", 256*680)
    take(f"b{il}.k_norm", 256*4)
    take(f"b{il}.wv", 256*680)
    take(f"b{il}.wo", 640*1088)
    take(f"b{il}.post_attn", 640*4)
    take(f"b{il}.ffn_norm", 640*4)
    take(f"b{il}.gate", 2048*680)
    take(f"b{il}.up", 2048*680)
    take(f"b{il}.down", 640*2176)
    take(f"b{il}.post_ffn", 640*4)
take("out_norm", 640*4)

def f32(name, n):
    f.seek(off[name]); return np.frombuffer(f.read(4*n), dtype="<f4").astype(np.float32).copy()

def q8(name, rows, k):
    f.seek(off[name]); raw = f.read(rows*(k//32)*34)
    b = np.frombuffer(raw, dtype=np.uint8).reshape(rows, k//32, 34)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)
    qs = b[:, :, 2:34].copy().view(np.int8).astype(np.float32)
    return (d * qs).reshape(rows, k)   # row n = weights for output n

W = {}
for il in range(n_layer):
    for nm in ["wq","wk","wv","wo","gate","up","down"]:
        W[f"b{il}.{nm}"] = None  # lazy

def wq8(name, rows, k):
    if name not in W or W[name] is None:
        W[name] = q8("." + name if name.startswith("b") else name, rows, k)
    return W[name]

def norm_w(name, n):
    w = f32(name, n)
    if "norm+1" in V: w = w + 1.0
    return w

def rmsnorm(x, w, eps=eps):
    s = 1.0/np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)
    return x * s * w

def rope(tab, pos_dim=128):
    # tab: [T,128,2] cos/sin
    return tab

T = len(IDS)
# embeddings
emb = None
xrows = []
for tid in IDS:
    f.seek(off["tok_embd"] + tid * 680)
    row = np.frombuffer(f.read(680), dtype=np.uint8).reshape(20, 34)
    dd = row[:, 0:2].copy().view(np.float16).astype(np.float32)
    qq = row[:, 2:34].copy().view(np.int8).astype(np.float32)
    xrows.append((dd * qq).reshape(640))
x = np.stack(xrows).astype(np.float32)
embscale = 1.0 if "no-embscale" in V else math.sqrt(640.0)
x = x * embscale

# rope tables
def rope_tables(base_log2):
    j = np.arange(128, dtype=np.float64)
    th = np.exp2(-(2*j/256.0) * base_log2)
    a = np.outer(np.arange(T, dtype=np.float64), th)
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)
csf, snf = rope_tables(log2f)
css, sns = rope_tables(log2s)

def apply_rope(v, cs, sn, mode_adjacent=True):
    # v: [T, 256]
    v = v.copy()
    if mode_adjacent:
        a = v.reshape(T, 128, 2)
        x0, x1 = a[..., 0], a[..., 1]
        r0 = x0*cs - x1*sn
        r1 = x0*sn + x1*cs
        return np.stack([r0, r1], axis=-1).reshape(T, 256)
    else:
        x0, x1 = v[:, :128], v[:, 128:]
        r0 = x0*cs - x1*sn
        r1 = x0*sn + x1*cs
        return np.concatenate([r0, r1], axis=1)

attnscale = 1.0 if "no-attnscale" in V else 1.0/math.sqrt(256.0)
adjacent = "rope-halves" not in V

for il in range(n_layer):
    is_swa = (il % 6) < 5
    cs, sn = (css, sns) if is_swa else (csf, snf)
    w_attn_norm = norm_w(f"b{il}.attn_norm", 640)
    h = rmsnorm(x, w_attn_norm)                              # [T,640]
    wq = q8(f"b{il}.wq", 1024, 640); wk = q8(f"b{il}.wk", 256, 640); wv = q8(f"b{il}.wv", 256, 640)
    q = h @ wq.T                                             # [T,1024]
    k = h @ wk.T                                             # [T,256]
    v = h @ wv.T
    if "qknorm-after-rope" in V:
        q = apply_rope(q.reshape(T,4,256).reshape(T,1024), np.tile(cs,(4,1)), np.tile(sn,(4,1)), adjacent) if False else q
    qn = np.empty_like(q)
    for hh in range(4):
        qn[:, hh*256:(hh+1)*256] = rmsnorm(q[:, hh*256:(hh+1)*256], norm_w(f"b{il}.q_norm", 256))
    q = qn
    k = rmsnorm(k, norm_w(f"b{il}.k_norm", 256))
    if "qknorm-after-rope" in V:
        # (rope first then norm) — implement properly:
        pass
    qr = np.empty_like(q)
    for hh in range(4):
        qr[:, hh*256:(hh+1)*256] = apply_rope(q[:, hh*256:(hh+1)*256], cs, sn, adjacent)
    kr = apply_rope(k, cs, sn, adjacent)
    q = qr * attnscale
    if "scale-on-scores" in V:
        q = qr
    # attention per head
    ao = np.empty((T, 1024), dtype=np.float32)
    for hh in range(4):
        sc = q[:, hh*256:(hh+1)*256] @ kr.T                  # [T,T]
        if "scale-on-scores" in V: sc = sc * attnscale
        M = np.zeros((T, T), dtype=bool)
        if "causal" in V:
            M = np.triu(np.ones((T, T), dtype=bool), k=1)
        elif is_swa:
            ii, jj = np.indices((T, T))
            M = np.abs(ii - jj) > n_swa // 2
        sc = sc - sc.max(axis=-1, keepdims=True)
        sc[M] = -1e30
        p = np.exp(sc); p[M] = 0.0
        p = p / p.sum(axis=-1, keepdims=True)
        ao[:, hh*256:(hh+1)*256] = p @ v
    wo = q8(f"b{il}.wo", 640, 1024)
    cur = ao @ wo.T                                          # [T,640]
    cur = rmsnorm(cur, norm_w(f"b{il}.post_attn", 640))
    x = x + cur
    h = rmsnorm(x, norm_w(f"b{il}.ffn_norm", 640))
    gate = h @ q8(f"b{il}.gate", 2048, 640).T
    up   = h @ q8(f"b{il}.up",   2048, 640).T
    if "gelu-erf" in V:
        gact = 0.5*gate*(1+np.vectorize(math.erf)(gate/math.sqrt(2)))
    else:
        gact = 0.5*gate*(1+np.tanh(0.7978845608028654*(gate+0.044715*gate**3)))
    cur = (gact*up) @ q8(f"b{il}.down", 640, 2048).T
    cur = rmsnorm(cur, norm_w(f"b{il}.post_ffn", 640))
    x = x + cur

h = rmsnorm(x[-1], norm_w("out_norm", 640))
emb = h / np.linalg.norm(h)
print(" ".join(f"{v:.7f}" for v in emb))

try:
    o = np.array([float(l) for l in open("/tmp/oracle.txt")])
    cos = float(np.dot(o, emb))
    print(f"# vs oracle: cosine={cos:.6f}", file=sys.stderr)
except FileNotFoundError:
    pass
