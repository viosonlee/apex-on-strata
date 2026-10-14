#!/usr/bin/env python3
"""Requantize ONLY the Q3_K expert tensors of a GGUF to Q4_K, streaming: every other tensor is
copied byte-verbatim, and the metadata is passed through unchanged.

    python tools/apex_requant_experts.py --gguf <in.gguf> --out <out.gguf>

Why: Strata's GPU expert kernels (STRATA_GU_FMTS / STRATA_D_FMTS) have no Q3_K template, while
Q4_K is fully covered (expert row-dot, grouped kernel, MMQ prompt path, CPU pool).  Q3_K -> Q4_K
is an upgrade in capacity (3.44 -> 4.5 bpw); the added requantization error is bounded by the
error already baked into the Q3_K source.  The drafter's Q8_0 block and every other tensor are
untouched.
"""
from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as Q  # noqa: E402
from gguf import GGML_QUANT_SIZES  # noqa: E402


def requant_q3k_to_q4k(data: bytes) -> bytes:
    """Q3_K blocks -> f32 -> Q4_K blocks (row-major, rows contiguous)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    f32 = gguf.quants.dequantize(arr, Q.Q3_K).astype(np.float32)
    out = gguf.quants.quantize(f32, Q.Q4_K).view(np.uint8)
    return out.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = open(a.gguf, "rb")
    assert src.read(4) == b"GGUF"
    (ver,) = struct.unpack("<I", src.read(4))
    n_tensors, n_kv = struct.unpack("<QQ", src.read(16))

    def rs(f):
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def rv(f, t):
        if t in (0, 1): return struct.unpack("<b", f.read(1))[0]
        if t in (2, 3): return struct.unpack("<h", f.read(2))[0]
        if t in (4, 5): return struct.unpack("<i", f.read(4))[0]
        if t == 6: return struct.unpack("<f", f.read(4))[0]
        if t == 7: return struct.unpack("<B", f.read(1))[0]
        if t == 8: return rs(f)
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rv(f, et) for _ in range(n)]
        if t in (10, 11): return struct.unpack("<q", f.read(8))[0]
        if t == 12: return struct.unpack("<d", f.read(8))[0]
        raise ValueError(t)

    for _ in range(n_kv):
        k = rs(src)
        t = struct.unpack("<I", src.read(4))[0]
        rv(src, t)
    kv_end = src.tell()

    entries = []            # (name, type_id, dims, offset)
    for _ in range(n_tensors):
        name = rs(src)
        nd = struct.unpack("<I", src.read(4))[0]
        dims = struct.unpack("<" + "Q" * nd, src.read(8 * nd))
        tt, = struct.unpack("<I", src.read(4))
        toff, = struct.unpack("<Q", src.read(8))
        entries.append((name, tt, dims, toff))
    data_start = src.tell()

    # GGUF's alignment is in the metadata (general.alignment, default 32).
    # We rewrite the header ourselves with recomputed offsets, so alignment is ours to keep.
    align = 32
    qk3, ty3 = GGML_QUANT_SIZES[Q.Q3_K]
    qk4, ty4 = GGML_QUANT_SIZES[Q.Q4_K]
    targets = sum(1 for (n, t, d, o) in entries if t == 11 and "_exps" in n)
    print(f"{targets} Q3_K expert tensors -> Q4_K (110 -> 144 B per 256 elems); alignment {align}")

    # ---- write the new file: header, KV verbatim, tensor directory with RECOMPUTED offsets
    dst = open(a.out, "wb")
    src.seek(0)
    header_and_kv = src.read(kv_end)
    dst.write(header_and_kv)

    # tensor directory: name, nd, dims, NEW type, NEW offset
    off = 0
    new_entries = []
    for (name, tt, dims, toff) in entries:
        new_tt = 12 if (tt == 11 and "_exps" in name) else tt
        from gguf import GGML_QUANT_SIZES as GQS
        qk, ty = GQS[Q(new_tt)]
        n = int(np.prod(dims))
        nbytes = n // qk * ty
        pad = (-off) % align
        off += pad
        new_entries.append((name, tt, new_tt, dims, off, nbytes))
        off += nbytes
    for (name, _, new_tt, dims, noff, nbytes) in new_entries:
        dst.write(struct.pack("<Q", len(name.encode())))
        dst.write(name.encode())
        dst.write(struct.pack("<I", len(dims)))
        dst.write(struct.pack("<" + "Q" * len(dims), *dims))
        dst.write(struct.pack("<I", new_tt))
        dst.write(struct.pack("<Q", noff))
    data_start_new = dst.tell()
    if data_start_new % align:
        dst.write(b"\0" * (align - data_start_new % align))
        data_start_new += align - data_start_new % align

    # ---- stream: copy verbatim spans, requantize the Q3_K expert spans
    by_name = {name: (new_tt, dims, noff, nbytes) for (name, _, new_tt, dims, noff, nbytes) in new_entries}
    with open(a.gguf, "rb") as reader:
        for (name, tt, dims, toff) in entries:
            new_tt, _, noff, nbytes = by_name[name]
            qk_old, ty_old = GGML_QUANT_SIZES[Q(tt)]
            n_old = int(np.prod(dims)) // qk_old * ty_old
            reader.seek(data_start + toff)
            raw = reader.read(n_old)
            if tt == 11 and "_exps" in name:
                new = requant_q3k_to_q4k(raw)
                assert len(new) == nbytes, (name, len(new), nbytes)
                dst.write(new)
            else:
                assert len(raw) == nbytes, (name, len(raw), nbytes)
                dst.write(raw)
    src.close()
    dst.close()
    print(f"done -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
