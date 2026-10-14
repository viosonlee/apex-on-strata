#!/usr/bin/env python3
"""Extract the APEX-Compact MTP block (blk.40, DeepSeek-style nextn.*) into the Strata
drafter's runtime files (dense.bin / dense.txt / experts.bin), the layout tools/mtp_rt.py
writes for the reference donor.

    python tools/apex_mtp_rt.py --gguf <APEX-MTP-Compact.gguf> --out <dir>

Name mapping (verified against MtpDrafter's geometry self-check, src/core/mtp.cpp):
  fc_embedding.weight                     <- nextn.eh_proj[0:N, :]        (Q8_0)
  fc_hidden.weight                        <- nextn.eh_proj[N:2N, :]       (Q8_0)
  pre_fc_norm_embedding.weight            <- nextn.enorm                  (F32, +1)
  pre_fc_norm_hidden.weight               <- nextn.hnorm                  (F32, +1)
  attn_hyper_connection.hc_norm.weight    <- attn_norm                    (F32, +1; plain reads the first N)
  mlp_hyper_connection.hc_norm.weight     <- post_attention_norm          (F32, +1)
  hyper_connection_mixer.hc_norm.weight   <- nextn.shared_head_norm       (F32, +1)
  self_attn.q/k/v_proj.weight             <- attn_q/k/v.weight            (Q8_0)
  self_attn.o_proj.weight                 <- attn_output.weight           (Q8_0)
  self_attn.q_norm / k_norm.weight        <- attn_q_norm / attn_k_norm    (F32)
  mlp.gate.weight                         <- ffn_gate_inp.weight          (BF16)
  mlp.shared_expert_gate.weight           <- ffn_gate_inp_shexp.weight    (BF16, 1 x N)
  mlp.shared_expert.{gate,up,down}_proj   <- ffn_{gate,up,down}_shexp     (Q8_0, verbatim)
  experts.bin                             <- ffn_{gate,up,down}_exps      (Q8_0, blob layout)
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402

H, FF, NE = 2048, 512, 256          # this body's geometry (hidden, expert FFN, experts)
N_HEAD, HEAD_DIM, N_KV = 16, 256, 2
QK = 64
BB = 18                             # Q2_0 block: 2 fp16-scale + 16 code bytes per 64 weights
BLOB = 3 * (H * FF * BB // QK)      # one expert's blob, matches cpu::BLOB
KQK = 64                            # q8_0 block: 32 values + 2-byte scale


def read_gguf_raw(path: str):
    """Minimal GGUF v3 reader: returns ({name: (type_id, dims, bytes)}, data_start)."""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a GGUF file"
    (ver,) = struct.unpack("<I", f.read(4))
    assert ver == 3, f"GGUF v{ver} not handled"
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    def rs():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def rv(t):
        if t in (0, 1): return struct.unpack("<b", f.read(1))[0]
        if t in (2, 3): return struct.unpack("<h", f.read(2))[0]
        if t in (4, 5): return struct.unpack("<i", f.read(4))[0]
        if t == 6: return struct.unpack("<f", f.read(4))[0]
        if t == 7: return struct.unpack("<B", f.read(1))[0]
        if t == 8: return rs()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rv(et) for _ in range(n)]
        if t in (10, 11): return struct.unpack("<q", f.read(8))[0]
        if t == 12: return struct.unpack("<d", f.read(8))[0]
        raise ValueError(t)

    for _ in range(n_kv):
        k = rs()
        t = struct.unpack("<I", f.read(4))[0]
        rv(t)
    tensors = {}
    off = f.tell()
    for _ in range(n_tensors):
        name = rs()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        tt, = struct.unpack("<I", f.read(4))
        toff, = struct.unpack("<Q", f.read(8))
        tensors[name] = (tt, dims, toff)
    data_start = f.tell()
    payload = f.seek(0, 2) - data_start

    from gguf import GGMLQuantizationType as Q

    def span(name):
        tt, dims, toff = tensors[name]
        from gguf import GGML_QUANT_SIZES
        qk, tySize = GGML_QUANT_SIZES[Q(tt)]
        n = 1
        for d in dims: n *= d
        nbytes = n // qk * tySize
        if toff + nbytes > payload:
            raise MemoryError(f"{name}: runs past the end of the file")
        return tt, dims, data_start + toff, nbytes

    f.close()

    def read_span(name):
        tt, dims, at, nbytes = span(name)
        with open(a_gguf_path, "rb") as g:
            g.seek(at)
            return tt, dims, g.read(nbytes)

    a_gguf_path = path
    out = {}

    class _Lazy:
        __slots__ = ("t", "d", "b")

        def __init__(self, t, d, b): self.t, self.d, self.b = t, d, b

        def __iter__(self):
            return iter((self.t, self.d, self.b))

    for name in tensors:
        out[name] = _Lazy(*read_span(name))
    return out, data_start


def dequant(ttype, dims, data):
    from gguf import GGMLQuantizationType as Q, quants
    arr = np.frombuffer(data, dtype=np.uint8)
    t = quants.dequantize(arr, Q(ttype))
    # GGUF dims are reversed (ne0 contiguous); reshape to (ne1, ne0) row-major
    return t.reshape(int(dims[1]), int(dims[0])) if len(dims) == 2 else t.reshape([int(d) for d in reversed(dims)])
def q8_0(x: np.ndarray) -> bytes:
    b = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def gemma_norm(w: np.ndarray) -> bytes:
    return (np.asarray(w, dtype=np.float32) + 1.0).tobytes()


def q2_0_q(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ggml quantize_row_q2_0_ref over the LAST axis of a (rows, n) f32 array, n a multiple of 64.
    Returns (codes as uint8 (rows, n/4), fp16 scales (rows, n/64))."""
    b = np.ascontiguousarray(x, dtype=np.float32)
    rows, n = b.shape
    assert n % 64 == 0, n
    g = b.reshape(rows, n // 64, 64)
    amax = np.abs(g).max(axis=2)
    d = amax / 3.0                       # ggml: d = amax / (1 << 2 - 1) = amax / 3; Q2_0 id -1..2 -> codes 0..3
    id_ = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = g * id_[:, :, None] + 1.0        # shift by +1: codes = round(x/d) + 1 in [0, 3]
    q = np.rint(v).astype(np.int8)
    np.clip(q, 0, 3, out=q)
    q = q.view(np.uint8)
    # pack 4 codes into 1 byte, little-endian nibbles (ggml order: value i -> nibble i)
    q4 = (q[..., 0::4] | (q[..., 1::4] << 2) | (q[..., 2::4] << 4) | (q[..., 3::4] << 6)).astype(np.uint8)
    scales = d.astype(np.float16)
    return q4.reshape(rows, n // 4), scales.reshape(rows, n // 64)


def q2_0_blob(gu: np.ndarray, dn: np.ndarray) -> bytes:
    """gu: (2*FF, H) f32 interleaved gate/up rows; dn: (H, FF) f32.  -> one Q2_0 blob in the
    drafter's layout: [gate/up codes | down codes | gate/up fp16 scales | down fp16 scales]."""
    ROW_GU, ROW_D, SC_GU, SC_D = H // 4, FF // 4, H // 64, FF // 64
    gu_c, gu_s = q2_0_q(gu)
    dn_c, dn_s = q2_0_q(dn)
    o_d_codes = 2 * FF * ROW_GU
    o_gu_scales = o_d_codes + H * ROW_D
    blob = bytearray(2 * FF * ROW_GU + H * ROW_D + 2 * FF * SC_GU * 2 + H * SC_D * 2)
    blob[0:o_d_codes] = gu_c.tobytes()
    blob[o_d_codes:o_gu_scales] = dn_c.tobytes()
    blob[o_gu_scales:o_gu_scales + 2 * FF * SC_GU * 2] = gu_s.tobytes()
    blob[o_gu_scales + 2 * FF * SC_GU * 2:] = dn_s.tobytes()
    assert len(blob) == BLOB, (len(blob), BLOB)
    return bytes(blob)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors, _ = read_gguf_raw(a.gguf)

    B = "blk.40."
    eh = dequant(*tensors[B + "nextn.eh_proj.weight"])       # (2N, N) rows
    assert eh.shape == (H, 2 * H), eh.shape
    lines, off = [], 0
    dense = open(out / "dense.bin", "wb")

    def put(name, raw, kind, rows, cols):
        nonlocal off
        pad = (-off) % 256
        if pad: dense.write(b"\0" * pad); off += pad
        dense.write(raw)
        lines.append(f"{name} {kind} {rows} {cols} {off} {len(raw)}")
        off += len(raw)

    # ---- the two fc projections: eh_proj is (out N, in 2N); the input is [emb_norm | h_norm] concatenated,
    # so the LEFT half of the columns is the embedding branch and the RIGHT half the hidden branch.
    put("fc_embedding.weight", q8_0(eh[:, :H]), "q8_0", H, H)
    put("fc_hidden.weight", q8_0(eh[:, H:]), "q8_0", H, H)
    put("pre_fc_norm_embedding.weight", gemma_norm(dequant(*tensors[B + "nextn.enorm.weight"])), "f32", 1, H)
    put("pre_fc_norm_hidden.weight", gemma_norm(dequant(*tensors[B + "nextn.hnorm.weight"])), "f32", 1, H)
    put("attn_hyper_connection.hc_norm.weight", gemma_norm(dequant(*tensors[B + "attn_norm.weight"])), "f32", 1, H)
    put("mlp_hyper_connection.hc_norm.weight", gemma_norm(dequant(*tensors[B + "post_attention_norm.weight"])), "f32", 1, H)
    put("hyper_connection_mixer.hc_norm.weight", gemma_norm(dequant(*tensors[B + "nextn.shared_head_norm.weight"])), "f32", 1, H)
    put("self_attn.q_proj.weight", q8_0(dequant(*tensors[B + "attn_q.weight"])), "q8_0", N_HEAD * 2 * HEAD_DIM, H)
    put("self_attn.k_proj.weight", q8_0(dequant(*tensors[B + "attn_k.weight"])), "q8_0", N_KV * HEAD_DIM, H)
    put("self_attn.v_proj.weight", q8_0(dequant(*tensors[B + "attn_v.weight"])), "q8_0", N_KV * HEAD_DIM, H)
    put("self_attn.o_proj.weight", q8_0(dequant(*tensors[B + "attn_output.weight"])), "q8_0", H, N_HEAD * HEAD_DIM)
    put("self_attn.q_norm.weight", dequant(*tensors[B + "attn_q_norm.weight"]).astype(np.float32).tobytes(), "f32", 1, HEAD_DIM)
    put("self_attn.k_norm.weight", dequant(*tensors[B + "attn_k_norm.weight"]).astype(np.float32).tobytes(), "f32", 1, HEAD_DIM)
    put("mlp.gate.weight", dequant(*tensors[B + "ffn_gate_inp.weight"]).view(np.uint16).tobytes(), "bf16", NE, H)
    put("mlp.shared_expert_gate.weight", dequant(*tensors[B + "ffn_gate_inp_shexp.weight"]).view(np.uint16).tobytes(), "bf16", 1, H)
    put("mlp.shared_expert.gate_proj.weight", tensors[B + "ffn_gate_shexp.weight"].b, "q8_0", H, FF)
    put("mlp.shared_expert.up_proj.weight", tensors[B + "ffn_up_shexp.weight"].b, "q8_0", H, FF)
    put("mlp.shared_expert.down_proj.weight", tensors[B + "ffn_down_shexp.weight"].b, "q8_0", FF, H)
    dense.close()

    # ---- the 256 routed experts, blob layout: gate/up rows interleaved, then down.
    gu_f = dequant(*tensors[B + "ffn_gate_exps.weight"])          # (NE, 2*FF, H) f32 (gate rows 0..FF-1, up FF..)
    up_f = dequant(*tensors[B + "ffn_up_exps.weight"])
    dn_f = dequant(*tensors[B + "ffn_down_exps.weight"])          # (NE, H, FF)
    assert gu_f.shape == (NE, FF, H) and dn_f.shape == (NE, H, FF), (gu_f.shape, dn_f.shape)
    # combine gate+up into the interleaved (2*FF, H) the blob wants
    gu2 = np.empty((NE, 2 * FF, H), dtype=np.float32)
    gu2[:, 0::2] = gu_f
    gu2[:, 1::2] = up_f
    blob_rows = []
    for e in range(NE):
        blob_rows.append(q2_0_blob(gu2[e], dn_f[e]))
    (out / "experts.bin").write_bytes(b"".join(blob_rows))
    experts_bytes = len(blob_rows[0]) * NE

    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"experts.bin {experts_bytes} B (NE={NE}, per-expert {experts_bytes // NE}), dense.bin {off} B, "
          f"{len(lines)} tensors -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
