#!/usr/bin/env python3
"""Extract the APEX-Compact MTP block (blk.40, DeepSeek-style nextn.*) into the Strata
drafter's runtime files (dense.bin / dense.txt / experts.bin), the layout tools/mtp_rt.py
writes for the reference donor.

    python tools/apex_mtp_rt.py --gguf <APEX-MTP-Compact.gguf> --out <dir>

Name mapping (verified against MtpDrafter's geometry self-check, src/core/mtp.cpp, and llama.cpp's
qwen35moe.cpp nextn forward @ 3cf03257f):
  fc_embedding.weight                     <- nextn.eh_proj[:, 0:N]        (Q8_0; concat is [e_norm | h_norm])
  fc_hidden.weight                        <- nextn.eh_proj[:, N:2N]       (Q8_0)
  pre_fc_norm_embedding.weight            <- nextn.enorm                  (F32, PLAIN RMSNorm - no Gemma '+1')
  pre_fc_norm_hidden.weight               <- nextn.hnorm                  (F32, PLAIN RMSNorm)
  attn_hyper_connection.hc_norm.weight    <- attn_norm                    (F32, PLAIN; plain reads the first N)
  mlp_hyper_connection.hc_norm.weight     <- post_attention_norm          (F32, PLAIN)
  hyper_connection_mixer.hc_norm.weight   <- nextn.shared_head_norm       (F32, PLAIN)
  self_attn.q/k/v_proj.weight             <- attn_q/k/v.weight            (Q8_0)
  self_attn.o_proj.weight                 <- attn_output.weight           (Q8_0)
  self_attn.q_norm / k_norm.weight        <- attn_q_norm / attn_k_norm    (F32)
  mlp.gate.weight                         <- ffn_gate_inp.weight          (BF16, real F32->BF16 RNE)
  mlp.shared_expert_gate.weight           <- ffn_gate_inp_shexp.weight    (BF16, 1 x N)
  mlp.shared_expert.{gate,up,down}_proj   <- ffn_{gate,up,down}_shexp     (Q8_0, verbatim)
  experts.bin                             <- ffn_{gate,up,down}_exps      (Q8_0 -> Q2_0 blob, per expert)
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
    """Minimal GGUF v3 reader: index only.  Returns ({name: (type_id, dims, file_offset)}, data_start).
    NOTHING is read into memory here - the callers read the spans they need through `open_span`.
    (An earlier version read all 16 GB to serve 3 tensors; on a 24 GB machine that alone could OOM.)"""
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
    for _ in range(n_tensors):
        name = rs()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        tt, = struct.unpack("<I", f.read(4))
        toff, = struct.unpack("<Q", f.read(8))
        tensors[name] = (tt, dims, toff)
    data_start = f.tell()
    payload = f.seek(0, 2) - data_start
    f.close()

    from gguf import GGMLQuantizationType as Q, GGML_QUANT_SIZES

    def tensor_bytes(tt, dims):
        qk, tySize = GGML_QUANT_SIZES[Q(tt)]
        n = 1
        for d in dims:
            n *= d
        return n // qk * tySize

    def open_span(name):
        """Returns a context manager yielding (type_id, dims, file-object positioned at the data)."""
        tt, dims, toff = tensors[name]
        nbytes = tensor_bytes(tt, dims)
        if toff + nbytes > payload:
            raise MemoryError(f"{name}: runs past the end of the file")

        class Span:
            def __enter__(self):
                self.f = open(path, "rb")
                self.f.seek(data_start + toff)
                return tt, dims, self.f

            def __exit__(self, *exc):
                self.f.close()
                return False

        return Span()

    return tensors, tensor_bytes, open_span


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


def norm_plain(w: np.ndarray) -> bytes:
    """Qwen3.6 MTP norms are PLAIN RMSNorm weights (llama.cpp LLM_NORM_RMS: y = w * norm(x)).
    The reference donor (Qwen3.8-Flash-Next checkpoint) needed the vLLM Gemma '+1'; this body does NOT."""
    return np.asarray(w, dtype=np.float32).tobytes()


def to_bf16(f32: np.ndarray) -> np.ndarray:
    """F32 -> BF16 with round-to-nearest-even (ggml's fp32_to_bf16).  A .view() is NOT a conversion:
    it reinterprets the bytes, so [1.0, 2.0] f32 would read back as [0, 1, 0, 2] - twice the values,
    twice the bytes.  RNE: round the low 16 bits into bit 16, on ties to even."""
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    # NaN/Inf keep their payload; finite values round the low half into the high half.
    rounded = (u >> 16) & 1
    u = (u >> 16) + rounded
    return u.astype(np.uint16)


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
    tensors, tensor_bytes, open_span = read_gguf_raw(a.gguf)

    B = "blk.40."
    with open_span(B + "nextn.eh_proj.weight") as (tt, dims, f):
        eh = dequant(tt, dims, f.read(tensor_bytes(tt, dims)))   # (N, 2N) rows
    assert eh.shape == (H, 2 * H), eh.shape
    lines, off = [], 0
    # write to temporary names and rename at the end: an interrupted run must not leave a dense.bin
    # that reads as finished (the engine refuses dense.bin without dense.txt, but be tidy anyway)
    dense_path, dense_tmp = out / "dense.bin", out / "dense.bin.tmp"
    dense = open(dense_tmp, "wb")

    def put(name, raw, kind, rows, cols):
        nonlocal off
        pad = (-off) % 256
        if pad: dense.write(b"\0" * pad); off += pad
        dense.write(raw)
        lines.append(f"{name} {kind} {rows} {cols} {off} {len(raw)}")
        off += len(raw)

    def deq(name):
        with open_span(name) as (tt, dims, f):
            return dequant(tt, dims, f.read(tensor_bytes(tt, dims)))

    # ---- the two fc projections: eh_proj is (out N, in 2N); the input is [emb_norm | h_norm] concatenated,
    # so the LEFT half of the columns is the embedding branch and the RIGHT half the hidden branch.
    put("fc_embedding.weight", q8_0(eh[:, :H]), "q8_0", H, H)
    put("fc_hidden.weight", q8_0(eh[:, H:]), "q8_0", H, H)
    put("pre_fc_norm_embedding.weight", norm_plain(deq(B + "nextn.enorm.weight")), "f32", 1, H)
    put("pre_fc_norm_hidden.weight", norm_plain(deq(B + "nextn.hnorm.weight")), "f32", 1, H)
    put("attn_hyper_connection.hc_norm.weight", norm_plain(deq(B + "attn_norm.weight")), "f32", 1, H)
    put("mlp_hyper_connection.hc_norm.weight", norm_plain(deq(B + "post_attention_norm.weight")), "f32", 1, H)
    put("hyper_connection_mixer.hc_norm.weight", norm_plain(deq(B + "nextn.shared_head_norm.weight")), "f32", 1, H)
    put("self_attn.q_proj.weight", q8_0(deq(B + "attn_q.weight")), "q8_0", N_HEAD * 2 * HEAD_DIM, H)
    put("self_attn.k_proj.weight", q8_0(deq(B + "attn_k.weight")), "q8_0", N_KV * HEAD_DIM, H)
    put("self_attn.v_proj.weight", q8_0(deq(B + "attn_v.weight")), "q8_0", N_KV * HEAD_DIM, H)
    put("self_attn.o_proj.weight", q8_0(deq(B + "attn_output.weight")), "q8_0", H, N_HEAD * HEAD_DIM)
    put("self_attn.q_norm.weight", deq(B + "attn_q_norm.weight").astype(np.float32).tobytes(), "f32", 1, HEAD_DIM)
    put("self_attn.k_norm.weight", deq(B + "attn_k_norm.weight").astype(np.float32).tobytes(), "f32", 1, HEAD_DIM)
    put("mlp.gate.weight", to_bf16(deq(B + "ffn_gate_inp.weight")).tobytes(), "bf16", NE, H)
    put("mlp.shared_expert_gate.weight", to_bf16(deq(B + "ffn_gate_inp_shexp.weight")).tobytes(), "bf16", 1, H)
    # the shared expert projections are ALREADY Q8_0 in the GGUF: copy verbatim
    put("mlp.shared_expert.gate_proj.weight", read_raw(open_span, B + "ffn_gate_shexp.weight"), "q8_0", H, FF)
    put("mlp.shared_expert.up_proj.weight", read_raw(open_span, B + "ffn_up_shexp.weight"), "q8_0", H, FF)
    put("mlp.shared_expert.down_proj.weight", read_raw(open_span, B + "ffn_down_shexp.weight"), "q8_0", FF, H)
    dense.close()

    # ---- the routed experts, blob layout: gate/up rows interleaved, then down.  ONE EXPERT AT A TIME
    # (2 MiB working set per expert instead of ~2 GiB of f32 for all 256): peak memory stays flat on a
    # 24 GB machine even with the OS file cache cold.
    experts_tmp = out / "experts.bin.tmp"
    n_done = 0
    with open_span(B + "ffn_gate_exps.weight") as (t_g, d_g, f_g), \
         open_span(B + "ffn_up_exps.weight") as (t_u, d_u, f_u), \
         open_span(B + "ffn_down_exps.weight") as (t_d, d_d, f_d):
        with open(experts_tmp, "wb") as fo:
            per_gu = tensor_bytes(t_g, d_g) // NE      # one expert's Q8_0 bytes (gate and up are the same size)
            per_dn = tensor_bytes(t_d, d_d) // NE
            for e in range(NE):
                # dequant with the PER-EXPERT dims: the GGUF dims (H, FF, NE) slice to (H, FF) per expert
                gu_f = dequant(t_g, (d_g[0], d_g[1]), f_g.read(per_gu))
                up_f = dequant(t_u, (d_u[0], d_u[1]), f_u.read(per_gu))
                dn_f = dequant(t_d, (d_d[0], d_d[1]), f_d.read(per_dn))
                assert gu_f.shape == (FF, H) and dn_f.shape == (H, FF), (gu_f.shape, dn_f.shape)
                gu2 = np.empty((2 * FF, H), dtype=np.float32)
                gu2[0::2] = gu_f
                gu2[1::2] = up_f
                fo.write(q2_0_blob(gu2, dn_f))
                n_done += 1
                if n_done % 32 == 0:
                    print(f"  expert {n_done}/{NE}", flush=True)
    (dense_tmp).replace(dense_path)
    experts_tmp.replace(out / "experts.bin")
    experts_bytes = BLOB * NE

    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"experts.bin {experts_bytes} B (NE={NE}, per-expert {experts_bytes // NE}), dense.bin {off} B, "
          f"{len(lines)} tensors -> {out}")
    return 0


def read_raw(open_span, name) -> bytes:
    with open_span(name) as (tt, dims, f):
        from gguf import GGMLQuantizationType as Q, GGML_QUANT_SIZES
        qk, ty = GGML_QUANT_SIZES[Q(tt)]
        n = 1
        for d in dims: n *= d
        return f.read(n // qk * ty)


if __name__ == "__main__":
    sys.exit(main())

