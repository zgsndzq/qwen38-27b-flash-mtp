"""Splice the Qwen3.5-4B MTP head into the 3D-Coder Q8_0 GGUF.

3D-Coder was fine-tuned without the mtp.* tensors - its config still says
mtp_num_hidden_layers=1 but the checkpoint has no such weights - so the
published GGUF carries nextn_predict_layers=0 and has no draft path at all.
The base model does have one MTP layer; this copies it in as blk.32.*, using
the same name mapping convert_hf_to_gguf.py applies, bumps block_count to 33
and appends a False to recurrent_layers.

Two details cost a debugging round each, so they are spelled out:

* The main weights are copied byte for byte (Q8_0 stays Q8_0). The new layer
  arrives as BF16, which the CPU side of the RMSNorm mul rejects -
  "binary_op: unsupported types: dst: f32, src0: f32, src1: bf16". Norms go in
  as F32 and matrices as F16 (same size, more mantissa).
* Qwen3.5 checkpoints store RMSNorm as (w - 1); the converter adds the 1 back,
  which is why the 3D-Coder weights here already average ~1.14. The base
  model's mtp.* norms are still in the offset form (pre_fc_norm_embedding
  averages -0.41), so they need the same +1. Without it the head runs and
  reports 0 accepted out of 1198.

Usage: python tools/fetch_mtp.py   # fills build/tmp/mtp
       python tools/mtp_splice.py  # writes weights/...-MTP-Q8_0.gguf
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # src/
sys.path.insert(0, str(ROOT / "code" / "gguf-py"))
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType  # noqa: E402

SRC = ROOT / "weights" / "Qwen3.5-4B-3D-Coder-Q8_0.gguf"
DST = ROOT / "weights" / "Qwen3.5-4B-3D-Coder-MTP-Q8_0.gguf"
MTP = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "tmp" / "mtp"
N_BLOCKS = 32          # 3D-Coder's own depth; the MTP layer lands on top
REMAP = {              # mtp.<x> -> blk.<n>.nextn.<y>
    "fc": "nextn.eh_proj",
    "pre_fc_norm_embedding": "nextn.enorm",
    "pre_fc_norm_hidden": "nextn.hnorm",
    "norm": "nextn.shared_head_norm",
}
LAYER = {  # mtp.layers.0.<x> -> blk.<n>.<y>
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "post_attention_norm",
    "mlp.down_proj": "ffn_down",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
}


def byte_view(raw, gguf_shape):
    """Raw tensor bytes -> the (rows, bytes_per_row) view add_tensor expects.

    add_tensor runs quant_shape_from_byte_shape when handed uint8, so the last
    dimension has to be the byte count, not the element count.
    """
    ne = [int(x) for x in gguf_shape]
    if len(ne) == 1:
        return raw
    rows = int(np.prod(ne[1:]))
    return raw.reshape(rows, len(raw) // rows)


def main():
    if DST.exists():
        sys.exit(f"{DST} already exists, refusing to overwrite")

    man = json.loads((MTP / "manifest.json").read_text(encoding="utf-8"))
    src = GGUFReader(SRC)
    print(f"source: {len(src.tensors)} tensors, {SRC.stat().st_size / 1e9:.2f} GB")

    out = GGUFWriter(DST, arch="qwen35", use_temp_file=True)

    # ---- metadata: copy, with three edits --------------------------------
    for key, f in src.fields.items():
        if key.startswith("GGUF."):      # version/counts: the writer emits these
            continue
        vtype = f.types[0]
        if vtype == GGUFValueType.ARRAY:
            val = list(f.contents())
            sub = f.types[1]
            if key == "qwen35.attention.recurrent_layers":
                val.append(False)        # the MTP block is a full-attention layer
        else:
            val = f.contents()
            sub = None
            if key == "general.architecture":
                continue                 # GGUFWriter already wrote it
            if key == "qwen35.block_count":
                val = N_BLOCKS + 1
            elif key == "qwen35.nextn_predict_layers":
                val = 1
        out.add_key_value(key, val, vtype, sub)

    # ---- existing tensors, byte for byte ---------------------------------
    for i, t in enumerate(src.tensors):
        raw = src.data[t.data_offset : t.data_offset + t.n_bytes]
        out.add_tensor(t.name, byte_view(raw, t.shape), raw_dtype=t.tensor_type)
        if (i + 1) % 200 == 0:
            print(f"  copied {i + 1}/{len(src.tensors)}", flush=True)

    # ---- the new layer ---------------------------------------------------
    seen = {t.name for t in src.tensors}
    for hf_name, info in sorted(man.items()):
        tail = hf_name[len("mtp."):]
        if tail.endswith(".weight"):
            tail = tail[: -len(".weight")]
        if tail.startswith("layers.0."):
            gguf_name = f"blk.{N_BLOCKS}.{LAYER[tail[len('layers.0.'):]]}.weight"
        else:
            gguf_name = f"blk.{N_BLOCKS}.{REMAP[tail]}.weight"
        if gguf_name in seen:
            sys.exit(f"{gguf_name} already exists in the source")

        raw = (MTP / info["file"]).read_bytes()
        shape = [int(x) for x in info["shape"]]
        f32 = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        if len(shape) == 1:
            out.add_tensor(gguf_name, f32 + 1.0, raw_dtype=GGMLQuantizationType.F32)
        else:
            out.add_tensor(gguf_name, f32.astype(np.float16).reshape(shape),
                           raw_dtype=GGMLQuantizationType.F16)
        print(f"  + {gguf_name:34s} {shape}")

    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file(progress=True)
    out.close()
    print(f"\nwrote {DST}  {DST.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
