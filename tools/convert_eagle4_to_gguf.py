"""Convert an EAGLE-4 draft (LlamaForCausalLMEagle4) into the EAGLE3 GGUF our fork loads.

The released draft layout is
    fc.weight, fc_norm.{0,1,2}.weight, norm.weight, embed_tokens.weight, lm_head.weight,
    efficient_lm_head{,_down_proj}.weight, layers.0.{hidden_norm,input_layernorm,
    post_attention_layernorm,self_attn.*,mlp.*}

What we keep: everything the eagle3 graph consumes. Skipped on purpose -
  lm_head / embed_tokens  the graph uses the target's (sharing the vocab), and carrying them
                          would add 5 GB for nothing;
  efficient_lm_head*      EAGLE-4's low-rank candidate head; the full head is the teacher of
                          that head, so dropping it only costs speed, never quality;
  layers.0.hidden_norm    not wired in our eagle3 graph yet.

HF stores linear weights as [out, in]; ggml wants [in, out], so they are transposed here.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "gguf-py"))
from gguf import GGUFWriter  # noqa: E402

COPY = {
    "layers.0.hidden_norm.weight":        "blk.0.attn_norm_2.weight",
    "layers.0.input_layernorm.weight":     "blk.0.attn_norm.weight",
    "layers.0.post_attention_layernorm.weight": "blk.0.ffn_norm.weight",
    "layers.0.self_attn.q_proj.weight":    "blk.0.attn_q.weight",
    "layers.0.self_attn.k_proj.weight":    "blk.0.attn_k.weight",
    "layers.0.self_attn.v_proj.weight":    "blk.0.attn_v.weight",
    "layers.0.self_attn.o_proj.weight":    "blk.0.attn_output.weight",
    "layers.0.mlp.gate_proj.weight":       "blk.0.ffn_gate.weight",
    "layers.0.mlp.up_proj.weight":         "blk.0.ffn_up.weight",
    "layers.0.mlp.down_proj.weight":       "blk.0.ffn_down.weight",
    "fc.weight":                           "fc.weight",
    "fc_norm.0.weight":                    "fc_norm.0.weight",
    "fc_norm.1.weight":                    "fc_norm.1.weight",
    "fc_norm.2.weight":                    "fc_norm.2.weight",
    "norm.weight":                         "output_norm.weight",
}

# 2d tensors are stored [out, in] and have to be transposed for ggml
TRANSPOSE = {k for k in COPY if k.endswith("_proj.weight") or k == "fc.weight"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory holding the HF draft (config.json + model.safetensors)")
    ap.add_argument("out", help="output .gguf")
    ap.add_argument("--tokenizer-from", default=None,
                    help="GGUF to copy tokenizer.* keys from (the target; the draft shares its vocab)")
    args = ap.parse_args()

    src = Path(args.src)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))

    n_embd      = cfg["hidden_size"]
    n_layer     = cfg["num_hidden_layers"]
    n_head      = cfg["num_attention_heads"]
    n_head_kv   = cfg["num_key_value_heads"]
    n_inter     = cfg["intermediate_size"]
    tgt_hidden  = cfg["target_hidden_size"]
    aux_layers  = cfg.get("eagle_config", {}).get("eagle_aux_hidden_state_layer_ids") \
                  or cfg["eagle_aux_hidden_state_layer_ids"]
    n_aux       = len(aux_layers)

    w = GGUFWriter(args.out, "eagle3")
    w.add_name("eagle4-draft")
    w.add_context_length(cfg.get("max_position_embeddings", 262144))
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_inter)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head_kv)
    w.add_layer_norm_rms_eps(float(cfg.get("rms_norm_eps", 1e-6)))
    w.add_rope_freq_base(float(cfg.get("rope_parameters", {}).get("rope_theta", 10000000.0)))
    w.add_target_layers([int(x) for x in aux_layers])
    w.add_target_hidden_size(int(tgt_hidden))
    w.add_uint32("eagle3.vocab_size", int(cfg.get("draft_vocab_size", cfg.get("vocab_size", 0))))
    w.add_uint32("eagle3.attention.key_length", int(cfg.get("head_dim", n_embd // n_head)))
    w.add_uint32("eagle3.attention.value_length", int(cfg.get("head_dim", n_embd // n_head)))
    w.add_bool("eagle3.norm_before_fc", bool(cfg.get("norm_before_fc", False)))
    # eagle4: the reference returns the normed sum as the next step's hidden
    w.add_bool("eagle3.norm_output", bool(cfg.get("norm_output", False)))

    # safetensors/numpy has no bfloat16, so the header is parsed here and bf16 is
    # widened through uint32 (bf16 is the top half of an fp32) before narrowing to fp16.
    raw = (src / "model.safetensors").read_bytes()
    hlen = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + hlen].decode("utf-8"))
    base = 8 + hlen

    def load(name):
        info = header[name]
        dt, shape = info["dtype"], info["shape"]
        b0, b1 = info["data_offsets"]
        buf = raw[base + b0: base + b1]
        if dt == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            return f32.astype(np.float32).reshape(shape)
        if dt == "F16":
            return np.frombuffer(buf, dtype=np.float16).reshape(shape)
        if dt == "F32":
            return np.frombuffer(buf, dtype=np.float32).reshape(shape)
        raise ValueError("unsupported dtype " + dt)

    written = []
    missing = [k for k in COPY if k not in header]
    if missing:
        print("missing tensors:", missing)
        return 2
    for hf_name, gguf_name in COPY.items():
        a = load(hf_name)
        # 1-D tensors are the norms; the graph multiplies them against f32
        # activations, so they stay f32 while the matrices go to f16
        a = np.ascontiguousarray(a.astype(np.float32 if a.ndim == 1 else np.float16))
        # no raw_shape: the writer emits the ggml dims reversed from the numpy shape,
        # which turns HF's [out, in] into ggml's ne = [in, out]
        w.add_tensor(gguf_name, a)
        written.append((gguf_name, a.shape))

    if args.tokenizer_from:
        from gguf import GGUFReader, GGUFValueType
        r = GGUFReader(args.tokenizer_from)
        n_tok = 0
        for key, field in r.fields.items():
            if not key.startswith("tokenizer."):
                continue
            val = field.contents()
            if hasattr(val, "tolist"):
                val = val.tolist()
            is_arr = len(field.types) > 1 and field.types[0] == GGUFValueType.ARRAY
            if is_arr:
                sub = field.types[1]
                if sub == GGUFValueType.STRING:
                    val = [v.decode("utf-8", "replace") if isinstance(v, bytes) else v for v in val]
                w.add_array(key, val)
            else:
                if isinstance(val, bytes):
                    val = val.decode("utf-8", "replace")
                w.add_key_value(key, val, field.types[0])
            n_tok += 1
        print("tokenizer keys copied:", n_tok)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    total = sum(int(np.prod(s)) for _, s in written)
    print("wrote", args.out, "| tensors", len(written), "| params %.1fM" % (total / 1e6))
    for n, s in written:
        print("  %-26s %s" % (n, "x".join(str(x) for x in s)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
