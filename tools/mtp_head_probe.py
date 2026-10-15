#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal GGUF header reader: dump the output head + nextn tensors + hadamard-ish metadata.
Answers, for the draft-vocab-shortlist design:
  1. n_vocab (the map header must match this exactly)
  2. which tensor is the draft head and its quant type (must be in draft_vocab_direct's allowlist)
  3. whether nextn.shared_head_head exists (separate head) or the head is model.output
  4. whether nextn.shared_head_head_s exists with >1 element (would silently disable the shortlist)
  5. whether any prism/hadamard metadata is present (would also silently disable it)
"""
import struct, sys, io

PATH = r"F:\super\qwen27b\weights\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"
OUT  = r"F:\super\.workbuddy\tmp\mtp_head_probe.txt"

GGUF_MAGIC = b"GGUF"

# ggml_type -> (name, block_size, type_size)
TYPES = {
    0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
    10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
    16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",
    22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",
    29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0",
}

def read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError("unexpected end of file")
    return b

def r_u32(f): return struct.unpack("<I", read_exact(f, 4))[0]
def r_u64(f): return struct.unpack("<Q", read_exact(f, 8))[0]

def r_str(f):
    n = r_u64(f)
    return read_exact(f, n).decode("utf-8", errors="replace")

def r_val(f, t):
    if t == 0: return struct.unpack("<B", read_exact(f,1))[0]
    if t == 1: return struct.unpack("<b", read_exact(f,1))[0]
    if t == 2: return struct.unpack("<H", read_exact(f,2))[0]
    if t == 3: return struct.unpack("<h", read_exact(f,2))[0]
    if t == 4: return r_u32(f)
    if t == 5: return struct.unpack("<i", read_exact(f,4))[0]
    if t == 6: return struct.unpack("<f", read_exact(f,4))[0]
    if t == 7: return struct.unpack("<B", read_exact(f,1))[0] != 0
    if t == 8: return r_str(f)
    if t == 9:  # array
        et = r_u32(f); n = r_u64(f)
        return [r_val(f, et) for _ in range(n)]
    if t == 10: return r_u64(f)
    if t == 11: return struct.unpack("<q", read_exact(f,8))[0]
    if t == 12: return struct.unpack("<d", read_exact(f,8))[0]
    raise ValueError("unknown gguf value type %d" % t)

def main():
    o = []
    with open(PATH, "rb") as f:
        magic = read_exact(f, 4)
        assert magic == GGUF_MAGIC, magic
        version = r_u32(f)
        n_tensors = r_u64(f)
        n_kv      = r_u64(f)
        o.append("file       : %s" % PATH)
        o.append("gguf ver   : %d" % version)
        o.append("n_tensors  : %d" % n_tensors)
        o.append("n_kv       : %d" % n_kv)
        o.append("")

        kv = {}
        for _ in range(n_kv):
            k = r_str(f)
            t = r_u32(f)
            kv[k] = r_val(f, t)

        tensors = []
        for _ in range(n_tensors):
            name = r_str(f)
            nd   = r_u32(f)
            dims = [r_u64(f) for _ in range(nd)]
            tt   = r_u32(f)
            off  = r_u64(f)
            tensors.append((name, dims, tt, off))

    o.append("=== key metadata ===")
    for k in ("general.architecture", "general.name", "general.file_type"):
        if k in kv:
            o.append("  %-38s = %s" % (k, kv[k]))
    arch = kv.get("general.architecture", "")
    for suffix in ("block_count", "nextn_predict_layers", "embedding_length",
                   "attention.head_count_kv", "attention.key_length", "vocab_size"):
        k = "%s.%s" % (arch, suffix)
        if k in kv:
            o.append("  %-38s = %s" % (k, kv[k]))
    if "tokenizer.ggml.tokens" in kv:
        o.append("  %-38s = %d entries" % ("tokenizer.ggml.tokens", len(kv["tokenizer.ggml.tokens"])))
    if "tokenizer.ggml.merges" in kv:
        o.append("  %-38s = %d entries" % ("tokenizer.ggml.merges", len(kv["tokenizer.ggml.merges"])))
    o.append("")

    o.append("=== prism / hadamard / rotation metadata (would DISABLE the shortlist) ===")
    hits = [k for k in kv if any(s in k.lower() for s in ("prism", "hadamard", "rotation", "rot"))]
    if hits:
        for k in hits:
            v = kv[k]
            o.append("  %-38s = %s" % (k, (str(v)[:70] + "...") if len(str(v)) > 70 else v))
    else:
        o.append("  (none)")
    o.append("")

    # wte / output / nextn tensors
    o.append("=== output + nextn tensors ===")
    def show(name, dims, tt, off):
        o.append("  %-44s %-24s %-8s off=%d" % (name, str(dims), TYPES.get(tt, "type%d" % tt), off))
    for name, dims, tt, off in tensors:
        if name in ("output.weight", "token_embd.weight", "output_norm.weight"):
            show(name, dims, tt, off)
    for name, dims, tt, off in tensors:
        if "nextn" in name:
            show(name, dims, tt, off)
    o.append("")

    o.append("=== all tensor names containing 'head' or 'norm' at top level ===")
    for name, dims, tt, off in tensors:
        if ("head" in name or "norm" in name) and "blk." not in name:
            show(name, dims, tt, off)
    o.append("")

    # the verdict inputs
    n_vocab_from_tok = len(kv.get("tokenizer.ggml.tokens", []))
    head_candidates = [(n, d, t) for n, d, t, _ in tensors if "nextn" in n and "head" in n]
    sep_head = [(n, d, t) for n, d, t in head_candidates if n.endswith("shared_head_head.weight") or n.endswith("nextn.head.weight")]
    scale_t  = [(n, d, t) for n, d, t in head_candidates if n.endswith("_s") or n.endswith("scale")]

    o.append("=== VERDICT INPUTS ===")
    o.append("  tokenizer n_vocab                     = %d" % n_vocab_from_tok)
    o.append("  separate nextn head tensor(s)         = %s" % (sep_head if sep_head else "none (head == model.output)"))
    for n, d, t in sep_head:
        ok = TYPES.get(t, "") in ("Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K",
                                  "IQ2_XXS","IQ2_XS","IQ2_S","IQ3_XXS","IQ3_S","IQ4_NL","IQ4_XS","IQ1_S","IQ1_M")
        o.append("    type=%s  in draft_vocab_direct allowlist = %s" % (TYPES.get(t, t), ok))
    for n, d, t, _ in tensors:
        if n == "output.weight":
            ok = TYPES.get(t, "") in ("Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K",
                                      "IQ2_XXS","IQ2_XS","IQ2_S","IQ3_XXS","IQ3_S","IQ4_NL","IQ4_XS","IQ1_S","IQ1_M")
            o.append("  model.output.weight type=%s  in allowlist = %s  shape=%s" % (TYPES.get(t, t), ok, d))
    o.append("  per-row scale tensor(s) on head       = %s" % (scale_t if scale_t else "none (head_s null or scalar)"))
    o.append("  prism/hadamard metadata present       = %s" % (bool(hits)))

    txt = "\n".join(o) + "\n"
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)

main()
