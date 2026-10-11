"""tok.py -- tokenization layer for the shortlist pipeline.

Two backends, because they answer different needs:

  ``service`` (authoritative)
      POST /tokenize on a running llama-server. This is the model's own tokenizer
      so the token ids are exactly what the draft head will have to emit.

  ``offline`` (approximate, no GPU required)
      Read ``tokenizer.ggml.tokens`` / ``.merges`` / ``.token_types`` straight out
      of the GGUF with the reader from tools/mtp_head_probe.py, then run byte-level
      BPE (GPT-2 style, merge-rank driven) locally.

      It is an approximation and it is labelled as one everywhere it surfaces:
      llama.cpp applies a per-family pre-tokenizer regex before the merges, and
      that regex is not reproduced here. Blocks are instead cut with a generic
      ``\\s*\\S+|\\s+`` rule, which keeps the leading-space forms (`` def``,
      `` return``) that matter for code but may also allow a merge or two across
      a digit/punctuation boundary that the real regex would have blocked. Treat
      an offline coverage curve as a preview: good enough to kill a bad idea,
      not good enough to certify a good one. When both backends have run, the
      build step reports their agreement so the approximation is quantified
      rather than assumed.

``auto`` (default) uses the service when one is reachable and falls back to
offline with a loud note.

Output: one JSON per corpus sample in ``corpus/ids/``, storing the token ids per
field (``reasoning`` / ``answer`` / ``text``). Keeping the fields separate lets
the analysis slice "code only" vs "including the thinking block" without
re-tokenizing anything.
"""

from __future__ import annotations

import json
import re
import struct
import unicodedata
from pathlib import Path

VOCAB_DIR = Path(__file__).resolve().parent
CORPUS = VOCAB_DIR / "corpus"
IDS_DIR = CORPUS / "ids"
DEFAULT_GGUF = VOCAB_DIR.parent / "models" / "Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"

# fields to tokenize per corpus source, in order
FIELDS = {"gen": ("reasoning", "answer"), "real": ("reasoning", "answer"),
          "local": ("text",)}

# --------------------------------------------------------------------- GGUF IO
_GGUF_TYPES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
               7: "bool", 8: "str", 9: "array", 10: "u64", 11: "i64", 12: "f64"}


def _r_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError("unexpected end of file while reading GGUF header")
    return b


def _r_val(f, t: int, want_tokens: bool):
    """Read one GGUF value. ``want_tokens`` skips materialising other big arrays."""
    if t in (0, 1, 7):
        return _r_exact(f, 1)[0]
    if t in (2, 3):
        return struct.unpack("<H" if t == 2 else "<h", _r_exact(f, 2))[0]
    if t == 4:
        return struct.unpack("<I", _r_exact(f, 4))[0]
    if t == 5:
        return struct.unpack("<i", _r_exact(f, 4))[0]
    if t == 6:
        return struct.unpack("<f", _r_exact(f, 4))[0]
    if t == 8:
        n = struct.unpack("<Q", _r_exact(f, 8))[0]
        return _r_exact(f, n).decode("utf-8", errors="replace")
    if t == 10:
        return struct.unpack("<Q", _r_exact(f, 8))[0]
    if t == 11:
        return struct.unpack("<q", _r_exact(f, 8))[0]
    if t == 12:
        return struct.unpack("<d", _r_exact(f, 8))[0]
    if t == 9:
        et = struct.unpack("<I", _r_exact(f, 4))[0]
        n = struct.unpack("<Q", _r_exact(f, 8))[0]
        if et == 8 and n > 1000:
            out = []
            for _ in range(n):
                ln = struct.unpack("<Q", _r_exact(f, 8))[0]
                out.append(_r_exact(f, ln).decode("utf-8", errors="replace"))
            return out
        return [_r_val(f, et, want_tokens) for _ in range(n)]
    raise ValueError("unknown GGUF value type %d" % t)


WANT_KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges",
             "tokenizer.ggml.token_type", "tokenizer.ggml.model",
             "tokenizer.ggml.pre", "general.architecture", "general.name",
             "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id")


def read_gguf_meta(path, want_keys=WANT_KEYS) -> dict:
    """Read selected header keys. Stops before the tensor data, so it is fast."""
    with open(path, "rb") as f:
        if _r_exact(f, 4) != b"GGUF":
            raise ValueError("%s is not a GGUF file" % path)
        _version = struct.unpack("<I", _r_exact(f, 4))[0]
        n_tensors, n_kv = struct.unpack("<QQ", _r_exact(f, 16))
        out = {}
        for _ in range(n_kv):
            key = _r_val(f, 8, True)
            t = struct.unpack("<I", _r_exact(f, 4))[0]
            if key in want_keys:
                out[key] = _r_val(f, t, True)
            else:
                _r_val(f, t, False)
        out["_n_tensors"] = n_tensors
        return out


# ------------------------------------------------------- byte-level BPE (offline)
def _bytes_to_unicode() -> dict:
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) +
          list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_ENCODER_MAP = _bytes_to_unicode()
BYTE_ENCODER = _BYTE_ENCODER_MAP                        # public: reused by shortlist.py
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

# generic block cutter: optional leading whitespace belongs to the following word,
# so "\\n    def" yields the block "\\n    def" rather than a bare "\\n".
_BLOCK_RE = re.compile(r"\s*\S+|\s+")
_MAX_BLOCK = 64          # guard against pathological no-whitespace runs

# ---------------------------------------------------------- pre-tokenizers
# ``tokenizer.ggml.pre`` selects a regex that llama.cpp applies before the BPE
# merges. The family for this model is "qwen35", defined in
# src/llama-vocab.cpp (LLAMA_VOCAB_PRE_TYPE_QWEN35) as, alternatives in order:
#
#   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
#   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
#   |\p{N}
#   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
#   |\s*[\r\n]+
#   |\s+(?!\S)
#   |\s+
#
# It is reproduced here by hand rather than by importing a Unicode-aware regex
# engine, because the environment has no PyPI access. Two behaviours differ from a
# naive whitespace split and both matter for code:
#   * \p{N} matches a SINGLE numeric character, so "10" is two tokens' worth of
#     input, not one;
#   * an optional leading punctuation/space char is absorbed into a following
#     letter run, so " def" and "(def" stay whole.
_ASCII_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_ASCII_DIGITS = frozenset("0123456789")
_CAT_CACHE: dict = {}


def _kind(ch: str) -> str:
    """'L' letter, 'M' mark, 'N' number, '.' anything else (Unicode-aware)."""
    if ch in _ASCII_LETTERS:
        return "L"
    if ch in _ASCII_DIGITS:
        return "N"
    if ord(ch) < 128:
        return "."
    k = _CAT_CACHE.get(ch)
    if k is None:
        k = unicodedata.category(ch)[0]
        _CAT_CACHE[ch] = k
    return k


def scan_qwen35(text: str) -> list[str]:
    """Exact left-to-right scan of the alternation above, first match wins."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        # 1. English contractions, case-insensitive
        if ch == "'" and i + 1 < n:
            if text[i:i + 2].lower() in ("'s", "'t", "'m", "'d"):
                out.append(text[i:i + 2])
                i += 2
                continue
            if text[i:i + 3].lower() in ("'re", "'ve", "'ll"):
                out.append(text[i:i + 3])
                i += 3
                continue
        # 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        j = i + 1 if (ch not in "\r\n" and _kind(ch) not in ("L", "N")) else i
        if j < n and _kind(text[j]) in ("L", "M"):
            k = j
            while k < n and _kind(text[k]) in ("L", "M"):
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # 3. \p{N} -- exactly one numeric character
        if _kind(ch) == "N":
            out.append(ch)
            i += 1
            continue
        # 4. " ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
        j = i + 1 if ch == " " else i
        k = j
        while k < n and not text[k].isspace() and _kind(text[k]) not in ("L", "M", "N"):
            k += 1
        if k > j:
            e = k
            while e < n and text[e] in "\r\n":
                e += 1
            out.append(text[i:e])
            i = e
            continue
        # 5. \s*[\r\n]+   (greedy \s* then backtracking to leave the line breaks)
        j = i
        while j < n and text[j].isspace() and text[j] not in "\r\n":
            j += 1
        if j < n and text[j] in "\r\n":
            k = j
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # 6. \s+(?!\S) -- whitespace that runs to the end of the text
        k = i
        while k < n and text[k].isspace():
            k += 1
        if k > i and k >= n:
            out.append(text[i:k])
            i = k
            continue
        # 7. \s+
        if k > i:
            out.append(text[i:k])
            i = k
            continue
        out.append(ch)          # safety: guarantees forward progress
        i += 1
    return out


def scan_generic(text: str) -> list[str]:
    """Fallback cutter for model families with no reproduced pre-tokenizer."""
    return _BLOCK_RE.findall(text)


SCANNERS = {"qwen35": scan_qwen35}


class OfflineBpe:
    """GPT-2 style byte-level BPE driven by the model's own merge ranks."""

    def __init__(self, gguf_path=None):
        meta = read_gguf_meta(gguf_path or DEFAULT_GGUF)
        self.model_type = meta.get("tokenizer.ggml.model", "?")
        self.pre = meta.get("tokenizer.ggml.pre", "?")
        self.tokens = meta.get("tokenizer.ggml.tokens") or []
        if not self.tokens:
            raise ValueError("GGUF has no tokenizer.ggml.tokens")
        self.n_vocab = len(self.tokens)
        self.token_types = meta.get("tokenizer.ggml.token_type") or [1] * self.n_vocab
        self.merges = meta.get("tokenizer.ggml.merges") or []

        self.piece_to_id = {}
        for i, s in enumerate(self.tokens):
            self.piece_to_id.setdefault(s, i)

        # byte symbol -> its single-token id (the BPE starting alphabet)
        self.byte_id = {}
        for b in range(256):
            ch = BYTE_ENCODER[b]
            i = self.piece_to_id.get(ch)
            if i is not None:
                self.byte_id[b] = i

        # merge tables keyed by token id pairs
        self.rank = {}
        self.merged = {}
        for r, m in enumerate(self.merges):
            if not isinstance(m, str) or " " not in m:
                continue
            a, b = m.split(" ", 1)
            ia, ib = self.piece_to_id.get(a), self.piece_to_id.get(b)
            if ia is None or ib is None:
                continue
            iab = self.piece_to_id.get(a + b)
            if iab is None:
                continue
            self.rank[(ia, ib)] = r
            self.merged[(ia, ib)] = iab

        # special tokens (CONTROL=3 / USER_DEFINED=4 / UNKNOWN=2) matched verbatim
        specials = [s for i, s in enumerate(self.tokens)
                    if i < len(self.token_types) and self.token_types[i] in (2, 3, 4) and s]
        specials.sort(key=len, reverse=True)
        self._special_re = (re.compile("(" + "|".join(re.escape(s) for s in specials[:400]) + ")")
                            if specials else None)

        # pre-tokenizer: exact when this family has been reproduced, else approximate
        self._scanner = SCANNERS.get(self.pre, scan_generic)
        self.exact_pre = self.pre in SCANNERS
        self._memo: dict = {}          # block -> token ids, see _bpe()

    # ------------------------------------------------------------------ helpers
    def piece(self, i: int) -> str:
        """Human-readable form of a token id (byte-level decoded)."""
        if not (0 <= i < self.n_vocab):
            return "<oob:%d>" % i
        s = self.tokens[i]
        try:
            return bytes(BYTE_DECODER[ch] for ch in s).decode("utf-8", errors="replace")
        except KeyError:
            return s

    def _bpe(self, block: str) -> list[int]:
        # Memoised: code text reuses a small vocabulary of blocks (" ", "def",
        # "self", "\n    ") thousands of times, so the merge loop almost always
        # hits. This is the difference between mining 84 MB in three minutes and
        # in well under one -- which matters because a background job here can be
        # reaped at the two minute mark.
        hit = self._memo.get(block)
        if hit is not None:
            return hit
        syms = [self.byte_id[b] for b in block.encode("utf-8") if b in self.byte_id]
        while len(syms) > 1:
            best_rank, best_i = None, -1
            for i in range(len(syms) - 1):
                r = self.rank.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            syms = syms[:best_i] + [self.merged[(syms[best_i], syms[best_i + 1])]] + syms[best_i + 2:]
        if len(self._memo) < 2_000_000:
            self._memo[block] = syms
        return syms

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        if self._special_re is not None:
            parts = self._special_re.split(text)
        else:
            parts = [text]
        for part in parts:
            if not part:
                continue
            sp = self.piece_to_id.get(part)
            if sp is not None and self._special_re is not None and self._special_re.fullmatch(part):
                ids.append(sp)
                continue
            for block in self._scanner(part):
                if len(block) <= _MAX_BLOCK:
                    ids.extend(self._bpe(block))
                else:
                    for k in range(0, len(block), _MAX_BLOCK):
                        ids.extend(self._bpe(block[k:k + _MAX_BLOCK]))
        return ids


# ------------------------------------------------------------------- pipeline
def _samples():
    """Yield (source, label, path) for every corpus sample on disk."""
    for source, fields in FIELDS.items():
        d = CORPUS / source
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("_"):
                continue
            yield source, p.stem, p


def cmd(args) -> int:
    IDS_DIR.mkdir(parents=True, exist_ok=True)
    samples = list(_samples())
    only = getattr(args, "only", None)
    if only:
        want = {s.strip() for s in only.split(",") if s.strip()}
        samples = [s for s in samples if s[0] in want]
    if not samples:
        print("no corpus samples found; run 'local' or 'gen' first")
        return 1

    backend = args.backend
    srv_obj = offline = None
    if backend in ("auto", "service"):
        import srv
        try:
            srv_obj = srv.Server(port=args.port, bat=args.bat,
                                 attach_only=(backend == "service"),
                                 log=VOCAB_DIR / "out" / "server_tok.log").ensure()
            backend = "service"
            print("tokenizer: service  (%s)" % srv_obj.attach_why)
        except Exception as e:
            if backend == "service":
                print("service backend requested but unavailable: %s" % e)
                return 1
            print("tokenizer: service unavailable (%s); falling back to offline" % e)
            backend = "offline"
    if backend == "offline":
        offline = OfflineBpe(args.gguf)
        print("tokenizer: offline-approx  (ggml.model=%s pre=%s n_vocab=%d merges=%d)"
              % (offline.model_type, offline.pre, offline.n_vocab, len(offline.merges)))
        print("           NOTE: approximate (no pre-tokenizer regex). Preview only.")

    encoder = srv_obj.tokenize if backend == "service" else offline.encode

    todo = []
    for source, label, path in samples:
        out = IDS_DIR / ("%s__%s.json" % (source, label))
        if out.exists() and out.stat().st_mtime >= path.stat().st_mtime and not args.overwrite:
            continue
        todo.append((source, label, path, out))
    print("samples=%d  already tokenized=%d  pending=%d"
          % (len(samples), len(samples) - len(todo), len(todo)))
    if not todo:
        return 0
    if args.limit:
        todo = todo[:args.limit]

    try:
        for i, (source, label, path, out) in enumerate(todo, 1):
            rec = json.loads(path.read_text(encoding="utf-8"))
            chunks, total = {}, 0
            for field in FIELDS[source]:
                text = rec.get(field) or ""
                ids = encoder(text) if text else []
                chunks[field] = {"n": len(ids), "ids": ids}
                total += len(ids)
            payload = {"label": rec.get("label", label), "source": source,
                       "group": rec.get("group", "?"), "backend": backend,
                       "n_tokens": total, "chunks": chunks}
            out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            print("  [%3d/%3d] %-46s tokens=%-6d %s"
                  % (i, len(todo), "%s__%s" % (source, label), total,
                     " ".join("%s=%d" % (k, v["n"]) for k, v in chunks.items())))
    finally:
        if srv_obj is not None:
            srv_obj.cleanup()
    print("tokenized %d sample(s) -> %s" % (len(todo), IDS_DIR))
    return 0


def consistency(n: int = 40, port: int = 18080, gguf=None) -> int:
    """Compare the offline tokenizer against the live one on identical text.

    The point is to quantify the approximation instead of trusting it: the offline
    backend reimplements the pre-tokenizer by hand, and the only way to know it is
    right is to diff it against the model's own /tokenize on real samples.
    Reports a per-sample agreement rate plus the first few divergences.
    """
    import srv as _srv
    off = OfflineBpe(gguf)
    print("offline : pre=%s exact_pre=%s merges=%d" % (off.pre, off.exact_pre, len(off.merges)))
    samples = list(_samples())
    if not samples:
        print("no corpus samples; nothing to compare")
        return 1
    if n:
        samples = samples[:n]
    print("comparing %d sample(s) against %s" % (len(samples), "the live server"))

    total_a = total_b = total_same = 0
    worst = []
    with _srv.Server(port=port, log=VOCAB_DIR / "out" / "server_verify.log") as s:
        print("server  : %s" % s.attach_why)
        for source, label, path in samples:
            rec = json.loads(path.read_text(encoding="utf-8"))
            for field in FIELDS[source]:
                text = rec.get(field) or ""
                if len(text) < 20:
                    continue
                ea, eb = off.encode(text), s.tokenize(text)
                same = sum(1 for x, y in zip(ea, eb) if x == y)
                denom = max(len(ea), len(eb))
                total_a += len(ea)
                total_b += len(eb)
                total_same += same
                rate = 100.0 * same / max(1, denom)
                if rate < 99.0:
                    worst.append((rate, label, field, len(ea), len(eb), ea, eb))
                print("  %-44s %-9s offline=%-6d live=%-6d token-match=%5.1f%%"
                      % (label[:44], field, len(ea), len(eb), rate))

    if not total_a:
        print("nothing comparable")
        return 1
    print("")
    print("TOTAL: offline produced %d ids, live produced %d, %d matched (%.3f%% of the longer)"
          % (total_a, total_b, total_same, 100.0 * total_same / max(total_a, total_b)))
    if worst:
        worst.sort()
        rate, label, field, na, nb, ea, eb = worst[0]
        print("worst sample: %s [%s] %.1f%% -- first divergence:" % (label, field, rate))
        for i, (x, y) in enumerate(zip(ea, eb)):
            if x != y:
                print("  at index %d: offline=%d (%r) live=%d (%r)"
                      % (i, x, off.piece(x), y, off.piece(y)))
                lo, hi = max(0, i - 3), min(len(ea), i + 4)
                print("  context offline: %r" % [off.piece(t) for t in ea[lo:hi]])
                print("  context live   : %r" % [off.piece(t) for t in eb[lo:hi]])
                break
    return 0
