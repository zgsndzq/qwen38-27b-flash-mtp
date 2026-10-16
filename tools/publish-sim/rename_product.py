"""Rename the product to Qwen3.8-27B-Flash-MTP.

Rule order matters, longest / most specific first, and two names must survive
untouched because they are attribution, not ours:

    prism-ml/Ternary-Bonsai-2-27B-gguf      upstream we derived from
    ukisai/Swift-Bonsai-2-GGUF              the fine-tune we merged
    graft.donor.name inside the gguf        provenance recorded in the file

Bytes in / bytes out, so CRLF and LF files keep their line endings and the
hardlinks into qwen27b stay intact.
"""

import os
import re

ROOT = r"F:\super\ada-ternary-runtime"

MODEL   = "Qwen3.8-27B-Flash-MTP"
HF_REPO = "Qwen3.8-27B-Flash-MTP-GGUF"
RUNTIME = "qwen38-27b-flash-mtp"
ENGINE  = "llama.cpp-qwen38-flash"

# (pattern, replacement, is_regex)
RULES = [
    # --- weights / repo ids -------------------------------------------------
    (r"Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf", MODEL + ".gguf", False),
    (r"SuperLogic/Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km", "SuperLogic/" + HF_REPO, False),
    (r"Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km", HF_REPO, False),

    # --- our own model name, never upstream's ------------------------------
    # the lookahead keeps prism-ml/Ternary-Bonsai-2-27B-gguf intact
    (r"Ternary-Bonsai-2-27B(?!-gguf)", MODEL, True),
    (r"Swift-Bonsai-2-27B(?!-GGUF)", MODEL, True),
    (r"Swift-Bonsai-2-SelfMTP-Q6K", MODEL, False),
    (r"SelfMTP-Q6K", MODEL, False),

    # --- repositories ------------------------------------------------------
    (r"ada-ternary-runtime", RUNTIME, False),
    (r"ada-ternary-win64-sm89", "qwen38-27b-flash-win64-sm89", False),
    (r"ada-ternary", ENGINE, False),
]

# line 1 is the document title: make it read as a product, not a repo name
TITLES = {
    "README.md":        "# %s — %s 运行包\n" % (RUNTIME, MODEL),
    "README.en.md":     "# %s — %s runtime\n" % (RUNTIME, MODEL),
}

TARGETS = [
    "README.md", "README.en.md", "build/README.md", "weights/README.md",
    "features.ini", "llmserver.bat", "llmserver.sh",
    "llmchat/README.md", "llmchat/llmchat.bat",
]

for rel in TARGETS:
    p = os.path.join(ROOT, rel)
    raw = open(p, "rb").read()
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        s = raw.decode("ascii")
    orig = s

    for pat, rep, is_re in RULES:
        s = re.sub(pat, rep, s) if is_re else s.replace(pat, rep)

    if rel in TITLES:
        crlf = "\r\n" if raw[:200].count(b"\r\n") else "\n"
        body = s.split("\n", 1)[1]
        s = TITLES[rel].rstrip("\n") + crlf + body.lstrip("\r\n")

    if s != orig:
        open(p, "wb").write(s.encode("utf-8"))
        print("  %-24s changed" % rel)
    else:
        print("  %-24s unchanged" % rel)

print("\ndone")