"""Second pass: the publishing tooling and the model card.

Same rules as rename_product.py, plus a few that only exist here:
  - the bare "Swift-Bonsai-2" used as a product title
  - the source-snapshot asset name, which should track the runtime repo rather
    than the engine repo name
  - simulate_user.py carries a staged-directory name that had already drifted
"""

import os
import re

D = r"F:\super\qwen27b\tools\publish-sim"

MODEL   = "Qwen3.8-27B-Flash-MTP"
HF_REPO = "Qwen3.8-27B-Flash-MTP-GGUF"
RUNTIME = "qwen38-27b-flash-mtp"
ENGINE  = "llama.cpp-qwen38-flash"

RULES = [
    (r"Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km\.gguf", MODEL + ".gguf", True),
    (r"SuperLogic/Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km", "SuperLogic/" + HF_REPO, False),
    (r"Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km", HF_REPO, False),
    (r"SHA256SUMS - Swift-Bonsai-2 \(PTQ1_0", "SHA256SUMS - " + MODEL + " (PTQ1_0", False),

    # product name, never the two upstream repos it is derived from
    (r"Ternary-Bonsai-2-27B(?!-gguf)", MODEL, True),
    (r"Swift-Bonsai-2-27B(?!-GGUF)", MODEL, True),
    (r"Swift-Bonsai-2(?!-GGUF)", MODEL, True),

    # repositories and asset names
    (r"ada-ternary-runtime", RUNTIME, False),
    (r"ada-ternary-win64-sm89", "qwen38-27b-flash-win64-sm89", False),
    (r"ada-ternary-src", "qwen38-27b-flash-src", False),
    (r"ada-ternary", ENGINE, False),

    # simulate_user.py had a stale staged-directory name
    (r'Ternary-Bonsai-2-27B-PTQ1_0-GGUF', HF_REPO, False),
]

for fn in ["make_release.py", "push_to_hf.py", "simulate_user.py",
           "README.md", "model-card.md"]:
    p = os.path.join(D, fn)
    s = open(p, "rb").read().decode("utf-8")
    orig = s
    for pat, rep, is_re in RULES:
        s = re.sub(pat, rep, s) if is_re else s.replace(pat, rep)
    if s != orig:
        open(p, "wb").write(s.encode("utf-8"))
        print("  %-20s changed" % fn)
    else:
        print("  %-20s unchanged" % fn)