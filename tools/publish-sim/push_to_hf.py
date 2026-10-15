"""Push the published weights to Hugging Face.

Run after `hf auth login` (needs a Write token) or with HF_TOKEN set.

  1) create the model repo (public)
  2) upload README.md        - the model card
  3) upload SHA256SUMS
  4) upload the two gguf files

Re-running is safe: uploads are idempotent and resume if interrupted.
"""

import os, sys, time

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("huggingface_hub not installed. Use the venv at "
             r"C:\Users\bestd\.workbuddy\binaries\python\envs\default")

USER   = "SuperLogic"
REPO   = "Swift-Bonsai-2-27B-GGUF"
FULL   = "%s/%s" % (USER, REPO)
SRC    = r"F:\super\_release_out\hf-model\Swift-Bonsai-2-27B-GGUF"

api = HfApi()

try:
    who = api.whoami()
except Exception as e:
    sys.exit("not logged in (%s). Run: hf auth login" % type(e).__name__)
print("logged in as: %s" % who.get("name"))

if not os.path.isdir(SRC):
    sys.exit("source directory missing: %s - run make_release.py first" % SRC)

# --- 1. repo ---
url = api.create_repo(FULL, repo_type="model", exist_ok=True, private=False)
print("repo: %s" % url)

# --- 2. small files first: the model card should be up before the weights ---
order = [f for f in sorted(os.listdir(SRC)) if not f.endswith(".gguf")]
order += [f for f in sorted(os.listdir(SRC)) if f.endswith(".gguf")]

total = sum(os.path.getsize(os.path.join(SRC, f)) for f in order)
done = 0
for f in order:
    p = os.path.join(SRC, f)
    if not os.path.isfile(p):
        continue
    size = os.path.getsize(p)
    print("\n--> %s  (%.2f GB)" % (f, size / 2**30))
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=p,
        path_in_repo=f,
        repo_id=FULL,
        repo_type="model",
    )
    dt = time.time() - t0
    done += size
    print("    done in %.0fs  (%.1f MB/s)   [%.1f / %.1f GB]"
          % (dt, size / 2**20 / max(dt, 0.1), done / 2**30, total / 2**30))

print("\nall uploaded: https://huggingface.co/%s" % FULL)
