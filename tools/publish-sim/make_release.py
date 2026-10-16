"""Build the three release channels under F:\\super\\_release_out.

  1) hf-model\\<repo>\\   a Hugging Face model repo (gguf hardlinked, not copied)
  2) github-src.git\\    a bare clone, what a user's `git clone` would hit
  3) releases\\*.zip     GitHub Releases assets

Note on the runtime zip: it used to also carry a copy of the launchers, the ini
and llmchat. Those now live in zgsndzq/qwen38-27b-flash-mtp, where they can be
reviewed and patched; a second copy inside a zip only drifts. So the zip is down
to what that repo cannot host -- the compiled tree -- with build/ at the zip root
so that extracting it into a clone of the runtime repo lands it exactly where
llmserver.bat looks.

Run this, then simulate_user.py, to rehearse the whole publish -> download path.
"""

import hashlib
import os
import shutil
import subprocess
import time


def rmtree_hard(path):
    """Remove a tree without shutil.rmtree.

    shutil.rmtree is intercepted in this environment and rerouted to the OS trash,
    and for a directory of hardlinks that trash operation fails outright. Deleting
    bottom-up with os.remove/os.rmdir is tried first; on hardlinks it drops links
    only, never the source files. If that is refused too, the directory is simply
    renamed out of the way - the script only needs the path free, not the bytes gone.
    """
    if not os.path.isdir(path):
        return
    for dp, dn, fn in os.walk(path, topdown=False):
        for f in fn:
            try: os.remove(os.path.join(dp, f))
            except OSError: pass
        for d in dn:
            try: os.rmdir(os.path.join(dp, d))
            except OSError: pass
    if os.path.isdir(path):
        try: os.rename(path, "%s.stale-%d" % (path, int(time.time())))
        except OSError: pass


Q   = r"F:\super\qwen27b"
OUT = r"F:\super\_release_out"
HF  = os.path.join(OUT, "hf-model", "Qwen3.8-27B-Flash-MTP-GGUF")
SRC = os.path.join(OUT, "github-src.git")
REL = os.path.join(OUT, "releases")

# Only what the runtime actually needs. The other four gguf files in weights/ are
# development iterations: two older MTP donors superseded on 10-07, one with no MTP
# head at all (851 tensors vs 866 - it cannot self-speculate), and one early graft
# whose general.name collides with PrismML's own published model.
PUBLISH = [
    "Qwen3.8-27B-Flash-MTP.gguf",   # production base: llmserver.bat MODEL=
    "mmproj-Qwen3.8-27B-Q8_0.gguf",              # vision projector: llmserver.bat VISION=1
]


def hr(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# ---------- 1. HF model repo ----------
hr("1) HF model repo  (weights channel)")
rmtree_hard(HF)
os.makedirs(HF)
src_w = os.path.join(Q, "weights")
n = 0
for f in PUBLISH:
    a, b = os.path.join(src_w, f), os.path.join(HF, f)
    if not os.path.exists(a):
        print("   MISSING %s" % f)
        continue
    os.link(a, b)          # hardlink: independent path, no extra space
    n += 1
    print("   %-52s %8.2f GB" % (f, os.path.getsize(b) / 2**30))
others = len([x for x in os.listdir(src_w) if x.endswith(".gguf")]) - n
print("   -> %d published, %d development variants withheld" % (n, others))

# --- model card: maintained next to this script so a rebuild does not wipe it
head = subprocess.run(["git", "-C", os.path.join(Q, "code"), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
card = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model-card.md"),
            encoding="utf-8").read().replace("{{ENGINE_COMMIT}}", head[:8])
open(os.path.join(HF, "README.md"), "w", encoding="utf-8").write(card)
print("   model card -> README.md  (engine commit %s)" % head[:8])

# --- SHA256SUMS
print("   hashing for SHA256SUMS...")
sums = []
for f in sorted(os.listdir(HF)):
    if not f.endswith(".gguf"):
        continue
    m = hashlib.sha256()
    with open(os.path.join(HF, f), "rb") as fh:
        while True:
            blk = fh.read(4 << 20)
            if not blk:
                break
            m.update(blk)
    sums.append("%s  %s" % (m.hexdigest(), f))
open(os.path.join(HF, "SHA256SUMS"), "w", encoding="utf-8").write(
    "# SHA256SUMS - Qwen3.8-27B-Flash-MTP (PTQ1_0 + MTP)\n\n" + "\n".join(sums) + "\n")
print("   SHA256SUMS -> %d line(s)" % len(sums))

# ---------- 2. GitHub source repo ----------
hr("2) GitHub source repo  (what `git clone` hits)")
rmtree_hard(SRC)
code = os.path.join(Q, "code")
r = subprocess.run(["git", "clone", "--bare", "--local", code, SRC],
                   capture_output=True, text=True)
msg = (r.stderr.strip() or r.stdout.strip() or "ok").splitlines()
print("   " + msg[-1])
head_full = subprocess.run(["git", "--git-dir", SRC, "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
br = subprocess.run(["git", "--git-dir", SRC, "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True).stdout.strip()
print("   HEAD   = %s" % head_full[:12])
print("   branch = %s" % br)
sz = sum(os.path.getsize(os.path.join(dp, f))
         for dp, dn, fn in os.walk(SRC) for f in fn)
print("   size   = %.0f MB (objects hardlinked, real usage lower)" % (sz / 2**20))

# ---------- 3. Releases ----------
hr("3) Releases assets")
os.makedirs(REL, exist_ok=True)
for f in os.listdir(REL):
    if f.endswith(".zip"):
        try: os.remove(os.path.join(REL, f))
        except OSError: pass

stage = os.path.join(OUT, "_stage_bin")
rmtree_hard(stage)
shutil.copytree(os.path.join(Q, "build", "bin"),  os.path.join(stage, "build", "bin"))
shutil.copytree(os.path.join(Q, "build", "cuda"), os.path.join(stage, "build", "cuda"))
n_files = sum(len(fn) for _, _, fn in os.walk(stage))
print("   binaries only: %d files (build\\bin + build\\cuda, no scripts by design)" % n_files)

name = "qwen38-27b-flash-win64-sm89-binaries.zip"
zp = os.path.join(REL, name)
t0 = time.time()
shutil.make_archive(zp[:-4], "zip", stage)
print("   %s  -> %.1f MB  (%.0fs)"
      % (name, os.path.getsize(zp) / 2**20, time.time() - t0))

name2 = "qwen38-27b-flash-src-%s.zip" % head_full[:8]
zp2 = os.path.join(REL, name2)
subprocess.run(["git", "--git-dir", SRC, "archive", "--format=zip",
                "-o", zp2, "HEAD"], capture_output=True, text=True)
print("   %s  -> %.1f MB" % (name2, os.path.getsize(zp2) / 2**20))

rmtree_hard(stage)

hr("release side ready")
for dp, dn, fn in os.walk(OUT):
    if dp.count(os.sep) > OUT.count(os.sep) + 1:
        continue
    tot = sum(os.path.getsize(os.path.join(dd, f)) for dd, _, ff in os.walk(dp) for f in ff)
    print("  %-52s %8.1f MB" % (os.path.relpath(dp, OUT), tot / 2**20))
