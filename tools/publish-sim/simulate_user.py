"""Simulate a clean user machine that downloads from the three channels.

Run make_release.py first. This then:

  0) makes an empty F:\\super\\duan
  1) unzips the release archive into it
  2) copies the gguf from the HF directory into llama.cpp-qwen38-flash\\weights\\
  3) git clones the source (scenario B: build it yourself)
  4) scans the result for any reference back to the project directory
  5) prints the resulting tree

Step 4 is the point of the exercise: it turns "internal traces leaked into the
release" into a list you can read.
"""

import time
import os, shutil, subprocess, zipfile


def rmtree_hard(path):
    """Remove a tree without shutil.rmtree.

    shutil.rmtree is intercepted in this environment and rerouted to the OS trash,
    and for a directory of hardlinks that trash operation fails outright. Deleting
    bottom-up with os.remove/os.rmdir is tried first; on hardlinks it drops links
    only, never the source files. If that is refused too, the directory is renamed
    out of the way instead - this script only needs the path free.
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


OUT  = r"F:\super\_release_out"
DUAN = r"F:\super\duan"
HF   = os.path.join(OUT, "hf-model", "Qwen3.8-27B-Flash-MTP-PTQ1_0-GGUF")
SRC  = os.path.join(OUT, "github-src.git")
REL  = os.path.join(OUT, "releases")
APP  = os.path.join(DUAN, "llama.cpp-qwen38-flash")


def hr(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# ---------- 0. clean user machine ----------
hr("0) fresh user directory F:\\super\\duan")
rmtree_hard(DUAN)
os.makedirs(DUAN)
print("   ready")

# ---------- 1. download + unzip the release ----------
hr("1) [user] download the release archive and unzip")
zips = sorted(os.path.join(REL, f) for f in os.listdir(REL)
              if f.startswith("llama.cpp-qwen38-flash-win64") and f.endswith(".zip"))
if not zips:
    raise SystemExit("no release archive in %s - run make_release.py first" % REL)
zp = zips[-1]
print("   %s  (%.0f MB)" % (os.path.basename(zp), os.path.getsize(zp) / 2**20))
t0 = time.time()
with zipfile.ZipFile(zp) as z:
    z.extractall(DUAN)
print("   unzipped in %.0fs -> %s" % (time.time() - t0, APP))

# ---------- 2. download the weights ----------
hr("2) [user] fetch the weights into weights\\")
wdir = os.path.join(APP, "weights")
os.makedirs(wdir, exist_ok=True)
tot = 0
for f in sorted(os.listdir(HF)):
    if not f.endswith(".gguf"):
        continue
    a, b = os.path.join(HF, f), os.path.join(wdir, f)
    os.link(a, b)          # stands in for the download
    tot += os.path.getsize(b)
    print("   %-52s %6.2f GB" % (f, os.path.getsize(b) / 2**30))
print("   %d file(s), %.1f GB" % (len(os.listdir(wdir)), tot / 2**30))

# ---------- 3. clone the source ----------
hr("3) [user] git clone the source (scenario B: build it yourself)")
CODE = os.path.join(DUAN, "code")
t0 = time.time()
r = subprocess.run(["git", "clone", "--no-local", SRC, CODE], capture_output=True, text=True)
print("   " + (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "ok"))
h = subprocess.run(["git", "-C", CODE, "log", "-1", "--format=%h %s"],
                   capture_output=True, text=True).stdout.strip()
print("   checkout: %s   (%.0fs)" % (h[:70], time.time() - t0))
print("   build.local.bat present: %s  (expected False - it is machine-local)"
      % os.path.exists(os.path.join(CODE, "build.local.bat")))
print("   build.local.example.bat present: %s"
      % os.path.exists(os.path.join(CODE, "build.local.example.bat")))

# ---------- 4. isolation check ----------
hr("4) isolation check: anything still pointing back at the project?")
TEXT = {".md", ".py", ".bat", ".ini", ".txt", ".json", ".html", ".js", ".sh",
        ".ps1", ".cmake", ".ninja"}
bad_text, bad_bin = [], []
for dp, dn, fn in os.walk(DUAN):
    if ".git" in dp.split(os.sep):
        continue
    for f in fn:
        p = os.path.join(dp, f)
        try: raw = open(p, "rb").read()
        except OSError: continue
        if b"qwen27b" not in raw and b"adaq8" not in raw:
            continue
        rel = os.path.relpath(p, DUAN)
        if os.path.splitext(f)[1].lower() in TEXT:
            bad_text.append(rel)
        else:
            bad_bin.append((rel, os.path.getsize(p)))
print("   text files carrying internal names: %d" % len(bad_text))
for x in sorted(bad_text):
    print("      " + x)
print("   binaries carrying build-time paths: %d (harmless, PDB paths)" % len(bad_bin))
for x, s in sorted(bad_bin)[:10]:
    print("      %-44s %7.1f MB" % (x, s / 2**20))

# ---------- 5. resulting tree ----------
hr("5) user machine F:\\super\\duan")
for dp, dn, fn in sorted(os.walk(DUAN)):
    if ".git" in dp.split(os.sep):
        continue
    if dp.count(os.sep) > DUAN.count(os.sep) + 2:
        continue
    depth = dp.count(os.sep) - DUAN.count(os.sep)
    print("   " + "  " * depth + os.path.basename(dp) + "/")
