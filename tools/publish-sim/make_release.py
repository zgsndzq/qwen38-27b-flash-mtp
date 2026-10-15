import os, shutil, subprocess, hashlib, json, time

Q  = r"F:\super\qwen27b"
OUT = r"F:\super\_release_out"
HF  = os.path.join(OUT, "hf-model", "Ternary-Bonsai-2-27B-PTQ1_0-GGUF")
SRC = os.path.join(OUT, "github-src.git")
REL = os.path.join(OUT, "releases")

def hr(t): print("\n" + "="*72 + "\n" + t + "\n" + "="*72)

# ---------- 1. HF model repo ----------
hr("1) HF model repo  (模拟 huggingface.co/…. 权重渠道)")
if os.path.isdir(HF): shutil.rmtree(HF)
os.makedirs(HF)
src_w = os.path.join(Q, "weights")
n = 0
for f in sorted(os.listdir(src_w)):
    if not f.lower().endswith(".gguf"): continue
    a, b = os.path.join(src_w, f), os.path.join(HF, f)
    os.link(a, b)          # hardlink = 路径独立但不占空间
    n += 1
    print("   %-52s %8.2f GB" % (f, os.path.getsize(b)/2**30))
print("   -> %d 个 gguf 就位" % n)

# ---------- 2. GitHub source repo ----------
hr("2) GitHub 源码仓  (模拟 git clone 的目标)")
if os.path.isdir(SRC): shutil.rmtree(SRC)
code = os.path.join(Q, "code")
r = subprocess.run(["git", "clone", "--bare", "--local", code, SRC],
                   capture_output=True, text=True)
print("   " + (r.stderr.strip() or r.stdout.strip() or "ok"))
head = subprocess.run(["git", "--git-dir", SRC, "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
br   = subprocess.run(["git", "--git-dir", SRC, "rev-parse", "--abbrev-ref", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print("   HEAD   = %s" % head[:12])
print("   branch = %s" % br)
sz = sum(os.path.getsize(os.path.join(dp, f))
         for dp, dn, fn in os.walk(SRC) for f in fn)
print("   size   = %.0f MB (对象走硬链接，实占更小)" % (sz/2**20))

# ---------- 3. Releases ----------
hr("3) Releases 资产  (模拟 GitHub Releases 的 zip 下载)")
os.makedirs(REL, exist_ok=True)
for f in os.listdir(REL):
    os.remove(os.path.join(REL, f))

# 3a. 预编译运行包 = bin + cuda + runtime 配置
stage = os.path.join(OUT, "_stage_runtime")
if os.path.isdir(stage): shutil.rmtree(stage)
os.makedirs(os.path.join(stage, "ada-ternary"), exist_ok=True)
dst = os.path.join(stage, "ada-ternary")
shutil.copytree(os.path.join(Q, "build", "bin"),  os.path.join(dst, "build", "bin"))
shutil.copytree(os.path.join(Q, "build", "cuda"), os.path.join(dst, "build", "cuda"))
for f in ["llmserver.bat", "features.ini", "README.md", "README.en.md"]:
    shutil.copy2(os.path.join(Q, f), os.path.join(dst, f))
shutil.copytree(os.path.join(Q, "maps"), os.path.join(dst, "maps"))
shutil.copytree(os.path.join(Q, "llmchat"), os.path.join(dst, "llmchat"),
                ignore=shutil.ignore_patterns("__pycache__", "*.bak_*", "*.bak-*"))
for junk in ("logs", "out"):
    p = os.path.join(dst, "llmchat", junk)
    if os.path.isdir(p): shutil.rmtree(p)

name = "ada-ternary-win64-sm89-b3e7c012.zip"
zp = os.path.join(REL, name)
t0 = time.time()
shutil.make_archive(zp[:-4], "zip", stage, "ada-ternary")
print("   %s" % name)
print("     %.0f MB   (%.0fs)" % (os.path.getsize(zp)/2**20, time.time()-t0))

# 3b. 源码 release 快照（给不装 git 的人）
name2 = "ada-ternary-src-b3e7c012.zip"
zp2 = os.path.join(REL, name2)
stage2 = os.path.join(OUT, "_stage_src")
if os.path.isdir(stage2): shutil.rmtree(stage2)
os.makedirs(stage2)
r = subprocess.run(["git", "--git-dir", SRC, "archive", "--format=zip",
                    "-o", zp2, "HEAD"], capture_output=True, text=True)
print("   %s  -> %.0f MB  %s" % (name2, os.path.getsize(zp2)/2**20 if os.path.exists(zp2) else 0,
                                 r.stderr.strip()[:60]))
shutil.rmtree(stage, ignore_errors=True)
shutil.rmtree(stage2, ignore_errors=True)

hr("发布端就绪")
for dp, dn, fn in os.walk(OUT):
    if dp.count(os.sep) > OUT.count(os.sep) + 1: continue
    tot = sum(os.path.getsize(os.path.join(dd, f)) for dd, _, ff in os.walk(dp) for f in ff)
    print("  %-52s %8.1f MB" % (os.path.relpath(dp, OUT), tot/2**20))
