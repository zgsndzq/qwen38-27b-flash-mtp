import os, shutil, subprocess, zipfile, time

OUT  = r"F:\super\_release_out"
DUAN = r"F:\super\duan"
HF   = os.path.join(OUT, "hf-model", "Ternary-Bonsai-2-27B-PTQ1_0-GGUF")
SRC  = os.path.join(OUT, "github-src.git")
REL  = os.path.join(OUT, "releases")
APP  = os.path.join(DUAN, "ada-ternary")

def hr(t): print("\n" + "="*72 + "\n" + t + "\n" + "="*72)

# ---------- 0. 干净的用户机 ----------
hr("0) 清空并新建用户目录 F:\\super\\duan")
if os.path.isdir(DUAN): shutil.rmtree(DUAN)
os.makedirs(DUAN)
print("   已建")

# ---------- 1. 从 Releases 下载解包 ----------
hr("1) [用户动作] 下载 release zip 并解包")
zp = sorted(os.path.join(REL, f) for f in os.listdir(REL)
            if f.startswith("ada-ternary-win64") and f.endswith(".zip"))[-1]
print("   下载 %s  (%.0f MB)" % (os.path.basename(zp), os.path.getsize(zp)/2**20))
t0 = time.time()
with zipfile.ZipFile(zp) as z:
    z.extractall(DUAN)
print("   解包完成 (%.0fs) -> %s" % (time.time()-t0, APP))

# ---------- 2. 从 HF 下载权重 ----------
hr("2) [用户动作] 从 Hugging Face 下载权重到 weights\\")
wdir = os.path.join(APP, "weights")
os.makedirs(wdir, exist_ok=True)
tot = 0
for f in sorted(os.listdir(HF)):
    if not f.endswith(".gguf"): continue
    a, b = os.path.join(HF, f), os.path.join(wdir, f)
    os.link(a, b)
    tot += os.path.getsize(b)
print("   %d 个 gguf, 合计 %.1f GB" % (len([x for x in os.listdir(wdir)]), tot/2**30))

# ---------- 3. 从 GitHub clone 源码 ----------
hr("3) [用户动作] git clone 源码（场景 B：自己编译）")
CODE = os.path.join(DUAN, "code")
t0 = time.time()
r = subprocess.run(["git", "clone", "--no-local", SRC, CODE], capture_output=True, text=True)
print("   " + (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "ok"))
h = subprocess.run(["git", "-C", CODE, "log", "-1", "--format=%h %s"],
                   capture_output=True, text=True).stdout.strip()
print("   checkout: %s   (%.0fs)" % (h[:70], time.time()-t0))
print("   build.local.bat 在不在: %s  (预期 False，它是本机文件)"
      % os.path.exists(os.path.join(CODE, "build.local.bat")))
print("   build.local.example.bat 在不在: %s" % os.path.exists(os.path.join(CODE, "build.local.example.bat")))

# ---------- 4. 隔离性自检 ----------
hr("4) 隔离性自检：duan 里有没有指向 qwen27b 的引用")
bad = []
for dp, dn, fn in os.walk(DUAN):
    if ".git" in dp.split(os.sep): continue
    for f in fn:
        p = os.path.join(dp, f)
        if os.path.getsize(p) > 4*2**20: continue
        try: s = open(p, "rb").read()
        except OSError: continue
        if b"qwen27b" in s or b"adaq8" in s:
            bad.append(os.path.relpath(p, DUAN))
print("   含 qwen27b/adaq8 字样的文件: %d 个" % len(bad))
for b in bad[:15]: print("      " + b)

# ---------- 5. 用户机结构 ----------
hr("5) 用户机 F:\\super\\duan 结构")
for dp, dn, fn in sorted(os.walk(DUAN)):
    if dp.count(os.sep) > DUAN.count(os.sep) + 2: continue
    if ".git" in dp.split(os.sep): continue
    depth = dp.count(os.sep) - DUAN.count(os.sep)
    print("   " + "  "*depth + os.path.basename(dp) + "/")
