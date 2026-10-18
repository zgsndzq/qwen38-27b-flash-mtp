"""Pack the release zip: prebuilt binaries + runtime + this package.

    python tools/make_release.py                 # build bin + cuda runtime + package
    python tools/make_release.py --with-source    # also ship code/ (engine source)
    python tools/make_release.py --dry-run        # list what would go in, write nothing

What never goes in: weights/*.gguf (Hugging Face), build/src + build/logs (CMake tree
and logs), logs/, __pycache__, .git. The zip lands next to the package in ../release/.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # src/
OUT_DIR = ROOT.parent / "release"

# 顶层文件与目录：整棵打进去
TOP_FILES = ["LICENSE", "NOTICE", "README.md", "README.en.md",
             "MAINTAINERS.md", "MAINTAINERS.en.md", "features.ini", "llmserver.bat"]
TOP_DIRS = ["maps", "llmchat", "tools", "build"]
# build/ 里只要这两块：bin 是 exe+dll，cuda 是随包带的 CUDA 运行时
BUILD_KEEP = ["build/bin", "build/cuda"]
SKIP_DIRS = {".git", "__pycache__", "logs", ".workbuddy", "weights"}
SKIP_SUFFIX = (".pyc", ".log", ".gguf")
# weights 只带说明，不带权重本身
EXTRA_FILES = ["weights/README.md"]


def version_tag() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if sha:
            dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=20).stdout.strip()
            return sha + ("-dirty" if dirty else "")
    except Exception:
        pass
    return time.strftime("%Y%m%d")


def iter_files(with_source: bool):
    for rel in TOP_FILES + EXTRA_FILES:
        p = ROOT / rel
        if p.is_file():
            yield p, rel
        else:
            print(f"[warn] 缺少 {rel}", file=sys.stderr)

    dirs = [ROOT / d for d in ["maps", "llmchat", "tools"]] + [ROOT / b for b in BUILD_KEEP]
    if with_source:
        dirs.append(ROOT / "code")

    for base in dirs:
        if not base.exists():
            print(f"[warn] 缺少目录 {base.relative_to(ROOT)}", file=sys.stderr)
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                if name.endswith(SKIP_SUFFIX):
                    continue
                p = Path(dirpath) / name
                yield p, str(p.relative_to(ROOT)).replace("\\", "/")


def sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-source", action="store_true", help="also ship code/ (engine source)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = list(iter_files(a.with_source))
    total = sum(p.stat().st_size for p, _ in files)
    if a.dry_run:
        print(f"{len(files)} 个文件，{total / 1e9:.2f} GB 未压缩")
        for p, rel in files[:15]:
            print(f"  {len(files) and p.stat().st_size / 1e6:9.1f} MB  {rel}")
        print(f"  ... 共 {len(files)} 个")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = version_tag()
    out = Path(a.out) if a.out else OUT_DIR / f"qwen38-27b-flash-mtp-win64-{tag}.zip"

    t0 = time.time()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i, (p, rel) in enumerate(files, 1):
            z.write(p, rel)
            if i % 200 == 0:
                print(f"  {i}/{len(files)} …", flush=True)

    size = out.stat().st_size
    print(f"\n{out}")
    print(f"  {len(files)} 个文件  原始 {total / 1e9:.2f} GB  压缩后 {size / 1e6:.0f} MB  "
          f"用 {time.time() - t0:.0f} s")
    print(f"  sha256 {sha256(out)}")
    print("\n发布时把上面三行写进 Release 说明；权重不进 zip，走 Hugging Face。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
