#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成图片记忆测试图（HTML，随后用 visual-shot.ps1 截成 PNG）。

每张图 = 一个唯一名字 + 一个唯一 4 位编号。问题问"写着「X」的那张图，编号是多少"，
⇒ 模型必须在 N 张图里定位到名字匹配的那张、再 OCR 出编号，这正是多图检索要测的。
"""

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "imghtml"
N = int(__import__("os").environ.get("IMG_N", "8"))

BEASTS = ["青鸾", "白泽", "玄武", "朱雀", "麒麟", "饕餮", "鲲鹏", "九尾"]

rnd = random.Random(31415)
used, codes = set(), []
while len(codes) < N:
    c = rnd.randint(1000, 9999)
    if c not in used:
        used.add(c)
        codes.append(c)

TPL = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;padding:0;width:600px;height:400px;overflow:hidden}}
.card{{width:600px;height:400px;box-sizing:border-box;background:#f7f2e7;
  border:10px solid #8b5a2b;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  font-family:"Microsoft YaHei","SimHei",sans-serif}}
.nm{{font-size:110px;color:#2b1810;letter-spacing:18px;line-height:1.1}}
.cd{{font-size:58px;color:#8b0000;letter-spacing:6px;margin-top:26px;font-weight:bold}}
</style></head><body>
<div class="card"><div class="nm">{nm}</div><div class="cd">编号 {code}</div></div>
</body></html>
"""

OUT.mkdir(parents=True, exist_ok=True)
for i, (nm, code) in enumerate(zip(BEASTS, codes), 1):
    (OUT / ("c%02d.html" % i)).write_text(TPL.format(nm=nm, code=code), encoding="utf-8")

print("生成 %d 张 -> %s" % (len(codes), OUT))
for i, (nm, code) in enumerate(zip(BEASTS, codes), 1):
    print("  c%02d  %s  %d" % (i, nm, code))
