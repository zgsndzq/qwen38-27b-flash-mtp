#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mem/ - standing memory: files that ride on the very first turn and then stay.

Unlike an /img attachment (one message, then gone), memory is meant to sit at the
head of the context for the whole session: a reference picture, a conventions file.
It is injected once, on the first turn, and from then on it is just history - so it
costs a prefill once and rides on the prompt cache afterwards.
"""

import re
import shutil
from pathlib import Path

MEM_DIR = Path(__file__).resolve().parent / "mem"

TEXT_SUFFIX = (".md", ".txt")
IMAGE_SUFFIX = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")

MEM_TEXT, MEM_IMAGE = 1, 2        # the bit flags LLAMA_CHAT_MEM used to take: 1 text,
#                                  2 image, 3 both, 0 off.
ALL = "all"                       # the shelf = mem/ itself, no subfolder

# --image-min-tokens is a floor, not a measurement: a big picture costs more. Kept
# here only so /mem can warn about the order of magnitude, not the exact number.
IMG_TOKEN_FLOOR = 1024

SPECS = {"0": "off", "1": "text only", "2": "images only", "3": "text + images"}

# A gallery picture answers to its own name plus a same-named sidecar (.md/.txt). That
# text is what gets matched, so how it is worded sets the hit rate - write the objects
# and the action, in the language you will actually type. Below this score, no picture.
MEM_MIN_SCORE = 0.16


def _grams(text):
    """Latin words, CJK single chars, CJK bigrams - no tokenizer, no dependency."""
    low = (text or "").lower()
    latin = set(re.findall(r"[a-z0-9]{2,}", low))
    cjk = [c for c in low if "\u4e00" <= c <= "\u9fff"]
    bigram = {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
    return latin, set(cjk), bigram


def score(query, hay):
    """How much of the question is covered by this picture's tags, 0.0 ~ 1.0.

    Bigrams weigh most (they catch 鹈鹕 / 骑车 as units), whole latin words next, single
    characters least (they fire on coincidences like 子 in almost any Chinese text).
    """
    q_lat, q_cjk, q_big = _grams(query)
    h_lat, h_cjk, h_big = _grams(hay)
    got = 3 * len(q_big & h_big) + 2 * len(q_lat & h_lat) + len(q_cjk & h_cjk)
    total = 3 * len(q_big) + 2 * len(q_lat) + len(q_cjk)
    return got / total if total else 0.0


def sidecar(image):
    """The tag text that belongs to one picture, or None."""
    for suffix in TEXT_SUFFIX:
        p = image.with_suffix(suffix)
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8-sig").strip()
            except (OSError, UnicodeDecodeError):
                return None
    return None


def gallery(group=ALL):
    """[(picture, tag text)] in name order - the pictures a query can hit."""
    root = shelf(group)
    if not root.is_dir():
        return []
    return [(p, sidecar(p)) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIX]


def rank(query, group=ALL):
    """Every gallery picture scored, best first. /mem find shows this."""
    rows = [(score(query, "%s %s" % (p.stem, tag or "")), p, tag)
            for p, tag in gallery(group)]
    rows.sort(key=lambda r: (-r[0], len(r[1].name)))
    return rows


def pick(query, group=ALL, min_score=MEM_MIN_SCORE):
    """(picture, tag, score) of the single best hit, or None when nothing fits.

    One picture per message, by design: each picture costs context on top of the ones
    already in the history, and a scene request usually has one subject.
    """
    rows = rank(query, group)
    if not rows or rows[0][0] < min_score:
        return None
    got, path, tag = rows[0]
    return path, tag, got


def shelf(group=ALL):
    """mem/ when the group is 'all', otherwise one named folder under it."""
    return MEM_DIR if not group or group == ALL else MEM_DIR / group


def groups():
    """Names of the shelves under mem/, so several memories can coexist."""
    if not MEM_DIR.is_dir():
        return []
    return sorted(p.name for p in MEM_DIR.iterdir() if p.is_dir())


def scan(group=ALL, mode=0):
    """(texts, images) sorted by path - the order must never shift, or the prompt
    cache stops matching and every turn re-prefills the whole memory.

    A text that names a picture is that picture's tag, not standing memory: it rides
    along when the picture is picked, so it never gets injected twice.
    """
    root = shelf(group)
    if not mode or not root.is_dir():
        return [], []
    tags = {p.with_suffix(s) for p, _ in gallery(group) for s in TEXT_SUFFIX}
    texts, images = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.lower() == "readme.md":
            continue                       # the shelf's own manual is not a memory
        suffix = p.suffix.lower()
        if suffix in TEXT_SUFFIX and mode & MEM_TEXT and p not in tags:
            texts.append(p)
        elif suffix in IMAGE_SUFFIX and mode & MEM_IMAGE:
            images.append(p)
    return texts, images


def parts(group=ALL, mode=0):
    """Message parts for the memory: text blocks first, then images, no question."""
    out = [{"type": "text", "text": "[memory %s]\n%s" % (name, body)}
           for name, body in _readable(group, mode) if body]
    out += [{"type": "image_url", "image_url": {"path": str(path)}}
            for path in scan(group, mode)[1]]
    return out


def _readable(group, mode):
    """Text memories as (name, body). A file that will not read stays visible as
    (name, None) rather than vanishing - silent drops are how bad sessions hide."""
    rows = []
    for path in scan(group, mode)[0]:
        try:
            body = path.read_text(encoding="utf-8-sig").strip()   # strips Notepad's BOM
        except (OSError, UnicodeDecodeError):
            rows.append((path.name, None))
            continue
        if body:
            rows.append((path.name, body))
    return rows


def describe(group=ALL, mode=0):
    """The /mem readout: which shelf, which mode, what is in it, what it costs."""
    lines = ["[mem] shelf=%s  mode=%s (%s)  dir=%s"
             % (group, mode, SPECS.get(str(mode), "?"), shelf(group))]
    names = groups()
    if names:
        lines.append("      shelves: all, " + ", ".join(names))
    if not mode:
        lines.append("      injection off - /mem 3 turns it on")
        return "\n".join(lines)
    texts = _readable(group, mode)
    images = scan(group, mode)[1]
    chars = 0
    for name, body in texts:
        if body is None:
            lines.append("      txt %-24s UNREADABLE - left out" % name)
        else:
            chars += len(body)
            lines.append("      txt %-24s %6d chars" % (name, len(body)))
    for path in images:
        lines.append("      img %-24s %6d KB" % (path.name, path.stat().st_size // 1024))
    if not texts and not images:
        lines.append("      shelf looks empty (or holds only unsupported types)")
    else:
        lines.append("      first turn pays >= %d image tokens (%d x %d floor) plus "
                     "%d chars of text; later turns reuse it from cache"
                     % (len(images) * IMG_TOKEN_FLOOR, len(images), IMG_TOKEN_FLOOR, chars))
    return "\n".join(lines)


def add(group, src):
    """Copy a file onto the shelf; the original stays where it is."""
    dest_dir = shelf(group)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(src).name
    shutil.copy2(src, dest)
    return dest
