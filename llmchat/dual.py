# -*- coding: utf-8 -*-
"""llmchat 双会话模式：左右双屏同时跑两个会话（配合 --parallel 2 的双槽服务）。

用法：
    python llmchat/dual.py [--base-url URL] [--temp X]

流程（每轮）：
    1. 输入 A 会话的问题（回车提交，quit 退出）
    2. 输入 B 会话的问题
    3. 两个请求同时发往服务端（服务端自动分配两个槽），左右分屏流式输出
    4. 状态行实时显示每路速度与合计；结束后显示精确 t/s 并落盘会话

会话文件：logs/llmchat_session_A.json / _B.json，与单人会话互不干扰。
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

if os.name == "nt":
    os.system("")          # enable ANSI escape processing on legacy consoles

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat as single                     # noqa: E402  复用 chat_sse / 默认值

HERE = Path(__file__).resolve().parent.parent
SESS_A = HERE / "llmchat" / "data" / "llmchat_session_A.json"
SESS_B = HERE / "llmchat" / "data" / "llmchat_session_B.json"

ANSI_ALT_ON, ANSI_ALT_OFF = "\x1b[?1049h", "\x1b[?1049l"
ANSI_HOME, ANSI_CLEAR_EOS = "\x1b[H", "\x1b[J"
ANSI_HIDE, ANSI_SHOW = "\x1b[?25l", "\x1b[?25h"


def sess_load(path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d.get("messages") or [], int(d.get("tokens") or 0)
    except Exception:
        return [], 0


def sess_save(path, messages, tokens):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"messages": messages, "tokens": int(tokens)},
                                   ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


class Session:
    """一路会话：自己的历史、自己的流式缓冲与计时。"""

    def __init__(self, tag, sess_path, base_url, temperature, slot_id=0):
        self.tag = tag                          # "A" / "B"
        self.sess_path = sess_path
        self.base_url = base_url
        self.temperature = temperature
        self.slot_id = slot_id                  # 钉死的槽号：A=0, B=1
        self.slot_file = f"dual_{tag}.slot"     # 各自的 KV 存档
        self.slot_ready = False                 # 本进程内是否已恢复过
        self.messages, self.tokens = sess_load(sess_path)
        self.lines = []                         # 已完成的显示行
        self.cur = ""                           # 正在输出的行
        self.t0 = None
        self.chars = 0
        self.done = False
        self.error = None
        self.final_tps = None
        self.final_n = None

    # --- 显示缓冲 -------------------------------------------------------
    def _feed(self, text):
        self.cur += text
        while "\n" in self.cur:
            line, self.cur = self.cur.split("\n", 1)
            self.lines.append(line)
            self.chars += len(line) + 1

    # --- 工作线程 -------------------------------------------------------
    def trim(self, cap=12000):
        """历史超过 cap（粗估 tok）时丢最旧的成对消息，只留 system + 最近一半。
        否则每轮都重发全量历史：prefill 越来越慢，接受率也越来越差。"""
        msgs = self.messages
        if len(msgs) < 5 or self.tokens < cap:
            return
        keep = msgs[:1] + msgs[len(msgs)//2:]
        self.messages = keep
        self.tokens = self.tokens // 2
        sess_save(self.sess_path, self.messages, self.tokens)

    def run(self, user_text):
        def worker():
            self.trim()
            self.messages = self.messages + [{"role": "user", "content": user_text}]
            self.t0 = time.time()
            try:
                reasoning, content, timings = single.chat_sse(
                    self.messages, self.base_url, self.temperature,
                    on_delta=lambda kind, text: self._feed(text) if kind == "content" else None,
                    id_slot=self.slot_id, cache_prompt=False)   # 测量态：关前缀缓存
            except Exception as e:
                self.error = str(e)
                self.done = True
                return
            self.chars += len(self.cur)
            if self.cur:
                self.lines.append(self.cur)
                self.cur = ""
            elapsed = time.time() - self.t0
            self.final_n = (timings or {}).get("predicted_n")
            self.final_tps = (timings or {}).get("predicted_per_second")
            if not self.final_tps and elapsed > 0 and content:
                self.final_tps = len(content) / elapsed / 1.6   # 兜底估算
            self.messages = self.messages + [{"role": "assistant", "content": content}]
            self.tokens = (self.tokens or 0) + len(content) // 2 + 10
            sess_save(self.sess_path, self.messages, self.tokens)
            # KV 槽位存档：下次（重启后）免 prefill，直接读回
            if (self.tokens or 0) >= 2048:
                single.slot_action(self.base_url, "save", self.slot_file, slot_id=self.slot_id)
            self.done = True

        threading.Thread(target=worker, daemon=True).start()

    # --- 速度 -----------------------------------------------------------
    def live_tps(self):
        if self.done:
            return self.final_tps or 0.0
        if not self.t0:
            return 0.0
        el = time.time() - self.t0
        return (self.chars / el / 1.6) if el > 0.5 else 0.0   # 中文≈1tok/1.6字符的粗估


def wrap(s, width):
    out = []
    for ln in s.split("\n"):
        while len(ln) > width:
            out.append(ln[:width])
            ln = ln[width:]
        out.append(ln)
    return out


def render(a, b, header):
    cols, rows = shutil.get_terminal_size((100, 30))
    half = max(20, cols // 2 - 1)
    pane_h = max(5, rows - 3)
    la = wrap("\n".join(a.lines + ([a.cur] if a.cur else [])), half)[-pane_h:]
    lb = wrap("\n".join(b.lines + ([b.cur] if b.cur else [])), half)[-pane_h:]
    la += [""] * (pane_h - len(la))
    lb += [""] * (pane_h - len(lb))

    ta, tb = a.live_tps(), b.live_tps()
    both_done = a.done and b.done
    total = (ta + tb) if not both_done else ((a.final_tps or 0) + (b.final_tps or 0))
    if both_done:
        status = (f"  [A] {a.final_tps or 0:.1f} t/s ({a.final_n or '?'} tok)   "
                  f"[B] {b.final_tps or 0:.1f} t/s ({b.final_n or '?'} tok)   "
                  f"合计 {total:.1f} t/s")
    else:
        status = (f"  [A] {ta:6.1f} t/s   [B] {tb:6.1f} t/s   "
                  f"合计 {total:6.1f} t/s   (实时估算)")

    out = [ANSI_HOME + header, status, "─" * half + "┼" + "─" * (cols - half - 1)]
    for x, y in zip(la, lb):
        out.append(x.ljust(half) + "│" + y)
    sys.stdout.write("\n".join(out) + ANSI_CLEAR_EOS)
    sys.stdout.flush()


def clear_session(s):
    """清空一路：历史归零 + 服务端对应槽位 KV 擦除 + 删存档。"""
    sysp = s.messages[0] if s.messages and s.messages[0].get("role") == "system" else None
    s.messages = [sysp] if sysp else []
    s.tokens = 0
    s.lines, s.cur, s.chars = [], "", 0
    r = single.slot_action(s.base_url, "erase", slot_id=s.slot_id)
    try:
        s.sess_path.unlink()
    except OSError:
        pass
    # 旧 KV 存档一并删掉：留着既占 158MB，下次启动还会读到一份对不上的历史。
    # 存档在服务端的 slot-save-path（src/data/slots/），不在会话目录。
    slots_dir = HERE / "data" / "slots"
    for f in (slots_dir / f"dual_{s.tag}.slot", slots_dir / f"dual_{s.tag}.slot.draft"):
        try:
            f.unlink()
        except OSError:
            pass
    print(f"[{s.tag}] 已清空（槽位 KV {'已擦除' if r else '擦除失败/无存档'}，会话历史已清）")


def run_round(a, b, header):
    qa = input("[A 会话] 问题（clear=双清 cleara/clearb=单清 quit=退出）> ").strip()
    low = qa.lower()
    if low in ("quit", "exit", "q"):
        return False
    if low == "clear":
        clear_session(a)
        clear_session(b)
        return True
    if low in ("cleara", "ca"):
        clear_session(a)
        return True
    if low in ("clearb", "cb"):
        clear_session(b)
        return True
    qb = input("[B 会话] 问题> ").strip()
    if qb.lower() in ("quit", "exit", "q"):
        return False

    a.run(qa)
    b.run(qb)
    ansi_ok = True
    try:
        sys.stdout.write(ANSI_ALT_ON + ANSI_HIDE)
        sys.stdout.flush()
    except OSError:
        ansi_ok = False
    if ansi_ok:
        try:
            while not (a.done and b.done):
                render(a, b, header)
                time.sleep(0.15)
            render(a, b, header)
            time.sleep(1.2)      # 让用户看一眼终态速度
        except Exception as e:
            print(f"[dual] 双屏渲染不可用（{e!r:.80}），降级为交错输出：")
            ansi_ok = False
        finally:
            if ansi_ok:
                sys.stdout.write(ANSI_SHOW + ANSI_ALT_OFF)
                sys.stdout.flush()
    if not ansi_ok:
        # 纯文本降级：谁有新内容就打谁的行，带 [A]/[B] 前缀
        seen = [0, 0]
        while not (a.done and b.done):
            time.sleep(0.2)
            for i, s in enumerate((a, b)):
                new_lines = s.lines[seen[i]:]
                for ln in new_lines:
                    print(f"[{s.tag}] {ln}")
                seen[i] = len(s.lines)
        for i, s in enumerate((a, b)):
            if s.cur:
                print(f"[{s.tag}] {s.cur}")
        print(f"  [A] {a.final_tps or 0:.1f} t/s   [B] {b.final_tps or 0:.1f} t/s   "
              f"合计 {(a.final_tps or 0) + (b.final_tps or 0):.1f} t/s")
    for s in (a, b):
        if s.error:
            print(f"[{s.tag}] 出错：{s.error[:200]}")
    return True


def main():
    ap = argparse.ArgumentParser(description="llmchat 双会话（左右双屏，配合 --parallel 2）")
    ap.add_argument("--base-url", default=single.DEFAULT_URL)
    ap.add_argument("--temp", type=float, default=0.6)
    args = ap.parse_args()

    info = single.probe(args.base_url) or {}
    model = (info.get("data") or [{}])[0].get("id", "?") if isinstance(info, dict) else "?"
    sysprompt = single.fill_sys(single.DEFAULT_SYS)

    a = Session("A", SESS_A, args.base_url, args.temp, slot_id=0)
    b = Session("B", SESS_B, args.base_url, args.temp, slot_id=1)
    for s in (a, b):
        if not s.messages:
            s.messages = [{"role": "system", "content": sysprompt}]
        # 槽位存档存在且历史非空才恢复：读回 KV，免掉整段 prefill（约 0.3-0.5s）
        if s.tokens and s.tokens >= 2048:
            r = single.slot_action(args.base_url, "restore", s.slot_file, slot_id=s.slot_id)
            if r and (r.get("n_saved") or r.get("n_restored")):
                s.slot_ready = True
                print(f"[{s.tag}] 槽位已读回（{s.tokens} tok），本轮免 prefill")
            else:
                print(f"[{s.tag}] 无可用槽位存档，首轮流式前会有一次 prefill")

    print(f"双会话模式  底座: {args.base_url}  temp={args.temp}  模型: {model}")
    print(f"A: {SESS_A.name}（{max(0, len(a.messages)-1)} 条）   "
          f"B: {SESS_B.name}（{max(0, len(b.messages)-1)} 条）")
    print("输入 quit 退出。两路提问输完后同时运行，左右屏各一路。")
    try:
        while run_round(a, b, f" 双槽双会话  {model} "):
            pass
    except (KeyboardInterrupt, EOFError):
        pass
    print("已退出；两路会话已各自落盘。")


if __name__ == "__main__":
    main()
