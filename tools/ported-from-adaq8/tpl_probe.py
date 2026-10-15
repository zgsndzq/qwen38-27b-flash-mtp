# -*- coding: utf-8 -*-
"""只渲染、不推理：看 **到底什么进了 prompt**。

用 POST /apply-template —— 它只把 messages 过一遍 chat template 把字符串吐回来，
不碰 slot 的 KV、不做任何推理，对运行中的服务是纯只读的。

两组对照：
  A  system 消息 与 reasoning_content 的处理（chat.py 回传思考对不对）
  B  reasoning_effort 往 system 里塞了什么（medium vs xhigh vs 不给）
"""
import io
import json
import os
import urllib.request

BASE = os.environ.get("TPL_BASE", "http://127.0.0.1:18080")
SYS_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llmchat", "sys.txt")

U1 = "What is 17 times 23?"
R1 = "Let me compute: 17*20=340, 17*3=51, 340+51=391."
C1 = "The answer is 391."


def apply_template(messages, kwargs=None):
    body = {"messages": messages}
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BASE + "/apply-template", data=data,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8")


def show(label, messages, kwargs=None, head=260):
    print("=== %s ===" % label)
    try:
        out = apply_template(messages, kwargs)
    except Exception as e:
        print("   FAILED: %r" % (e,))
        print()
        return
    print(out[:head].replace("\n", "\\n"))
    print("   [len=%d  system blocks=%d  'Reasoning effort'=%d]"
          % (len(out), out.count("<|im_start|>system"), out.count("Reasoning effort")))
    print()


# ---------------- A: reasoning_content ----------------
print("######## A. reasoning_content（chat.py 回传思考对不对）########\n")

def msgs(with_reasoning):
    a = {"role": "assistant", "content": C1}
    if with_reasoning:
        a["reasoning_content"] = R1
    return [{"role": "user", "content": U1}, a, {"role": "user", "content": "Now divide by 17."}]

show("A1 回传 reasoning_content   <- chat.py 现在这么做", msgs(True))
show("A2 不回传                    <- 思考会整段变空", msgs(False))

# ---------------- B: effort injection ----------------
print("######## B. reasoning_effort 往 system 里塞什么 ########\n")

user_only = [{"role": "user", "content": U1}]
show("B1 没有 system 消息, 不给 effort (模板 default = xhigh)", user_only)
show("B2 没有 system 消息, effort=xhigh", user_only, {"reasoning_effort": "xhigh"})
show("B3 没有 system 消息, effort=low", user_only, {"reasoning_effort": "low"})
show("B4 没有 system 消息, effort=medium   <- 新配置", user_only, {"reasoning_effort": "medium"})

try:
    txt = io.open(SYS_TXT, encoding="utf-8-sig").read().strip()[:60]
except OSError:
    txt = "You are a helpful assistant."
with_sys = [{"role": "system", "content": txt}, user_only[0]]
show("B5 带 system, effort=xhigh  <- 旧配置：system = effort句 + 我们的文本", with_sys,
     {"reasoning_effort": "xhigh"})
show("B6 带 system, effort=medium <- 新配置：system = 我们的文本，一字不多", with_sys,
     {"reasoning_effort": "medium"})
