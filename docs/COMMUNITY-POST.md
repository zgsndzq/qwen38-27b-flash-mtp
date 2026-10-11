# 社区分享文案（中英双语）

> 给技术社区/推特/群分享用。两版都在 200 字上下，只讲结论、可复现。

---

## 中文版

**8 GB 笔记本上，让 27B 模型同时当聊天和 JEV 判据引擎**

硬件：RTX 4060 Laptop 8 GB。模型：Qwen3-27B，三值量化（PTQ1_0）后权重仅 6.21 GB——这是装进 8 GB 卡的前提。

只跑一个 llama-server，对话和 **JEV** 共用同一份权重、同一个 `/v1/chat/completions`。**JEV 是一种零输出的判据接口：不生成任何 token**，用 `max_tokens:1 + logprobs + top_logprobs:40` 直读最后一层 logits，在候选标签上 softmax 取最大——约 0.5 s 返回一个带概率的判定，可做路由、判定、分类、选动作；对话约 50 t/s，prefill 540 t/s。两者共享主机内存 prompt cache，来回切换只重算 1 个 ubatch（冷启 19.7 s → 切换 0.5 s），互不串味。

集成四件事：三值量化、自训 MTP 推测头、24k 草稿短表、q4_0 KV 偏置标定。复现：llama.cpp（PrismML prism 分支）+ 上述参数，单卡即可。

---

## English

**One 27B model, two jobs — chat + JEV decision engine — on an 8 GB laptop**

Hardware: RTX 4060 Laptop, 8 GB VRAM. Model: Qwen3-27B, ternary-quantized (PTQ1_0) down to 6.21 GB of weights — the only reason it fits.

A single llama-server. Chat and **JEV** share the same weights and the same `/v1/chat/completions`. **JEV is a zero-output decision interface: it emits no tokens.** `max_tokens:1 + logprobs + top_logprobs:40` reads the final-layer logits directly and runs softmax + argmax over the candidate labels — one probability-scored decision in ~0.5 s (routing, judging, classification, action picking) — while chat runs ~50 t/s and prefill hits 540 t/s. Both share the host-RAM prompt cache, so switching between them costs one ubatch (19.7 s cold → 0.5 s on switch), with no cross-talk.

Four pieces integrated: ternary weights, a self-trained MTP speculative head, a 24k draft-vocab shortlist, and q4_0 KV bias calibration. To reproduce: llama.cpp (PrismML `prism` branch) + the params above.
