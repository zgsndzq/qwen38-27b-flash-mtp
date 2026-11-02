# 3D 测试提示词

服务端 `llmserver3d.bat`（Qwen3.5-4B-3D-Coder Q8_0，18080），客户端 `llmchat3d.bat`。
System prompt 由 `sys3d.txt` 提供，这里只放要粘贴进对话的用户消息。

采样参数固定在服务端：`--temp 0.7 --top-p 0.95 --top-k 20 --repeat-penalty 1.0`，
客户端 `config.3d.json` 只覆盖 temperature（同样是 0.7），不用再传。

没有参考图的纯文本测试要关掉场景记忆，否则 `mem/scene/scene-convention.md`
会让它先去列"参考图里看到的证据"，而这次并没有参考图：

    python chat.py 3d --mem 0

输出保存成 `.html` 双击即可，Three.js 走 cdnjs，需要联网。

## 1. 龙骑士海蚀柱（复杂体素）

2026-11-02 首次跑通：11003 token / 254 s（约 45 t/s），成品
`3d-demo/scene1-dragonrider.html`。模型原输出有一处重复 `const shoreY`，
整个 module 直接 SyntaxError 白屏，改名后正常——验收时先看这一项。

```text
Build a self-contained HTML scene with Three.js from cdnjs featuring a
dragonrider aerie on a sea stack, with perches, a hatchery, and a windlass.

The aerie occupies a flat-topped sea stack, reached by a rope windlass from
the shore. Four projecting stone perches ring the central hatchery, a
vaulted cave with sand nests and a smoke hole. A timber landing deck holds
a hoist crane and harness racks, while switchback stairs connect the
perches to a water-level boathouse.

Render it as a deliberate 3D voxel model built from visible cubic blocks,
with clear depth, readable silhouettes, and solid spatial relationships.
```

### 验收点

- 页面能直接打开，不是白屏（白屏基本等于 SyntaxError，控制台一查就有）
- 不引 addons / examples/js，相机控制写在文件内
- 鸟巢、四个石台、绞盘、栈道确实都在，且有前后层次，不是平铺
- 由可见方块构成体素感，不是光滑几何体堆叠

## 2. 待补

官方还留了 ATC 塔（带动画）和运河船闸（一句话）两个场景，提示词在
`../../3d-demo/task.md`，验证完第 1 条再往里加。
