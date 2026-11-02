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

## 2. 动画 ATC 塔

2026-11-02 用 MTP 版权重跑通：8072 token / 92 s（88.5 t/s），成品
`3d-demo/scene2-atc-tower.html`。语法检查通过，但有 3 处运行期错误，全在
animate/相机上，已修：

1. `panelRotation.clone().multiplyScalar(-1)` —— `Euler` 没有 `multiplyScalar`，
   改成 `new THREE.Euler(0.25, 0, 0)`；
2. `angles[radarTargetIndex]` 用浮点索引（递增 0.035），取到 `undefined` 再
   `.copy()` 就报 `reading 'x'`，改成 `Math.floor(...) % length`；
3. **相机从没设过位置**，默认停在原点＝塔基内部，画面只有远处跑道灯。补
   `camera.position.set(34, 24, 40)` + `lookAt(0, 15, 0)`。

```text
Make a standalone HTML scene with Three.js from cdnjs depicting an animated
air traffic control tower with rotating radar and light signals.

A tapered control tower rises from a rectangular base to a cantilevered
glass cab with angled glazing panels and an external catwalk. A primary
surveillance radar on a roof pedestal turns continuously, and a rotating
beacon sits above the cab.
```

### 验收点

- 塔身锥形收分、玻璃塔台外挑、外部走道、屋顶雷达 + 旋转信标都在
- 雷达确实在转（两帧截图逐字节比对有变化，不是静态摆件）
- 相机默认视角能一眼看全塔，不需要拖鼠标才找到模型
- 整体偏暗：`toneMappingExposure` 只有 1.1，夜景风格，看细节要调亮

## 3. 待补

运河船闸（一句话提示）还没跑，提示词在 `../../3d-demo/task.md`。
