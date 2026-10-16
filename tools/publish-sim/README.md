# publish-sim — 发布演练

用两个脚本在本地把"发布 -> 下载 -> 跑起来"整条链路走一遍，不需要真的推到 GitHub / Hugging Face。

```
make_release.py     造出三个"远端"渠道      F:\super\_release_out\
simulate_user.py    建一台干净的用户机      F:\super\duan\
```

## 怎么跑

```bash
python tools/publish-sim/make_release.py     # 约 1 分钟 + 打包 50 秒
python tools/publish-sim/simulate_user.py    # 约 1 分钟
```

然后双击 `F:\super\duan\llama.cpp-qwen38-flash\llmserver.bat`（权重已就位）。

## 两个脚本各自做什么

**make_release.py** 模拟三个发布渠道：

| 产物 | 模拟 | 说明 |
|---|---|---|
| `_release_out\hf-model\<repo>\` | Hugging Face model repo | gguf 用**硬链接**，不占 29 GB |
| `_release_out\github-src.git\` | GitHub 源码仓 | 裸 clone，模拟用户 `git clone` 的目标 |
| `_release_out\releases\llama.cpp-qwen38-flash-win64-*.zip` | GitHub Releases | `build\bin` + `build\cuda` + runtime 配置 |

**simulate_user.py** 模拟使用者：

1. 解压 Release zip 到 `duan\llama.cpp-qwen38-flash\`
2. 从 HF 目录取权重到 `llama.cpp-qwen38-flash\weights\`
3. `git clone` 源码到 `duan\code\`（场景 B 用）
4. **隔离性自检**：扫 `duan\` 里还有没有指向 `qwen27b` / `adaq8` 的引用

第 4 步是这轮演练最有价值的输出 —— 它把"内部痕迹泄漏到发布物"这件事变成一条可看的清单。

## 场景 B（自己编译）

脚本不自动跑编译（20 分钟），要手动：

```bat
cd F:\super\duan\code
copy build.local.example.bat build.local.bat
notepad build.local.bat          :: 填 6 个值，模板里有注释
_build_q8.bat
```

产物在 `duan\build\src\bin\`，覆盖到 `duan\llama.cpp-qwen38-flash\build\bin\` 就能用。

## 注意

- 源和目标路径写死在两个脚本顶部（`Q` / `DUAN` / `OUT`），换机器要改。
- 脚本会**清空**目标目录 —— `simulate_user.py` 开头就把 `F:\super\duan` 整个删掉重建，别拿它当还原点。
- 真实发布时权重走 HF 上传，不走硬链接；这里是唯一为了省空间的简化。
