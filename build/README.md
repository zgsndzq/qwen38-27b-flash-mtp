# build — 这里放引擎二进制

这个目录**不进 git**，因为二进制有两条不同的来路。

## 路线 A：用预编译的（推荐，不用装编译环境）

去 [Releases](https://github.com/zgsndzq/llama.cpp-qwen38-flash/releases) 下
`qwen38-27b-flash-win64-sm89-*.zip`，解压出来的内容放到这里：

```
build\
├── bin\          llama-server.exe + 全部 dll
└── cuda\bin\     CUDA 运行时 dll（cublas / cudart / nvrtc …）
```

**不用装 CUDA Toolkit**，`cuda\bin` 里的 dll 会被启动脚本加进 PATH。

## 路线 B：自己编（改了引擎、或者显卡不是 sm_89）

```bat
git clone https://github.com/zgsndzq/llama.cpp-qwen38-flash code
copy code\build.local.example.bat code\build.local.bat
notepad code\build.local.bat        :: 填 6 个值，文件里有注释
code\_build_q8.bat
```

耗时约 20–60 分钟，产物落在 `build\src\bin\`。

需要装：**VS2022（MSVC）+ CUDA Toolkit + CMake + Ninja**。
`build.local.bat` 里 `CUDA_ARCH` 要填**你自己显卡的算力** —— 4060/4070/4090 = `89`，
3060/3080 = `86`。这是唯一一个"填错了编译能过、但跑不起来"的值。

日志在 `build\logs\_build_q8.log`，最后一行是 `DONE` 就成功了。

> 忘了建 `build.local.bat` **不会静默出错**：脚本会写 `MISSING build.local.bat`
> 然后以退出码 9 结束。

## 两条路线可以共存

`llmserver.bat` 两个位置都认，**优先用你自己编的**：

| 位置 | 谁放的 |
|---|---|
| `build\src\bin\` | 路线 B，`code\_build_q8.bat` |
| `build\bin\` | 路线 A，解压 Release |
