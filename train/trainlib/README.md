# trainlib — 代码训练库

从本机 `F:\github` 的每个项目里**分别采样**一些真实文件，按类型分类成目录树，
供后续**不断补充**。采集口径偏向"这个模型实际要写的东西"：
**Python / TS / JS / CSS / HTML / Markdown（文档说明）/ SQL / Shell / 配置 / 标记语言**。

## 现状（2026-09-22）

| 项 | 值 |
|---|---|
| 文件数 | 980 |
| 体积 | 8.79 MB |
| 覆盖项目 | 21 / 23（`superai` 为空目录） |
| 语料 token 数 | 2,799,728（用模型自己的 tokenizer 数的） |
| bigram 类型 | 512,607 |

分类分布：`markdown 183 / python 147 / js 141 / ts 129 / config 126 / html 73 / shell 60 / other 55 / css 37 / markup 18 / sql 11`

## 目录结构

```
trainlib/
  collect.py         采集器（可重复运行，只补新的）
  build_cache.py     把训练库编译成 llama-server 的静态 n-gram 缓存
  MANIFEST.csv       全量索引：src, project, category, ext, bytes, lines, sha1, dest
  STATS.json         分类 / 分项目统计
  README.md
  <category>/<project>/<原相对路径>
  lookup/            编译产物 *.lngc + 同名 .json 元信息（采集器会跳过）
  _tools/            采集侧小工具（采集器会跳过）
```

分类桶：`python ts js css html markdown sql shell config markup other`。
**每个文件在库里只存一份**，项目名在路径第二级，所以按类型看和按项目看都能用。

## 怎么继续补充

新项目：把仓库放到 `F:\github\` 下，然后重跑采集器即可（已采过的按 `src` + `sha1` 跳过）：

```powershell
python trainlib\collect.py
```

想加量就放开配额（默认每项目每扩展名 12 个、每项目上限 150 个）：

```powershell
python trainlib\collect.py --per-ext 30 --per-project 400
python trainlib\collect.py --fresh          # 清空重建
python trainlib\collect.py --projects claw,pi,openjev
```

采集口径（改 `collect.py` 顶部的常量）：跳过 `node_modules/dist/build/vendor/.venv`
等目录、跳过 minified/`*.min.js`/lockfile/`.env`/`.pem`/密钥、跳过非 UTF-8、
跳过单行 >500 字符（机器压缩产物）与 >300 KB 的文件；全局按 sha1 去重。
配额里的中文注释不再赘述——看代码里的 `CAT_CAP` / `EXT_CAP`。

## 下游消费者

1. **`build_cache.py` → `lookup/*.lngc`**
   llama-server 的 lookup 解码静态表（见 `docs/JEV.md`）。表的粒度是 **token bigram**
   （引擎只按 2 个 token 查表），并且只保留"续接足够集中"的 bigram。

   ```powershell
   python trainlib\build_cache.py            # 同时产出 precise + wide
   ```

   ⚠️ **改动这张表之后，必须 grep 服务日志确认它真的加载了**，不能只看服务起来没起来：
   加载失败在这套代码里**只是一行 `W`**，服务照常启动、`/health` 照常 200、生成照常正常，
   引擎会**静默退化成"只有上下文表"**。证据行：

   ```
   I common_ngram_cache_load_file: ... has no header (older format), vocab identity not checked, 41587 n-grams
   I spec common_specu: static lookup cache loaded: ...\precise.lngc (41587 n-grams)
   ```

   表用**无头布局**写（`ngram-cache.cpp:397-416` 支持的 older format）：v2 头的词表指纹校验
   在本机实测对不上（两边都数 248320 个 token，但哈希不同），而拒收只是警告 ⇒ 会造成静默失效。
   无头布局跳过指纹、**但保留 token id 的词表范围校验**；指纹记在同名 `.json` 里备查。

2. **词表短表**（`..\vocab\`）：统计代码里高频命中的 token，选短表。
   训练库比生成语料更贴真实用法。

## 注意

- 采集的是本机自己 clone 的开源代码，**不是**去网上爬。库里的文件是副本，
  项目更新后重跑采集器只会新增（按 `src` 判断），不会同步删除——要同步就 `--fresh`。
- `MANIFEST.csv` 是唯一可信索引；手工往目录里丢文件不会进索引，下游会漏掉。
  要手工加，就补一行 manifest，或者直接重跑采集器。
