# APEX on Strata

> **Fork 说明**：本仓库 fork 自 [1314521gjy/cyber-on-strata](https://github.com/1314521gjy/cyber-on-strata)
> （Strata 引擎的 Cyber-Tiel-Coder fork，MIT）。`main` 分支保留上游原样；
> **`apex` 分支是全部改动**，本文件描述的就是 `apex` 分支的内容。
>
> **预编译版**：见 [Releases](../../releases) —— 下载 zip 解压即用（含 strata.exe + APEX 工具 + 使用说明），
> 只需自备模型 GGUF 和 CUDA 13 运行时 DLL。

在 [cyber-on-strata](https://github.com/1314521gjy/cyber-on-strata)（Strata 引擎 fork，MIT）基础上，
让 **APEX 量化的 Qwen3.6-35B-A3B**（如 `Qwen3.6-35B-A3B-APEX-MTP-Compact`）也能跑 Strata 的
专家流式推理。

## 实测（RTX 2060 6GB / i7-9700 / 32GB RAM）

| 模型 | 配置 | decode | 专家缓存命中 |
|---|---|---|---|
| APEX-MTP-Compact | 无 MTP | 19–23 tok/s | 0.80 |
| APEX-MTP-Compact | **MTP（spec 2）** | **25–29 tok/s** | 0.59–0.65 |
| **APEX-I-Mini**（IQ2_S down + Q3_K，无 MTP） | 无 MTP | **18.3 tok/s** | **0.85** |

Mini 是 13.3GB 文件（比 Compact 小 2.8GB），内存紧张的机器（24GB 笔记本）首选。
两者都比 llama.cpp 同模型（6–12 tok/s）快 1.5–3 倍。

## 与上游的差异（本 fork 改了什么）

| # | 文件 | 改动 | 原因 |
|---|---|---|---|
| 1 | `src/kernels/cuda/iq_kernels.cu` | 移植 llama.cpp `vec_dot_q3_K_q8_1` + `Fmt<11>`，Q3_K 加入 `STRATA_GU_FMTS` / `STRATA_D_FMTS` / `STRATA_MMVQ_FMTS` | APEX 的专家是逐层 Q3_K/Q4_K 混合，上游 GPU 专家 kernel 没有 Q3_K 模板 |
| 2 | `src/kernels/cuda/iq_kernels.cu` | IQ2_S 加入 `STRATA_D_FMTS`；`embed_type_supported` 放行 K-quant | APEX-I-Mini 的 down 投影是 IQ2_S；`token_embd.weight` 是 Q4_K |
| 3 | `include/strata/artifact/gguf_reader.hpp` | 架构守卫改为结构性检查：block_count 40（纯主干）和 41（含 MTP 块）都合法，并用 MTP 张量的存在性交叉验证 | 上游硬编码 41，拒绝无 MTP 的 APEX-I-Mini 文件 |
| 4 | `tools/apex_mtp_rt.py`（新增） | 从模型的 blk.40（DeepSeek 式 nextn 块）提取 Strata 草稿器运行时（dense.bin/dense.txt/experts.bin），按专家流式转换 | 上游 `mtp_rt.py` 只支持 Qwen3.8-Flash-Next 检查点的 donor 格式 |
| 5 | `tools/apex_requant_experts.py`（新增，备选） | 把 Q3_K 专家重排为 Q4_K 的 GGUF 流式重写器 | 若不想改 kernel，可用它重排模型（本 fork 最终走了 kernel 路线，此工具留作备选） |
| 6 | `build-apex.cmd`（新增） | Windows 一键配置+编译：自动找 VS（vswhere）和 CUDA Toolkit，架构作参数传入 | 方便复现，无硬编码路径 |

## 支持范围

- ✅ **APEX-MTP-Compact**：Q3_K/Q4_K 专家 + Q4_K 嵌入；MTP 可用（接受率 0.75）
- ✅ **APEX-I-Mini**：Q3_K/IQ2_S 专家 + Q4_K 嵌入，无 MTP（IQ2_S down 路径已加入 GPU kernel）
- ✅ 一般 `qwen35moe` 架构、专家量化在 GPU 支持列表内（Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/Q2_0/IQ 系）的 GGUF
- ✅ 带或不带 MTP 块（blk.40）都可以；不带 MTP 时去掉 `--mtp` 参数即可，性能无损失。
  架构守卫按 MTP 张量是否存在做结构性校验（block_count 40/41 都合法）
- 引擎会在加载前校验（`check_architecture` + `check_experts_gguf`），不兼容会明确报错而非输出乱码

## 使用步骤（Windows）

前置：NVIDIA GPU（sm_75+）、CUDA Toolkit 13.x、VS 2022+（C++/CUDA）、Python 3.10+（`pip install gguf numpy`）。

```bat
:: 1. 编译（sm_75;86 覆盖 RTX 20/30 系，按需增删）
build-apex.cmd

:: 2. 打 native pack（对任何支持的 GGUF 一次即可）
set STRATA_GGUF_PY=<pip 的 gguf 包目录的父目录>
python tools\iq_pack.py --gguf <模型.gguf> --out pack-apex-compact --compat-bf16

:: 3. 生成 expert profile（命中率 2.97% -> 80% 的关键，必做）
::    a) 先造一个空 profile（身份排序）
python tools\make_profile.py --no-base --n-layer 40 --n-expert 256 --out profile-seed.bin
::    b) 用 8 个不同主题的 prompt 各跑一次 --dump-routing（每个 trace ~550KB）
::    c) 合成
python tools\make_profile.py trace1.bin ... trace8.bin --no-base --n-layer 40 --n-expert 256 --out profile-apex-compact.bin

:: 4. 运行（先 tokenize：可用任意 llama.cpp 的 llama-tokenize --ids）
build-apex\strata.exe --pack pack-apex-compact --native <模型.gguf> --no-ple ^
  --tokens <ids> --expert-profile profile-apex-compact.bin ^
  --expert-cache auto --vram-reserve-mib 800 --prefill 2048 --max-context 4096 ^
  --max-new 128 --kv int8 --kv-resident 0 --spec 2 --suffix-draft 0 ^
  --pool-workers 6 --pool-affinity all --stats
```

MTP 草稿（可用，`spec 2` 实测接受率 0.44–0.46，decode 25–26 tok/s）：

```bat
python tools\apex_mtp_rt.py --gguf <带 blk.40 的模型.gguf> --out mtp-dir
set STRATA_MTP_PLAIN=1
:: 上面的命令加 --mtp mtp-dir，并注意：
::  - 不要传 --mtp-max-t（它会把草稿数清零，"drafts accepted 0 of 0" 即此因）
::  - spec 3 更慢（起草成本 8.8ms/轮超过收益），spec 2 是甜点
```

MTP 移植要点（`tools/apex_mtp_rt.py` 的映射依据，llama.cpp `qwen35moe.cpp` @ 3cf03257f）：

- `eh_proj` 是 `(N, 2N)`，输入为 `concat(e_norm, h_norm)` —— 左半列是 embedding 分支
- **norm 是普通 RMSNorm（`LLM_NORM_RMS`），不要加 Gemma 式 "+1"**（那是对旧 donor 检查点的，
  加了就是接受率 0 的来源）
- 草稿专家按引擎 Q2_0 blob 布局重排（Q8_0 源重排，有轻微精度损失，接受率 ~0.45 可接受）

## 参数提示

- `--vram-reserve-mib`：唯一必须按显卡调的参数。6GB 卡 500–800；8GB 卡 1500
- 内存需求：专家文件 ~14.8GB 需常驻内存（pinned），24GB 机器紧张但可跑（`STRATA_ARENA_PIN_GIB` 可手动限制）
- `--dump-routing` 与 `--spec 2` 可同用，但必须已挂 profile（用 seed profile 先跑第一轮）

## 已知问题

- MTP 草稿接受率 0.75（上游 Cyber 参考值 0.55–0.58，本模型更高）——剩余损失来自 Q8_0→Q2_0
  草稿专家重排；若追求更高接受率需给草稿器加 Q8_0 blob 路径（引擎改动）
- `--spec 3` 及以上不划算（起草成本 8.8ms/轮超过收益），用 `--spec 2`
- Q3_K / IQ2_S kernel 数值只做了"输出合理"级验证，未做与 llama.cpp 的逐 token 贪心对照
  （上游 `docs/cyber/04-correctness.md` 的流程适用）
- Q3_K 专家的 prefill 走非 MMQ 回退路径（MMQ 表无 Q3_K），长 prompt 填充速度有提升空间

## 许可

与上游一致（MIT）；模型权重遵循其各自的发布条款。
