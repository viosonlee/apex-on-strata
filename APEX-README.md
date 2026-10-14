# APEX on Strata

> **Fork 说明**：本仓库 fork 自 [1314521gjy/cyber-on-strata](https://github.com/1314521gjy/cyber-on-strata)
> （Strata 引擎的 Cyber-Tiel-Coder fork，MIT）。`main` 分支保留上游原样；
> **`apex` 分支是全部改动**，本文件描述的就是 `apex` 分支的内容。
>
> **预编译版（推荐）**：见 [Releases](../../releases) 的 `v0.1.0-apex` —— 含 strata.exe（sm_75+sm_86、
> CUDA 13 DLL）、启动器 GUI（LlamaManager.exe）、APEX 工具、现成 pack/profile/MTP runtime 和
> [QUICK-START.txt](QUICK-START.txt)。**想跑起来先读 QUICK-START.txt，不必从源码编译。**
> 配套启动器仓库：[viosonlee/llamacppLauncher](https://github.com/viosonlee/llamacppLauncher)。

在 [cyber-on-strata](https://github.com/1314521gjy/cyber-on-strata)（Strata 引擎 fork，MIT）基础上，
让 **Qwen3.6-35B-A3B（qwen35moe）家族的各种量化 GGUF** —— APEX、UNSLOTH 等不同量化的
Compact / Mini / Occamy 变种 —— 都能跑 Strata 的专家流式推理。支持范围见下文，
不限于下列三个已验证样本。

## 快速开始（不编译）

1. 下载 Release `v0.1.0-apex` 的 zip，解压（engine + GUI + tools + pack/profile/MTP 都在包内）。
2. 准备模型 GGUF（包内 `models\` 目录或任意路径）。**每个 GGUF 需要自己的 pack**；
   包内已带 Occamy 的 `pack-occamy`，其它变种用 GUI 的"一键生成 Pack"或下面的命令行生成。
3. NVIDIA 驱动需支持 CUDA 13（2025 年以后的 Game Ready / Studio 驱动均可）。
   无需 CUDA Toolkit、Visual Studio、.NET；Python 3.10+ 仅在为新模型生成 pack 时需要。
4. 用 GUI（`LlamaManager.exe`，Strata 标签页）或 QUICK-START.txt 里的命令行运行。

已验证的两类输入：固定提示词下 Strata 与 llama.cpp 的输出 token 完全一致（Occamy 128/128、
Compact 118/118）；kernel 数值对照（Q3_K、IQ2_S、IQ3_XXS 的解量化/MMVQ/MMQ + compute-sanitizer
0 错误）。注意：同模型不同量化、不同 batch 切分不保证逐位一致；重复运行中也观察到过输出分叉
（专家缓存 GPU/CPU 舍入差异，上游已知），不可把单次一致当作所有配置下逐位一致的证明。

## 实测（RTX 2060 6GB / i7-9700 / 32GB RAM，2026-10-14）

| 模型 | 配置 | decode | prefill | 备注 |
|---|---|---|---|---|
| APEX-MTP-Compact | 无 MTP | 14–23 tok/s | ~740 tok/s（Q3_K MMQ） | 命中 0.82 |
| APEX-MTP-Compact | **MTP spec 2** | **25–32 tok/s** | 同上 | 接受率 0.75–0.86 |
| APEX-MTP-Compact | MTP spec 4 | 31–37 tok/s | 同上 | 接受率 0.58–0.65，吞吐更高 |
| Occamy-1.0 MiniPlus | 无 MTP | 16–19 tok/s | 48–61 tok/s | 命中 0.75–0.77 |
| Occamy-1.0 MiniPlus | **接 Compact MTP spec 2** | 最高 ~27 tok/s | 同上 | 接受率 0.84–0.90，**跨模型草稿，实验性** |
| Occamy（对照） | llama.cpp build 11149 | 22–25 tok/s | 18–51 tok/s | `--n-cpu-moe 32`；**该配置下 Strata 略慢** |

- 同模型逐 token 对照：Occamy/Compact 固定输入下与 llama.cpp 输出完全一致。
- **Occamy + Compact MTP 是跨模型草稿**：接受率高、可跑，但属于实验性用法，收益未在多提示词上验证。
- spec 4 接受率低于 spec 2 但吞吐更高——优化看端到端速度，别只看接受率。
- Occamy 的 20/40 层专家是 IQ3_XXS（type 18）：本 fork 已把它加入 down 投影（`STRATA_D_FMTS`）。
- profile 校准实验（3 条独立 trace）未证明普遍收益；扫参中 6 workers + reserve 500 MiB 略优，
  7 workers 更慢。线程数不是越多越快。

**Q3_K prefill MMQ**：Q3_K 专家层的 prompt 从 fp16 去量化回退（~370–440 tok/s）提到
**~740–750 tok/s，约 2 倍**（`mmq-instance-q3_k.cu` + `moe_mmq.cu` 两个 switch）。
MMQ 与回退路径不是位级相同（上游已注明 batch split 非数值中性）。

## 与上游的差异（本 fork 改了什么）

| # | 文件 | 改动 | 原因 |
|---|---|---|---|
| 1 | `src/kernels/cuda/iq_kernels.cu` | 移植 llama.cpp `vec_dot_q3_K_q8_1` + `Fmt<11>`，Q3_K 加入 `STRATA_GU_FMTS` / `STRATA_D_FMTS` / `STRATA_MMVQ_FMTS`；IQ2_S 加入 `STRATA_D_FMTS` | APEX/Occamy 的专家是逐层 Q3_K/Q4_K/IQ3_XXS 混合，上游 GPU 专家 kernel 没有这些模板 |
| 2 | `src/kernels/cuda/iq_kernels.cu` | **IQ3_XXS（type 18）加入 `STRATA_D_FMTS`** | Occamy 第 10–29 层 gate/up/down 全是 IQ3_XXS；缺失时 down 走 fp16 回退慢路径 |
| 3 | `src/prefill/moe_mmq.cu` + `CMakeLists.txt` | Q3_K 加入 prefill MMQ（`mmq-instance-q3_k.cu` + supported/run 两个 switch） | ~31/40 层的 prompt 从 fp16 回退（370-440 tok/s）提到 MMQ 的 ~740-750 tok/s |
| 4 | `include/strata/artifact/gguf_reader.hpp` | 架构守卫改为结构性检查：block_count 40（纯主干）和 41（含 MTP 块）都合法，并用 MTP 张量的存在性交叉验证 | 上游硬编码 41，拒绝无 MTP 的模型文件 |
| 5 | `tools/apex_mtp_rt.py`（新增） | 从模型的 blk.40（DeepSeek 式 nextn 块）提取 Strata 草稿器运行时（dense.bin/dense.txt/experts.bin），按专家流式转换 | 上游 `mtp_rt.py` 只支持 Qwen3.8-Flash-Next 检查点的 donor 格式 |
| 6 | `tools/apex_requant_experts.py`（新增，备选） | 把 Q3_K 专家重排为 Q4_K 的 GGUF 流式重写器 | 若不想改 kernel，可用它重排模型（本 fork 最终走了 kernel 路线，此工具留作备选） |
| 7 | `build-apex.cmd`（新增） | Windows 一键配置+编译：自动找 VS（vswhere）和 CUDA Toolkit，架构作参数传入 | 方便复现，无硬编码路径 |

## 支持范围

**架构**：`qwen35moe` 的 35B-A3B 家族（40 层主干，含 MTP 块时 41 层；hidden=2048、
attention heads=16 / KV heads=2、256 专家 top-8）。**不限于某个单一 GGUF 文件**——
同一架构的 APEX、UNSLOTH、GSQ 等各种量化/微调变种都在范围内，每个 GGUF 只需生成一次 pack。

**专家量化格式（GPU kernel 已覆盖并数值验证）**：

| 格式 | GGML type | 说明 |
|---|---|---|
| Q3_K | 11 | gate/up 和 down 都支持（MMVQ + prefill MMQ） |
| Q4_K | 12 | 上下投影均支持 |
| Q5_K / Q5_1 / Q6_K | 13/14/15 | K-quant 家族（`STRATA_MMQ_KQUANTS`） |
| Q8_0 | 8 | 支持草稿层稠密矩阵 |
| Q2_0 | 42 | 本仓库自有格式 |
| IQ2_XXS / IQ2_XS / IQ2_S | 16/17/22 | i-quant；IQ2_S down 已加入 `STRATA_D_FMTS` |
| **IQ3_XXS** | **18** | gate/up 和 down 都支持（Occamy 第 10–29 层在用） |
| IQ3_S / IQ4_NL / IQ4_XS | 21/20/23 | i-quant 家族 |
| IQ1_M | 29 | gate/up 支持（down 除外） |

非专家权重（注意力投影、router、GDN/SSM、嵌入、LM head）接受 F32/F16/BF16 及上述量化。
模型不带 MTP 层时去掉 `--mtp` 即可，无性能损失；带 blk.40 的模型可用 `apex_mtp_rt.py` 提取草稿。

**已验证样本**（不是全集，只是测过的）：

| 模型 | 专家布局 | 状态 |
|---|---|---|
| Qwen3.6-35B-APEX-MTP-Compact | Q3_K + Q4_K 混合，41 层含 MTP | 完整验证（含 MTP 0.75–0.86） |
| Qwen3.6-35B-APEX-I-Mini | Q3_K + IQ2_S down，无 MTP | 验证（18.3 tok/s，命中 0.85） |
| Occamy-1.0.APEX-I-MiniPlus-V2.1 | Q3_K + IQ3_XXS，GDN/SSM 混合层 | 验证（含跨模型 MTP 实验） |

其它同架构量化变种**预期可用但未逐个实测**：生成 pack 时若专家量化不受支持，
`iq_pack.py` 会明确报错而不是产出坏 pack；引擎加载前也会做架构与张量校验。

- 引擎会在加载前校验（`check_architecture` + `check_experts_gguf`），不兼容会明确报错而非输出乱码

## 使用步骤（Windows，从源码）

前置：NVIDIA GPU（sm_75+）、CUDA Toolkit 13.x、VS 2022+（C++/CUDA）、Python 3.10+（`pip install gguf numpy regex`）。

```bat
:: 1. 编译（sm_75;86 覆盖 RTX 20/30 系，按需增删）
build-apex.cmd

:: 2. 打 native pack（对任何支持的 GGUF 一次即可；pack 必须与该 GGUF 对应）
set STRATA_GGUF_PY=<pip 的 gguf 包目录的父目录>
python tools\iq_pack.py --gguf <模型.gguf> --out pack-<名字> --compat-bf16

:: 3. 生成 expert profile（初始 profile 即可运行；命中率 2.97% -> 80% 的关键）
::    a) 先造一个空 profile（身份排序）
python tools\make_profile.py --no-base --n-layer 40 --n-expert 256 --out profile-seed.bin
::    b) 用 8 个不同主题的 prompt 各跑一次 --dump-routing（每个 trace ~550KB）
::    c) 合成
python tools\make_profile.py trace1.bin ... trace8.bin --no-base --n-layer 40 --n-expert 256 --out profile-<名字>.bin

:: 4. 运行（先 tokenize：可用任意 llama.cpp 的 llama-tokenize --ids）
build-apex\strata.exe --pack pack-<名字> --native <模型.gguf> --no-ple ^
  --tokens <ids> --expert-profile profile-<名字>.bin ^
  --expert-cache auto --vram-reserve-mib 800 --prefill 2048 --max-context 4096 ^
  --max-new 128 --kv int8 --kv-resident 0 --spec 2 --suffix-draft 0 ^
  --pool-workers 6 --pool-affinity all --stats
```

MTP 草稿（模型自带 blk.40 时；固定输入实测接受率 0.75–0.86）：

```bat
python tools\apex_mtp_rt.py --gguf <带 blk.40 的模型.gguf> --out mtp-dir
set STRATA_MTP_PLAIN=1
:: 上面的命令加 --mtp mtp-dir，并注意：
::  - 不要传 --mtp-max-t（它会把草稿数清零，"drafts accepted 0 of 0" 即此因）
::  - spec 2 是甜点；spec 4 接受率更低但吞吐可能更高，看端到端速度
```

MTP 移植要点（`tools/apex_mtp_rt.py` 的映射依据，llama.cpp `qwen35moe.cpp` @ 3cf03257f）：

- `eh_proj` 是 `(N, 2N)`，输入为 `concat(e_norm, h_norm)` —— 左半列是 embedding 分支
- **norm 是普通 RMSNorm（`LLM_NORM_RMS`），不要加 Gemma 式 "+1"**（那是对旧 donor 检查点的，
  加了就是接受率 0 的来源）
- 草稿专家按引擎 Q2_0 blob 布局重排（Q8_0 源重排，有轻微精度损失）

## 参数提示

- `--vram-reserve-mib`：唯一必须按显卡调的参数。6GB 卡 500–800；8GB 卡 1500。
  任务管理器显存占用率高本身不是目标。
- 内存需求：专家文件 ~13–15GB 需常驻内存（pinned），24GB 机器紧张但可跑
  （`STRATA_ARENA_PIN_GIB=12` 起步）。这是主机内存上限，不是显存。
- `--dump-routing` 与 `--spec 2` 可同用，但必须已挂 profile（用 seed profile 先跑第一轮）
- GUI（LlamaManager.exe）的"运行参数"页有每个参数的说明和 6GB 起步预设

## 已知问题

- 跨模型 MTP（Occamy 接 Compact 草稿）接受率高（0.84–0.90）但属实验性：单提示词验证，
  重复运行间观察到输出分叉；不要当作稳定特性依赖
- 重复运行中观察到输出分叉（专家缓存 GPU/CPU 舍入差异，上游已知）；同模型同输入
  与 llama.cpp 的固定提示词对照是一致的
- `--spec 3` 起草成本 8.8ms/轮，是否划算取决于具体模型与提示词（spec 4 在 Compact 上吞吐更高）
- kernel 数值验证已完成（iq_parity 10 格式解量化相对误差 0；Q3_K/IQ2_S/IQ3_XXS 的
  MMVQ/MMQ 对照 + compute-sanitizer 0 错误），但"同模型不同量化/配置"不保证逐位一致

## 许可

与上游一致（MIT）；模型权重遵循其各自的发布条款。
