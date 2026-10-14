# APEX on Strata

在 [cyber-on-strata](https://github.com/1314521gjy/cyber-on-strata)（Strata 引擎 fork，MIT）基础上，
让 **APEX 量化的 Qwen3.6-35B-A3B**（如 `Qwen3.6-35B-A3B-APEX-MTP-Compact`）也能跑 Strata 的
专家流式推理。

**实测**（RTX 2060 6GB / i7-9700 / 32GB RAM）：decode **19–23 tok/s**，专家缓存命中率 **0.80**，
比 llama.cpp（同模型 `--n-cpu-moe` 分层卸载）约 6–12 tok/s 快 2 倍以上。

## 与上游的差异（本 fork 改了什么）

| # | 文件 | 改动 | 原因 |
|---|---|---|---|
| 1 | `src/kernels/cuda/iq_kernels.cu` | 移植 llama.cpp `vec_dot_q3_K_q8_1` + `Fmt<11>`，Q3_K 加入 `STRATA_GU_FMTS` / `STRATA_D_FMTS` / `STRATA_MMVQ_FMTS` | APEX-Compact 的专家是逐层 Q3_K/Q4_K 混合，上游 GPU 专家 kernel 没有 Q3_K 模板 |
| 2 | `src/kernels/cuda/iq_kernels.cu` | `embed_type_supported` 放行 K-quant（Q3_K/Q4_K/Q5_K/Q6_K） | APEX-Compact 的 `token_embd.weight` 是 Q4_K；嵌入 gather 只需要去量化器，`dq_q4_k` 已存在 |
| 3 | `tools/apex_mtp_rt.py`（新增） | 从模型的 blk.40（DeepSeek 式 nextn 块）提取 Strata 草稿器运行时（dense.bin/dense.txt/experts.bin） | 上游 `mtp_rt.py` 只支持 Qwen3.8-Flash-Next 检查点的 donor 格式 |
| 4 | `tools/apex_requant_experts.py`（新增，备选） | 把 Q3_K 专家重排为 Q4_K 的 GGUF 流式重写器 | 若不想改 kernel，可用它重排模型（本 fork 最终走了 kernel 路线，此工具留作备选） |
| 5 | `build-apex.cmd`（新增） | Windows 一键配置+编译（vcvars64 + CUDA 13 + Ninja） | 方便复现 |

## 支持范围

- ✅ `qwen35moe` 架构、专家量化在 GPU 支持列表内（Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/Q2_0/IQ 系）的 GGUF
- ✅ 带或不带 MTP 块（blk.40）都可以；不带 MTP 时去掉 `--mtp` 参数即可，性能无损失
- ❌ **APEX-I-Mini 等带 `.scale` / `.input_scale` 侧挂校正张量的量化**：Strata 直接 mmap 原始
  量化块，不读这些平面，跑了结果会是错的。需要引擎级 kernel 改动（另一个量级的工作）
- 引擎会在加载前校验（`check_experts_gguf`），不兼容会明确报错而非输出乱码

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

MTP 草稿（可选，目前接受率为 0，不推荐）：

```bat
python tools\apex_mtp_rt.py --gguf <带 blk.40 的模型.gguf> --out mtp-dir
set STRATA_MTP_PLAIN=1
:: 上面的命令加 --mtp mtp-dir --mtp-max-t 1
```

## 参数提示

- `--vram-reserve-mib`：唯一必须按显卡调的参数。6GB 卡 500–800；8GB 卡 1500
- 内存需求：专家文件 ~14.8GB 需常驻内存（pinned），24GB 机器紧张但可跑（`STRATA_ARENA_PIN_GIB` 可手动限制）
- `--dump-routing` 与 `--spec 2` 可同用，但必须已挂 profile（用 seed profile 先跑第一轮）

## 已知问题

- MTP 草稿头：`apex_mtp_rt.py` 的 `eh_proj` 分支映射或 Q2_0 重排精度待修正，
  草稿被提出但接受率 0。修好前 `--mtp` 是纯开销
- Q3_K kernel 数值只做了"输出合理"级验证，未做与 llama.cpp 的逐 token 贪心对照
  （上游 `docs/cyber/04-correctness.md` 的流程适用）

## 许可

与上游一致（MIT）；模型权重遵循其各自的发布条款。
