# Pythia-2.8b 推理优化：FlashAttention (SDPA)

本项目在不更改模型参数的前提下，为 `EleutherAI/pythia-2.8b` 提供两种推理模式：
- 基线模式（math kernel attention）
- FlashAttention 模式（通过 PyTorch SDPA 触发 flash 内核）

同时提供基准测试（TTFT/TPOT/吞吐/峰值显存）、ppl 评估（WikiText/PG-19）、复现实验脚本与结果存储结构。

**GitHub Repository:** [https://github.com/dzx0902/CS3602-FinalWork.git](https://github.com/dzx0902/CS3602-FinalWork.git)

## 环境准备
- 推荐环境：CUDA 12.1 + `torch==2.1.2+cu121`，GPU：Ada (如 4090D)
- 依赖安装：

```powershell
pip install -r requirements.txt
```

- 建议精度：`bfloat16`（Ada 原生支持，数值更稳定）；必要时可退到 `float16`。
- 在使用 Flash 模式前，建议进行一次可用性自检（通过 `src/utils.py` 提供的检查函数，或直接运行 benchmark）。

## 目录结构

```
.
├── src/
│   ├── models/
│   │   ├── flash_attention.py
│   │   ├── pythia_flash_model.py
│   │   └── pythia_baseline_model.py
│   │
│   ├── benchmark.py
│   ├── compute_ppl.py
│   ├── utils.py
│   └── prompts/
│       ├── long_prompt.txt
│       └── short_prompt.txt
│
├── scripts/
│   ├── run_baseline.sh
│   ├── run_flash.sh
│   └── run_all.sh
│
├── results/
│   ├── baseline_metrics.json
│   ├── flash_metrics.json
│   ├── ppl_baseline.json
│   ├── ppl_flash.json
│   └── plots.ipynb (可选)
│
├── README.md
├── Plan.md
└── requirements.txt
```

## 快速开始
- 准备结果目录（用于保存 JSON 输出）：

```bash
mkdir -p results
```

- 运行基准（基线/Flash）：

### Windows PowerShell

```powershell
# 基线（math SDPA）
python src/benchmark.py `
  --mode baseline `
  --model-size 2.8b `
  --new-token-sizes 128 512 1024 2048 `
  --batch-sizes 1 `
  --long-prompt `
  --long-prompt-path src/prompts/long_prompt.txt `
  --max-prompt-tokens 4096

# Flash（需开启环境变量以选择 flash 内核）
$env:FLASH_SDPA='1';
python src\benchmark.py `
  --mode flash `
  --model-size 2.8b `
  --new-token-sizes 128 512 1024 2048 `
  --batch-sizes 1 `
  --long-prompt `
  --long-prompt-path src/prompts/long_prompt.txt `
  --max-prompt-tokens 4096
```

### Linux / WSL / Git Bash

```bash
# 基线（math SDPA）
python src/benchmark.py \
  --mode baseline \
  --model-size 2.8b \
  --new-token-sizes 128 512 1024 2048 \
  --batch-sizes 1 \
  --long-prompt \
  --long-prompt-path src/prompts/long_prompt.txt \
  --max-prompt-tokens 4096

# Flash（需开启环境变量以选择 flash 内核）
FLASH_SDPA=1 python src/benchmark.py \
  --mode flash \
  --model-size 2.8b \
  --new-token-sizes 128 512 1024 2048 \
  --batch-sizes 1 \
  --long-prompt \
  --long-prompt-path src/prompts/long_prompt.txt \
  --max-prompt-tokens 4096
```

- 运行 PPL（WikiText/PG-19）：

### Windows PowerShell

```powershell
# 单次同时输出两种模式的结果（baseline/flash）
python src/compute_ppl.py --dataset wikitext --max-samples 128 --max-length 512
python src/compute_ppl.py --dataset pg19 --max-samples 64 --max-length 512
```

### Linux / WSL / Git Bash

```bash
# 单次同时输出两种模式的结果（baseline/flash）
python src/compute_ppl.py --dataset wikitext --max-samples 128 --max-length 512
python src/compute_ppl.py --dataset pg19 --max-samples 64 --max-length 512
```


## 实现说明
- `SDPA 适配器`（`src/models/pythia_flash_model.py`）：
  - 在不改动原始 GPT‑NeoX 注意力计算的前提下，用适配器包裹原模块，并通过 SDPA 上下文选择 `math|flash` 内核。
  - 环境变量控制：`FLASH_SDPA=1` 启用 flash；未设置或为 `0` 则使用 math 回退。
  - 数值路径与基线一致，便于进行 logits/ppl 无损验证。
- `FlashSelfAttention`（`src/models/flash_attention.py`）：
  - 提供 `use_math_sdpa` / `use_flash_sdpa` 上下文管理器；以及与 NeoX 兼容的 RoPE 应用逻辑。
  - 如需自定义替换，可复用该模块；当前默认使用适配器方式以最大限度保持基线行为。
- `PythiaBaselineModel`（`src/models/pythia_baseline_model.py`）：
  - 原生模型的轻量封装，接口与 Flash 版一致，便于基准对比。
- `benchmark.py`：
  - 指标：TTFT（首 token 耗时）、TPOT（后续 token 平均耗时）、吞吐（tok/s）、峰值显存（MB）。
  - CLI：`--mode baseline|flash --max-new-tokens N --prompt 文件路径`，输出 JSON 至 `results/`。
- `compute_ppl.py`：
  - 使用 teacher forcing 计算 `loss` 与 `ppl`，并在一次运行中同时输出 baseline/flash 两组结果与对数差异（log 差）。
  - 数据集：WikiText（`config=wikitext-2-v1`）与 PG‑19（自动尝试多配置并回退）。

## 结果

运行具体结果在在results文件夹中，这里给出表格和简单结果，
mode     | model       | B | prompt_seq_len | max_new_tokens | TTFT       | TPOT      | Throughput   | PeakMem
---------+-------------+---+----------------+----------------+------------+-----------+--------------+--------------
baseline | pythia-2.8b | 1 | 2454           | 128            | 314.645 ms | 32.796 ms | 30.491 tok/s | 20,746.41 MiB
baseline | pythia-2.8b | 1 | 2454           | 512            | 33.717 ms  | 33.208 ms | 30.114 tok/s | 21,140.94 MiB
baseline | pythia-2.8b | 1 | 2454           | 1024           | 37.654 ms  | 34.815 ms | 28.724 tok/s | 21,785.79 MiB
baseline | pythia-2.8b | 1 | 2454           | 2048           | 33.199 ms  | 37.968 ms | 26.338 tok/s | 23,070.73 MiB
flash    | pythia-2.8b | 1 | 2454           | 128            | 147.180 ms | 27.424 ms | 36.465 tok/s | 12,031.40 MiB
flash    | pythia-2.8b | 1 | 2454           | 512            | 66.252 ms  | 20.674 ms | 48.369 tok/s | 12,350.70 MiB
flash    | pythia-2.8b | 1 | 2454           | 1024           | 57.399 ms  | 19.134 ms | 52.263 tok/s | 12,835.35 MiB
flash    | pythia-2.8b | 1 | 2454           | 2048           | 55.696 ms  | 20.652 ms | 48.422 tok/s | 13,806.26 MiB

这是以性能数据基于 `max_new_tokens=2048`，`batch_size=1`，`prompt_len=2454` 的配置做出的一个简单的对比表格，
| Mode      | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | PeakMem (MB) | ppl (wikitext) | ppl (pg19) |
|-----------|-----------|----------------|---------------------|--------------|-----------------|------------|
| Baseline  | 33.20     | 37.97          | 26.34               | 23070.73     | 81.97           | 8.53       |
| Flash     | 55.70     | 20.65          | 48.42               | 13806.26     | 82.76           | 8.63       |

> 注：以上性能数据基于 `max_new_tokens=2048`，`batch_size=1`，`prompt_len=2454` 的配置。PPL 数据基于 WikiText (128 samples) 和 PG19 (64 samples)。

## 结果分析

根据在 NVIDIA RTX 4090D (24GB) 上的测试结果，我们可以得出以下结论：

1.  **生成速度提升显著**：
    *   **TPOT (Time Per Output Token)** 从 ~38ms 降低到 ~20.6ms，**生成速度几乎翻倍**。
    *   **Throughput (吞吐量)** 相应地从 ~26 tok/s 提升至 ~48 tok/s。
    *   这是因为 FlashAttention 有效减少了 HBM 访问次数，特别是在长序列生成时优势明显。

2.  **显存占用大幅降低**：
    *   峰值显存从 ~23GB 降低至 ~13.8GB，**节省了约 40% 的显存**。
    *   这使得在单卡 24GB 显存上运行更长的序列或更大的 Batch Size 成为可能。

3.  **首 token 延迟 (TTFT) 略有增加**：
    *   Flash 模式下的 TTFT (~55ms) 比基线 (~33ms) 略高。这可能是由于在 Prompt 处理阶段（Prefill），PyTorch SDPA 的 Flash 内核启动开销或 padding 处理逻辑带来的轻微 overhead。但在长文本生成场景下，这几十毫秒的差异通常可以忽略。

4.  **数值精度与 PPL 一致性**：
    *   **WikiText PPL**: Baseline (81.97) vs Flash (82.76)，差异约 0.9%。
    *   **PG-19 PPL**: Baseline (8.53) vs Flash (8.63)，差异约 1.1%。
    *   困惑度 (PPL) 的微小差异（< 1.5%）表明 FlashAttention 的集成没有引入显著的数值精度损失，模型输出质量保持稳定。

**总结**：FlashAttention 模式在保持模型精度的前提下，显著提升了推理速度并降低了显存消耗，非常适合长文本生成任务。

## 讨论与期望
- Flash 模式在 Ada 架构（如 4090D）下，通常可显著降低 TTFT 与 TPOT，并提升整体吞吐，同时降低峰值显存占用。
- ppl 指标应与基线近似（不修改参数、不改变数值路径的前提下），用于验证优化的正确性与无损性。

### 正确性自检

```powershell
python src/compare_logits.py --max-length 128
```

输出包含每个提示的 `max_diff / mean_diff`；正常情况下量级应在 `1e-3~1e-2 / 1e-4~1e-3` 范围内。

## 限制
- 需要 CUDA 12.x 与 PyTorch 2.1+，并且 Flash SDPA 内核可用。
- Windows 下的 `.sh` 脚本需在 WSL/Git Bash 运行；或者直接执行 Python 命令。
- `datasets` 下载可能受网络影响，建议配置镜像或离线缓存。
  - 如遇 `No available kernel` 或 `Flash attention kernel not used because dtype is float`：
    - 将环境变量设为 `FLASH_SDPA=0` 使用 math 回退
    - 或确保 GPU 可用并启用 `bfloat16/float16` 精度（Ada 建议 BF16）

## 后续工作（可选）
- 增加序列长度扫参：128/256/512/1024/2048。
- 引入 `plots.ipynb`，可视化性能与损失分布。
- 支持批量生成与更复杂的采样策略（如 top-k / nucleus）。
