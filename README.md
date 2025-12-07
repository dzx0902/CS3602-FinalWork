# Pythia-70M 推理优化：FlashAttention (SDPA)

本项目在不更改模型参数的前提下，为 `EleutherAI/pythia-70m` 提供两种推理模式：
- 基线模式（math kernel attention）
- FlashAttention 模式（通过 PyTorch SDPA 触发 flash 内核）

同时提供基准测试（TTFT/TPOT/吞吐/峰值显存）、ppl 评估（WikiText/PG-19）、复现实验脚本与结果存储结构。

## 环境准备
- 推荐环境：CUDA 12.1 + `torch==2.1.2+cu121`，GPU：Ada (如 4090D)
- 依赖安装：

```bash
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

```bash
# 基线（math SDPA）
python src/benchmark.py --mode baseline --max-new-tokens 128

# Flash（需开启环境变量以选择 flash 内核）
# Windows PowerShell:
$env:FLASH_SDPA='1'; python src/benchmark.py --mode flash --max-new-tokens 128
# Linux/WSL/Git Bash:
FLASH_SDPA=1 python src/benchmark.py --mode flash --max-new-tokens 128
```

- 运行 PPL（WikiText/PG-19）：

```bash
# 单次同时输出两种模式的结果（baseline/flash）
python src/compute_ppl.py --dataset wikitext --max-samples 128 --max-length 512
python src/compute_ppl.py --dataset pg19 --max-samples 64 --max-length 512

# 如需显式分别运行，也可保留 --mode 参数（脚本仍会同时对比输出）
python src/compute_ppl.py --mode baseline --dataset wikitext
python src/compute_ppl.py --mode flash --dataset wikitext
```

- 脚本（Linux/WSL/Git Bash 环境）：

```bash
sh scripts/run_baseline.sh
sh scripts/run_flash.sh
sh scripts/run_all.sh
```

Windows 原生环境可直接使用上面的 `python` 命令，无需 `.sh`。

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

## 结果模板

| Mode      | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | PeakMem (MB) | ppl (wikitext) | ppl (pg19) |
|-----------|-----------|----------------|---------------------|--------------|-----------------|------------|
| Baseline  | ...       | ...            | ...                 | ...          | ...             | ...        |
| Flash     | ...       | ...            | ...                 | ...          | ...             | ...        |

> 运行完成后，相关数值会以 JSON 形式保存到 `results/` 目录，可据此填充上表。

## 讨论与期望
- Flash 模式在 Ada 架构（如 4090D）下，通常可显著降低 TTFT 与 TPOT，并提升整体吞吐，同时降低峰值显存占用。
- ppl 指标应与基线近似（不修改参数、不改变数值路径的前提下），用于验证优化的正确性与无损性。

### 正确性自检

```bash
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
