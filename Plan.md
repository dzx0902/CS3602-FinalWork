# plan.md — Detailed Implementation Plan for Environment-Level Optimization (FlashAttention) on Pythia-70M

## 0. Overview

This plan describes **all tasks the coding agent must perform** to build a complete, reproducible, well-structured project implementing **environment-level inference optimization** for the **Pythia-70M** model using **FlashAttention via PyTorch SDPA (scaled_dot_product_attention)**.

This implementation:
- MUST NOT modify any model parameters.
- MUST provide two inference modes:
  1. Baseline mode (math kernel attention)
  2. FlashAttention mode (flash kernel attention)
- MUST include benchmarking, ppl evaluation, scripts, results, and README.

The plan provides **explicit file structures**, **module responsibilities**, **API signatures**, and **step-by-step tasks** to enable automated project generation.

---

# 1. Project Requirements

## 1.1 Functional Requirements
The project must:

- Load and run inference with `EleutherAI/pythia-70m`.
- Implement a FlashAttention-based attention layer using SDPA.
- Replace original attention with FlashAttention while preserving weight parameters.
- Execute generation with KV-cache support.
- Provide quantitative comparisons:
  - TTFT: Time to First Token
  - TPOT: Time per Output Token
  - Throughput: tokens per second
  - GPU peak memory
  - ppl (wikitext, pg-19)
- Store results in JSON.
- Provide shell scripts to reproduce all experiments.
- Provide complete README.

## 1.2 Non-functional Requirements
- Code must be deterministic & reproducible.
- Must run on CUDA GPU with PyTorch 2.x.
- Must not require model finetuning.
- Must maintain compatibility with HF generate() API.
- Must separate baseline vs optimized modes cleanly.

---

# 2. Directory Structure (Agent Must Create)

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
│   └── plots.ipynb
│
├── README.md
├── plan.md
└── requirements.txt
```

---

# 3. Implementation Tasks (Highly Detailed)

Below are tasks the coding agent must complete in order.

---

# Task 1 — Environment Setup

## 1.1 Create requirements.txt
Include:

```
torch>=2.1
transformers>=4.36
accelerate
datasets
tqdm
numpy
```

## 1.2 Ensure GPU / SDPA availability
Add diagnostic utility to `src/utils.py`:

- check_cuda()
- log GPU properties
- warn if Flash kernel unavailable

---

# Task 2 — Implement FlashAttention Module

File: `src/models/flash_attention.py`

## 2.1 Class: FlashSelfAttention
Signature:

```python
class FlashSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        ...
```

### Responsibilities
- Project hidden states to Q/K/V using a single Linear layer (same as Pythia).
- Reshape Q/K/V → (batch, heads, seq_len, head_dim)
- Append to past_key_value if provided
- Use PyTorch SDPA for attention:

```python
F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    is_causal=True
)
```

- Return:
  - attn_output
  - updated past_key_value

### Required features
- EXACT shape matching Pythia's attention output.
- Support for decoding with KV-cache.
- Support for dynamic seq_len.
- Must not alter parameters.

---

# Task 3 — Build Flash-Optimized Pythia Model Wrapper

File: `src/models/pythia_flash_model.py`

Class: `PythiaFlashModel`

## 3.1 Behavior Requirements
- Load Pythia-70M via AutoModelForCausalLM.
- Replace each transformer block’s attention module with FlashSelfAttention:
  - Copy all weight matrices (qkv_proj and output_proj).
  - Keep feed-forward, layer norms, etc. unchanged.
- Ensure forward() signature matches HF causal LM:
  - input_ids
  - attention_mask
  - past_key_values
  - use_cache
  - return_dict

## 3.2 Ensure generate() works
- The model must be fully compatible with HF's `generate()`.
- Must support greedy decoding and KV caching.

---

# Task 4 — Baseline Pythia Wrapper

File: `src/models/pythia_baseline_model.py`

Class: `PythiaBaselineModel`

## Requirements
- Load vanilla HF Pythia-70M.
- No architectural changes.
- Must provide identical API to PythiaFlashModel.

---

# Task 5 — Benchmark Pipeline

File: `src/benchmark.py`

## 5.1 Benchmark Metrics
Agent must implement functions:

```
measure_ttft()
measure_tpot()
measure_throughput()
measure_peak_memory()
```

## 5.2 Step-by-Step Greedy Decoding
TTFT must be measured as:

1. Encode prompt
2. Perform first decoding step
3. Capture the elapsed time until first new token

TPOT measured across all additional tokens.

## 5.3 CLI
Allow:

```
python src/benchmark.py --mode baseline --max-new-tokens 128
python src/benchmark.py --mode flash --max-new-tokens 128
```

## 5.4 Output JSON Format

```
{
  "ttft_ms": ...,
  "tpot_ms": ...,
  "throughput_tps": ...,
  "peak_mem_mb": ...,
  "config": {...}
}
```

---

# Task 6 — PPL Evaluation

File: `src/compute_ppl.py`

## 6.1 Compute loss using teacher forcing

```
outputs = model(input_ids, labels=input_ids)
loss = outputs.loss
ppl = exp(loss)
```

## 6.2 Evaluate on:
- wikitext (HF datasets: wikiText-2 or wikiText-103)
- pg-19 (one long sample; can load from `datasets`)

## 6.3 CLI

```
python src/compute_ppl.py --mode baseline
python src/compute_ppl.py --mode flash
```

## 6.4 Output JSON:

```
{
  "ppl": ...,
  "avg_loss": ...,
  "num_samples": ...
}
```

---

# Task 7 — Scripts

Directory: `scripts/`

Create:

## 7.1 run_baseline.sh
Runs:
- benchmark baseline
- compute ppl baseline
- save outputs to `results/`

## 7.2 run_flash.sh
Same but flash mode.

## 7.3 run_all.sh
Runs both + prints comparison summary.

---

# Task 8 — README.md

Agent must generate README including:

## 8.1 Sections
- Overview
- Environment Setup
- Running Baseline vs Flash Models
- Benchmark Results Table
- PPL Results Table
- Example Commands
- Discussion of Results
- Limitations
- Future Work

## 8.2 Auto-generated template example

Table (agent fills numbers):

```
| Mode      | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | PeakMem (MB) | ppl (wikitext) | ppl (pg19) |
|-----------|-----------|----------------|---------------------|--------------|-----------------|------------|
| Baseline  | ...       | ...            | ...                 | ...          | ...             | ...        |
| Flash     | ...       | ...            | ...                 | ...          | ...             | ...        |
```

---

# 9. Stretch Goals (Optional for Agent)

- Implement FlashDecoding and compare.
- Add sequence length sweep:
  - 128 / 256 / 512 / 1024 / 2048 tokens
- Add plots.ipynb for visualization.
- Add batch inference support.

---

# 10. Success Criteria (Must Be Met)

A correct implementation must:

1. Run Pythia-70M baseline generation successfully.
2. Run FlashAttention generation successfully.
3. Produce measurable acceleration in:
   - TTFT
   - TPOT
   - Throughput
   - Memory footprint
4. Maintain similar ppl values.
5. Provide reproducible results via scripts.
6. Be fully understandable through README.
7. Be structured exactly as described in this plan.

