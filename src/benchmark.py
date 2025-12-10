import argparse
import json
import time
import torch
from transformers import AutoTokenizer
from models.pythia_baseline_model import PythiaBaselineModel
from models.pythia_flash_model import PythiaFlashModel
from utils import get_device, set_reproducibility, resolve_pythia_model_name  # 多了一个

def measure_ttft(model, input_ids, attention_mask):
    past = None
    start = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    ttft_ms = (time.perf_counter() - start) * 1000.0
    past = outputs.past_key_values
    return ttft_ms, next_token, past

def measure_tpot_and_throughput(model, next_token, past, attention_mask, max_new_tokens):
    total_time = 0.0
    for i in range(max_new_tokens - 1):
        start = time.perf_counter()
        outputs = model(
            input_ids=next_token,
            attention_mask=None,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        past = outputs.past_key_values
        total_time += time.perf_counter() - start

    if max_new_tokens <= 1:
        tpot_ms = None
        throughput = None
    else:
        tpot_ms = (total_time / (max_new_tokens - 1)) * 1000.0
        throughput = (max_new_tokens - 1) / total_time if total_time > 0 else None
    return tpot_ms, throughput

def measure_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return lambda: torch.cuda.max_memory_allocated() / (1024 ** 2)
    return lambda: None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "flash"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", type=str, default="src/prompts/short_prompt.txt")
    # 新增：选择 Pythia 规模，默认 2.8B
    parser.add_argument(
        "--model-size",
        choices=["70m", "2.8b", "7b"],
        default="2.8b",
        help="选择 Pythia 模型规模：70m / 2.8b / 7b(6.9b)",
    )
    args = parser.parse_args()

    set_reproducibility(42)
    device = get_device()

    # 解析出完整 HF 模型名
    model_name = resolve_pythia_model_name(args.model_size)

    # 选择 baseline 或 flash 包装
    if args.mode == "baseline":
        wrapper = PythiaBaselineModel(model_name=model_name)
    else:
        wrapper = PythiaFlashModel(model_name=model_name)

    model = wrapper.to(device)
    tokenizer = wrapper.tokenizer

    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    peak_fn = measure_peak_memory()
    ttft_ms, next_token, past = measure_ttft(model, input_ids, attention_mask)
    tpot_ms, throughput = measure_tpot_and_throughput(
        model, next_token, past, attention_mask, args.max_new_tokens
    )
    peak_mem_mb = peak_fn()

    out = {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "throughput_tps": throughput,
        "peak_mem_mb": peak_mem_mb,
        "config": {
            "mode": args.mode,
            "max_new_tokens": args.max_new_tokens,
            "device": str(device),
            "model_name": model_name,
        },
    }
    path = "results/flash_metrics.json" if args.mode == "flash" else "results/baseline_metrics.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
