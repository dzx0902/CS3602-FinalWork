import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from models.pythia_baseline_model import PythiaBaselineModel
from models.pythia_flash_model import PythiaFlashModel
from utils import get_device, set_reproducibility, resolve_pythia_model_name


# -----------------------------
# Helpers: formatting / printing
# -----------------------------
def _fmt_ms(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:,.3f} ms"


def _fmt_tps(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:,.3f} tok/s"


def _fmt_mb(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:,.2f} MiB"


def _hr(char: str = "-", width: int = 88) -> str:
    return char * width


def _print_kv(title: str, kv: Dict[str, Any], indent: int = 2) -> None:
    print(title)
    pad = max(len(k) for k in kv.keys()) if kv else 0
    for k, v in kv.items():
        print(" " * indent + f"{k:<{pad}} : {v}")


def _print_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]]) -> None:
    """
    columns: list of (key, header)
    """
    # Determine column widths
    widths = []
    for key, header in columns:
        max_len = len(header)
        for r in rows:
            max_len = max(max_len, len(str(r.get(key, ""))))
        widths.append(max_len)

    # Header
    header_line = " | ".join(header.ljust(w) for (_, header), w in zip(columns, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    print(header_line)
    print(sep_line)

    # Rows
    for r in rows:
        line = " | ".join(str(r.get(key, "")).ljust(w) for (key, _), w in zip(columns, widths))
        print(line)


# -----------------------------
# Benchmark core
# -----------------------------
def measure_ttft(model, input_ids, attention_mask):
    """
    TTFT: time to first token
    Supports batch_size >= 1
    """
    past = None
    start = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    logits = outputs.logits[:, -1, :]  # (B, vocab)
    next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
    ttft_ms = (time.perf_counter() - start) * 1000.0
    past = outputs.past_key_values
    return ttft_ms, next_token, past


def measure_tpot_and_throughput(model, next_token, past, max_new_tokens: int):
    """
    TPOT: time per output token (from 2nd token)
    throughput: tokens / second
    """
    total_time = 0.0
    if max_new_tokens <= 1:
        return None, None

    # Generate max_new_tokens - 1 tokens
    for _ in range(max_new_tokens - 1):
        start = time.perf_counter()
        outputs = model(
            input_ids=next_token,
            attention_mask=None,          # decoding phase typically doesn't need mask
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        past = outputs.past_key_values
        total_time += time.perf_counter() - start

    tpot_ms = (total_time / (max_new_tokens - 1)) * 1000.0
    throughput = (max_new_tokens - 1) / total_time if total_time > 0 else None
    return tpot_ms, throughput


def _maybe_reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _get_cuda_peak_mb() -> Optional[float]:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "flash"], required=True)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Default max_new_tokens for a single run. Overridden by --new-token-sizes if provided.",
    )
    parser.add_argument(
        "--new-token-sizes",
        type=int,
        nargs="+",
        help="Test multiple max_new_tokens in one run. Example: --new-token-sizes 64 128 256",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="src/prompts/short_prompt.txt",
        help="Prompt file path (short prompt by default).",
    )
    parser.add_argument(
        "--long-prompt",
        action="store_true",
        help="If set, use --long-prompt-path as prompt file.",
    )
    parser.add_argument(
        "--long-prompt-path",
        type=str,
        default="src/prompts/long_prompt.txt",
        help="Long prompt file path used with --long-prompt.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Test multiple batch sizes in one run. Example: --batch-sizes 1 2 4",
    )
    parser.add_argument(
        "--model-size",
        choices=["70m", "2.8b", "7b"],
        default="2.8b",
        help="Pythia model scale: 70m / 2.8b / 7b(6.9b).",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Optional truncation: limit prompt length (in tokens). If not set, no truncation.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Warmup forward passes (per config) to reduce cold-start noise.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print human-readable report in terminal (default: on).",
    )
    args = parser.parse_args()

    # default to pretty output if not explicitly disabled
    if not args.pretty:
        args.pretty = True

    set_reproducibility(42)
    device = get_device()

    model_name = resolve_pythia_model_name(args.model_size)

    # Load wrapper
    if args.mode == "baseline":
        wrapper = PythiaBaselineModel(model_name=model_name)
    else:
        wrapper = PythiaFlashModel(model_name=model_name)

    model = wrapper.to(device)
    tokenizer = wrapper.tokenizer

    prompt_path = args.long_prompt_path if args.long_prompt else args.prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    # Tokenize prompt
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=(args.max_prompt_tokens is not None),
                    max_length=args.max_prompt_tokens)
    base_input_ids = enc["input_ids"].to(device)
    base_attention_mask = enc.get("attention_mask", None)
    if base_attention_mask is not None:
        base_attention_mask = base_attention_mask.to(device)

    seq_len = int(base_input_ids.shape[1])

    # Determine new_token_sizes
    if args.new_token_sizes and len(args.new_token_sizes) > 0:
        new_token_sizes = list(args.new_token_sizes)
    else:
        new_token_sizes = [int(args.max_new_tokens)]

    results: List[Dict[str, Any]] = []

    # Header / config summary
    if args.pretty:
        print(_hr("="))
        print("BENCHMARK REPORT")
        print(_hr("="))
        _print_kv(
            "Config:",
            {
                "mode": args.mode,
                "model_name": model_name,
                "device": str(device),
                "FLASH_SDPA": os.environ.get("FLASH_SDPA", "(unset)"),
                "prompt_path": prompt_path,
                "prompt_seq_len": seq_len,
                "max_prompt_tokens": args.max_prompt_tokens if args.max_prompt_tokens is not None else "(none)",
                "batch_sizes": args.batch_sizes,
                "new_token_sizes": new_token_sizes,
                "warmup_iters": args.warmup_iters,
            },
        )
        print(_hr("-"))

    # Run benchmark grid
    for batch_size in args.batch_sizes:
        # Build batch (replicate prompt)
        if batch_size == 1:
            input_ids = base_input_ids
            attention_mask = base_attention_mask
        else:
            input_ids = base_input_ids.expand(batch_size, -1)
            attention_mask = base_attention_mask.expand(batch_size, -1) if base_attention_mask is not None else None

        for max_new in new_token_sizes:
            # Warmup (same shapes)
            with torch.no_grad():
                for _ in range(max(args.warmup_iters, 0)):
                    _maybe_reset_cuda_peak()
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            # Actual measurement
            _maybe_reset_cuda_peak()
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft_ms, next_token, past = measure_ttft(model, input_ids, attention_mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                tpot_ms, throughput = measure_tpot_and_throughput(model, next_token, past, max_new)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            peak_mem_mb = _get_cuda_peak_mb()

            results.append(
                {
                    "batch_size": batch_size,
                    "max_new_tokens": max_new,
                    "prompt_seq_len": seq_len,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "throughput_tps": throughput,
                    "peak_mem_mb": peak_mem_mb,
                    "config": {
                        "mode": args.mode,
                        "device": str(device),
                        "model_name": model_name,
                        "prompt_path": prompt_path,
                        "max_prompt_tokens": args.max_prompt_tokens,
                        "FLASH_SDPA": os.environ.get("FLASH_SDPA"),
                    },
                }
            )

            if args.pretty:
                print(f"[DONE] batch={batch_size}  max_new={max_new}  seq_len={seq_len}  "
                      f"TTFT={_fmt_ms(ttft_ms)}  TPOT={_fmt_ms(tpot_ms)}  "
                      f"Throughput={_fmt_tps(throughput)}  PeakMem={_fmt_mb(peak_mem_mb)}")

    # Final JSON output (for files & machine parsing)
    out = {
        "results": results,
        "summary": {
            "mode": args.mode,
            "model_name": model_name,
            "device": str(device),
            "prompt_path": prompt_path,
            "prompt_seq_len": seq_len,
            "max_prompt_tokens": args.max_prompt_tokens,
            "batch_sizes": args.batch_sizes,
            "new_token_sizes": new_token_sizes,
            "warmup_iters": args.warmup_iters,
            "FLASH_SDPA": os.environ.get("FLASH_SDPA"),
        },
    }

    # Write to results
    os.makedirs("results", exist_ok=True)
    path = "results/flash_metrics.json" if args.mode == "flash" else "results/baseline_metrics.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Pretty summary table
    if args.pretty:
        print(_hr("-"))
        print("Summary table:")
        table_rows = []
        for r in results:
            table_rows.append(
                {
                    "mode": args.mode,
                    "model": model_name.split("/")[-1],
                    "B": r["batch_size"],
                    "seq": r["prompt_seq_len"],
                    "new": r["max_new_tokens"],
                    "TTFT": _fmt_ms(r["ttft_ms"]),
                    "TPOT": _fmt_ms(r["tpot_ms"]),
                    "Tok/s": _fmt_tps(r["throughput_tps"]),
                    "PeakMem": _fmt_mb(r["peak_mem_mb"]),
                }
            )
        _print_table(
            table_rows,
            columns=[
                ("mode", "mode"),
                ("model", "model"),
                ("B", "B"),
                ("seq", "prompt_seq_len"),
                ("new", "max_new_tokens"),
                ("TTFT", "TTFT"),
                ("TPOT", "TPOT"),
                ("Tok/s", "Throughput"),
                ("PeakMem", "PeakMem"),
            ],
        )
        print(_hr("="))
        print(f"Saved JSON -> {path}")
        print(_hr("="))

    # Also print machine-readable JSON at the end (so you can redirect)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
