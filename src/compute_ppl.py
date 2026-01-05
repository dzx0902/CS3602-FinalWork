import argparse
import json
import math
import os
import time
from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset
import requests

from models.pythia_baseline_model import PythiaBaselineModel
from models.pythia_flash_model import PythiaFlashModel
from utils import get_device, set_reproducibility, resolve_pythia_model_name


# ----------------------------
# Helpers: download / io
# ----------------------------
def _safe_text(ex) -> str:
    for key in ("text", "passage", "content"):
        if key in ex and isinstance(ex[key], str):
            return ex[key]
    return str(ex)


def download_text_file(url: str, out_path: str, retries: int = 5, timeout: int = 30) -> str:
    """
    Download a text file from `url` to `out_path` with retries.
    Uses requests and respects environment proxies (HTTP_PROXY / HTTPS_PROXY).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Downloading pg19 sample from: {url}")
            print(f"[INFO] Saving to: {out_path} (attempt {attempt}/{retries})")

            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                # try to guess encoding; if not, just write bytes and decode later
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)

            # sanity check
            if os.path.getsize(out_path) < 64:
                raise RuntimeError("Downloaded file is too small; likely failed or HTML error page.")
            return out_path

        except Exception as e:
            last_err = e
            print(f"[WARN] Download failed: {repr(e)}")
            if attempt < retries:
                time.sleep(1.5 * attempt)
            continue

    raise RuntimeError(f"Failed to download after {retries} retries. Last error: {repr(last_err)}")


def read_text_file(path: str, max_chars: Optional[int] = None) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {path}")
    with open(path, "rb") as f:
        data = f.read()
    # robust decode
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except Exception:
            text = None
    if text is None:
        text = data.decode("utf-8", errors="ignore")
    if max_chars is not None and max_chars > 0:
        text = text[:max_chars]
    return text


# ----------------------------
# PPL: sliding window (long text friendly)
# ----------------------------
@torch.no_grad()
def ppl_sliding_window(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    window: int = 1024,
    stride: int = 512,
    max_total_tokens: Optional[int] = None,
) -> Dict:
    """
    Compute perplexity using a sliding window.
    We compute cross-entropy only on the "new" part each step to avoid double-counting.
    """
    model.eval()

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]  # (T,)
    if max_total_tokens is not None and max_total_tokens > 0:
        input_ids = input_ids[:max_total_tokens]

    seq_len = int(input_ids.numel())
    if seq_len < 2:
        return {"ppl": None, "avg_loss": None, "num_tokens": 0, "seq_len": seq_len}

    # Ensure window/stride sane
    window = max(16, int(window))
    stride = max(1, int(stride))
    if stride > window:
        stride = window

    total_nll = 0.0
    total_count = 0

    # Move to device in chunks to reduce peak memory spikes
    input_ids = input_ids.to(device)

    # We evaluate tokens in chunks: [begin:end), labels only for the last `trg_len` tokens
    for begin in range(0, seq_len - 1, stride):
        end = min(begin + window, seq_len)
        chunk = input_ids[begin:end].unsqueeze(0)  # (1, L)

        # labels: ignore all except the last (end - begin - prev_end) tokens
        # We want to count each token exactly once.
        # Let prev_end = begin if first step else begin + (window - stride)?? Classic approach:
        # Compute target length = end - begin if first, else min(stride, end - begin)
        # But in language modeling labels shift by one. We'll mask tokens except last trg_len.
        if begin == 0:
            trg_len = end - begin
        else:
            trg_len = min(stride, end - begin)

        labels = chunk.clone()
        # Ignore all tokens except last trg_len
        # Also ignore first token in the chunk because loss compares shifted logits vs labels
        # We'll be safe: mask everything except last trg_len, and also mask the first position.
        labels[:, :-trg_len] = -100

        # HF causal LM loss already does shift internally.
        out = model(input_ids=chunk, labels=labels, use_cache=False, return_dict=True)
        loss = float(out.loss.detach().float().item())

        # Convert chunk loss to total nll for counted tokens:
        # HF loss is averaged over non-ignored labels.
        # Count how many labels are not -100
        count = int((labels != -100).sum().item())
        if count > 0 and not math.isnan(loss):
            total_nll += loss * count
            total_count += count

        if end >= seq_len:
            break

    if total_count == 0:
        return {"ppl": None, "avg_loss": None, "num_tokens": 0, "seq_len": seq_len}

    avg_loss = total_nll / total_count
    ppl = float(math.exp(avg_loss))
    return {"ppl": ppl, "avg_loss": float(avg_loss), "num_tokens": int(total_count), "seq_len": seq_len}


@torch.no_grad()
def ppl_dataset_simple(
    model,
    tokenizer,
    dataset_name: str,
    split: str,
    device: torch.device,
    max_samples: int = 128,
    max_length: int = 512,
    config_name: Optional[str] = None,
) -> Dict:
    """
    Simple PPL on dataset samples by truncating each example to `max_length`.
    Good for wikitext quick compare.
    """
    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    losses = []
    n = 0

    model.eval()
    for ex in ds:
        if n >= max_samples:
            break
        text = _safe_text(ex)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            n += 1
            continue

        out = model(input_ids=input_ids, labels=input_ids, use_cache=False, return_dict=True)
        loss = float(out.loss.detach().float().item())
        if not math.isnan(loss):
            losses.append(loss)
        n += 1

    if len(losses) == 0:
        return {"ppl": None, "avg_loss": None, "num_samples": 0}

    avg_loss = sum(losses) / len(losses)
    ppl = float(math.exp(avg_loss))
    return {"ppl": ppl, "avg_loss": float(avg_loss), "num_samples": len(losses)}


def run_one_mode(
    mode: str,
    model_name: str,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict, str]:
    """
    mode: baseline / flash
    Returns (result_dict, mode)
    """
    if mode == "baseline":
        wrapper = PythiaBaselineModel(model_name=model_name)
    else:
        # flash
        # Make sure environment toggle is set before creating model (adapter reads env at init).
        os.environ["FLASH_SDPA"] = "1"
        wrapper = PythiaFlashModel(model_name=model_name)

    model = wrapper.to(device)
    tokenizer = wrapper.tokenizer

    if args.dataset == "wikitext":
        res = ppl_dataset_simple(
            model=model,
            tokenizer=tokenizer,
            dataset_name="wikitext",
            split="test",
            device=device,
            max_samples=args.max_samples,
            max_length=args.max_length,
            config_name="wikitext-2-v1",
        )
        return res, mode

    # pg19
    if args.pg19_mode == "sample1":
        # Ensure sample file exists; if not, download.
        if (not os.path.exists(args.pg19_sample_path)) or (os.path.getsize(args.pg19_sample_path) < 64) or args.force_download:
            download_text_file(args.pg19_url, args.pg19_sample_path, retries=args.download_retries, timeout=args.download_timeout)

        text = read_text_file(args.pg19_sample_path, max_chars=args.pg19_max_chars)
        res = ppl_sliding_window(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            window=args.window,
            stride=args.stride,
            max_total_tokens=args.pg19_max_tokens,
        )
        return res, mode

    # pg19 full dataset (not recommended for slow networks)
    ds = load_dataset("pg19", split="validation")
    # Use the first example as long text if needed.
    ex = ds[0]
    text = _safe_text(ex)
    res = ppl_sliding_window(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        window=args.window,
        stride=args.stride,
        max_total_tokens=args.pg19_max_tokens,
    )
    return res, mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")

    # mode: baseline / flash / both
    parser.add_argument("--mode", choices=["baseline", "flash", "both"], default="both")

    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument("--model-size", choices=["70m", "2.8b", "7b"], default="2.8b")

    # pg19 options
    parser.add_argument("--pg19-mode", choices=["sample1", "full"], default="sample1")
    parser.add_argument("--pg19-sample-path", type=str, default="src/prompts/pg19_sample1.txt")

    # Default: one pg19 file from DeepMind Gutenberg mirror used by datasets pg19 builder (smallish txt).
    # You can change to your own mirror if SSL/proxy issues occur.
    parser.add_argument(
        "--pg19-url",
        type=str,
        default="https://storage.googleapis.com/deepmind-gutenberg/train/23381.txt",
        help="URL to download a pg19 sample text (txt).",
    )

    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--download-timeout", type=int, default=30)

    # sliding window
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)

    # optional limits
    parser.add_argument("--pg19-max-chars", type=int, default=0, help="0 means no limit")
    parser.add_argument("--pg19-max-tokens", type=int, default=0, help="0 means no limit")

    args = parser.parse_args()

    set_reproducibility(42)
    device = get_device()
    model_name = resolve_pythia_model_name(args.model_size)

    # normalize limits
    if args.pg19_max_chars <= 0:
        args.pg19_max_chars = None
    if args.pg19_max_tokens <= 0:
        args.pg19_max_tokens = None

    out = {
        "model_name": model_name,
        "dataset": args.dataset,
        "mode": args.mode,
    }

    # Always compute both baseline/flash if mode == both (this matches your expectation).
    if args.mode == "baseline":
        base_res, _ = run_one_mode("baseline", model_name, device, args)
        out["baseline"] = base_res
    elif args.mode == "flash":
        flash_res, _ = run_one_mode("flash", model_name, device, args)
        out["flash"] = flash_res
    else:
        base_res, _ = run_one_mode("baseline", model_name, device, args)
        flash_res, _ = run_one_mode("flash", model_name, device, args)
        out["baseline"] = base_res
        out["flash"] = flash_res

        if base_res.get("ppl") is not None and flash_res.get("ppl") is not None:
            log_diff = abs(math.log(float(flash_res["ppl"])) - math.log(float(base_res["ppl"])))
            out["log_ppl_diff"] = float(log_diff)
            if log_diff > 0.5:
                print("[WARN] PPL difference is large")

    # Save
    os.makedirs("results", exist_ok=True)
    try:
        with open("results/ppl.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
