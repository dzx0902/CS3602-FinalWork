import torch
import json
import argparse
from models.pythia_baseline_model import PythiaBaselineModel
from models.pythia_flash_model import PythiaFlashModel
from utils import get_device, set_reproducibility

def load_long_prompt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-prompt", type=str, default="src/prompts/long_prompt.txt")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    set_reproducibility(42)
    device = get_device()
    base = PythiaBaselineModel().to(device)
    flash = PythiaFlashModel().to(device)
    base.device = device
    flash.device = device
    tok = base.tokenizer
    prompts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Deep learning models require careful attention masking.",
    ]
    lp = load_long_prompt(args.long_prompt)
    if lp:
        prompts.append(lp)
    results = []
    base.eval()
    flash.eval()
    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            ob = base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            of = flash(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        lb = ob.logits
        lf = of.logits
        n = min(lb.size(1), lf.size(1))
        lb = lb[:, :n, :]
        lf = lf[:, :n, :]
        diff = (lb - lf).abs().float()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        results.append({"prompt": p[:80], "max_diff": max_diff, "mean_diff": mean_diff})
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
