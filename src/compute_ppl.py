import argparse
import json
import math
import torch
from datasets import load_dataset
from models.pythia_baseline_model import PythiaBaselineModel
from models.pythia_flash_model import PythiaFlashModel
from utils import get_device, set_reproducibility

def _safe_text(ex):
    for key in ("text", "passage", "content"):
        if key in ex and isinstance(ex[key], str):
            return ex[key]
    return str(ex)

def _load_pg19(split: str):
    last_err = None
    for cfg in (None, "plain_text", "pg19"):
        try:
            if cfg is None:
                return load_dataset("pg19", split=split)
            else:
                return load_dataset("pg19", cfg, split=split)
        except Exception as e:
            last_err = e
            continue
    raise last_err

def eval_dataset(model, tokenizer, name: str, split: str, max_samples: int = 128, max_length: int = 512, config_name: str = None):
    if name == "pg19":
        ds = _load_pg19(split)
    else:
        ds = load_dataset(name, config_name, split=split) if config_name else load_dataset(name, split=split)
    losses = []
    count = 0
    model.eval()
    for ex in ds:
        if count >= max_samples:
            break
        text = _safe_text(ex)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(model.device).long()
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        if input_ids.shape[1] < 2:
            count += 1
            continue
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=True)
            loss = outputs.loss.detach().float().item()
        if not (loss != loss):
            losses.append(loss)
        count += 1
    if len(losses) == 0:
        return {"ppl": None, "avg_loss": None, "num_samples": 0}
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    return {"ppl": ppl, "avg_loss": avg_loss, "num_samples": len(losses)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "flash"], required=True)
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    set_reproducibility(42)
    device = get_device()
    base_wrapper = PythiaBaselineModel()
    flash_wrapper = PythiaFlashModel()
    base_model = base_wrapper.to(device)
    flash_model = flash_wrapper.to(device)
    base_tokenizer = base_wrapper.tokenizer
    flash_tokenizer = flash_wrapper.tokenizer
    base_model.device = device
    flash_model.device = device
    if args.dataset == "wikitext":
        base_res = eval_dataset(base_model, base_tokenizer, "wikitext", "test", max_samples=args.max_samples, max_length=args.max_length, config_name="wikitext-2-v1")
        flash_res = eval_dataset(flash_model, flash_tokenizer, "wikitext", "test", max_samples=args.max_samples, max_length=args.max_length, config_name="wikitext-2-v1")
    else:
        base_res = eval_dataset(base_model, base_tokenizer, "pg19", "validation", max_samples=args.max_samples, max_length=args.max_length)
        flash_res = eval_dataset(flash_model, flash_tokenizer, "pg19", "validation", max_samples=args.max_samples, max_length=args.max_length)
    out = {
        "baseline": base_res,
        "flash": flash_res,
    }
    try:
        with open("results/ppl_baseline.json", "w", encoding="utf-8") as f:
            json.dump(base_res, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    try:
        with open("results/ppl_flash.json", "w", encoding="utf-8") as f:
            json.dump(flash_res, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    if base_res["ppl"] is not None and flash_res["ppl"] is not None:
        log_diff = abs(math.log(flash_res["ppl"]) - math.log(base_res["ppl"]))
        out["log_ppl_diff"] = log_diff
        if log_diff > 0.5:
            print("Warning: PPL difference is large")
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
