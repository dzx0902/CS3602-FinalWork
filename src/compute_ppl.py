import argparse
import json
import math
import torch
from datasets import load_dataset
from models.pythia_baseline_model import PythiaBaselineModel
from models.pythia_flash_model import PythiaFlashModel
from utils import get_device, set_reproducibility

def eval_dataset(model, tokenizer, name: str, split: str, max_samples: int = 128, max_length: int = 512, config_name: str = None):
    ds = load_dataset(name, config_name, split=split) if config_name else load_dataset(name, split=split)
    losses = []
    count = 0
    for ex in ds:
        if count >= max_samples:
            break
        text = ex["text"] if "text" in ex else str(ex)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(model.device)
        outputs = model(input_ids=input_ids, labels=input_ids, return_dict=True)
        loss = outputs.loss.detach().float().item()
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
    args = parser.parse_args()
    set_reproducibility(42)
    device = get_device()
    if args.mode == "baseline":
        wrapper = PythiaBaselineModel()
    else:
        wrapper = PythiaFlashModel()
    model = wrapper.to(device)
    tokenizer = wrapper.tokenizer
    model.device = device
    if args.dataset == "wikitext":
        res = eval_dataset(model, tokenizer, "wikitext", "test", config_name="wikitext-2-v1")
    else:
        res = eval_dataset(model, tokenizer, "pg19", "validation")
    path = "results/ppl_flash.json" if args.mode == "flash" else "results/ppl_baseline.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    main()
