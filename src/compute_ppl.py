import argparse
import json
import math
import os

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


def eval_dataset(
    model: torch.nn.Module,
    tokenizer,
    name: str,
    split: str,
    max_samples: int = 128,
    max_length: int = 512,
    config_name: str | None = None,
    use_autocast: bool = False,
) -> dict:
    """
    计算给定模型在某个 HF 数据集上的平均 loss 和 ppl。

    model: 已经 .to(device) 且 .eval() 的因果 LM
    tokenizer: 对应的 tokenizer
    name: 数据集名，如 "wikitext" 或 "pg19"
    split: 数据集切分，如 "test" / "validation"
    use_autocast: True 时在 GPU 上用 fp16 autocast（给 flash 路径用）
    """
    if name == "pg19":
        ds = _load_pg19(split)
    else:
        ds = load_dataset(name, config_name, split=split) if config_name else load_dataset(name, split=split)

    device = model.device
    losses: list[float] = []
    count = 0
    model.eval()

    if use_autocast and device.type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        autocast_ctx = torch.cuda.amp.autocast(enabled=False)

    for ex in ds:
        if count >= max_samples:
            break

        text = _safe_text(ex)
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = enc["input_ids"].to(device).long()
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 太短没信息，跳过
        if input_ids.shape[1] < 2:
            count += 1
            continue

        with torch.no_grad(), autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                return_dict=True,
            )
            loss = outputs.loss.detach().float().item()

        if not math.isnan(loss):
            losses.append(loss)

        count += 1

    if not losses:
        return {"ppl": None, "avg_loss": None, "num_samples": 0}

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    return {"ppl": ppl, "avg_loss": avg_loss, "num_samples": len(losses)}


def main():
    parser = argparse.ArgumentParser()
    # 保留 mode 参数，方便和 benchmark 那边风格统一
    parser.add_argument("--mode", choices=["baseline", "flash", "both"], default="both")
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    set_reproducibility(42)
    device = get_device()

    # ===== 1. baseline 模型：原生 Pythia-70M =====
    base_wrapper = PythiaBaselineModel()
    base_model = base_wrapper.to(device)
    base_model.device = device
    base_tokenizer = base_wrapper.tokenizer

    # ===== 2. flash 模型：同一个结构，但注意力内部切到 Flash SDPA =====
    # 这里确保 FLASH_SDPA 打开，再初始化 flash wrapper（NeoXFlashAttentionAdapter 会读这个环境变量）
    os.environ.setdefault("FLASH_SDPA", "1")
    flash_wrapper = PythiaFlashModel()
    flash_model = flash_wrapper.to(device)
    if device.type == "cuda":
        # 让整个模型用 fp16，这样 SDPA 才会真正走 Flash kernel
        flash_model = flash_model.half()
    flash_model.device = device
    flash_tokenizer = flash_wrapper.tokenizer

    # ===== 3. 评测 =====
    if args.dataset == "wikitext":
        base_res = eval_dataset(
            base_model,
            base_tokenizer,
            "wikitext",
            "test",
            max_samples=args.max_samples,
            max_length=args.max_length,
            config_name="wikitext-2-v1",
            use_autocast=False,   # baseline 走 float32 + math kernel
        )
        flash_res = eval_dataset(
            flash_model,
            flash_tokenizer,
            "wikitext",
            "test",
            max_samples=args.max_samples,
            max_length=args.max_length,
            config_name="wikitext-2-v1",
            use_autocast=True,    # flash 走 fp16 + Flash kernel
        )
    else:
        base_res = eval_dataset(
            base_model,
            base_tokenizer,
            "pg19",
            "validation",
            max_samples=args.max_samples,
            max_length=args.max_length,
            use_autocast=False,
        )
        flash_res = eval_dataset(
            flash_model,
            flash_tokenizer,
            "pg19",
            "validation",
            max_samples=args.max_samples,
            max_length=args.max_length,
            use_autocast=True,
        )

    out = {
        "baseline": base_res,
        "flash": flash_res,
    }

    # ===== 4. 保存结果 =====
    os.makedirs("results", exist_ok=True)
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

    # 给出 log(ppl) 差距，方便你在报告里引用
    if base_res["ppl"] is not None and flash_res["ppl"] is not None:
        log_diff = abs(math.log(flash_res["ppl"]) - math.log(base_res["ppl"]))
        out["log_ppl_diff"] = log_diff
        if log_diff > 0.5:
            print("Warning: PPL difference is large")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
