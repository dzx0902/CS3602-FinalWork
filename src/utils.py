import os
import random
import torch
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def check_cuda():
    info = {}
    info["available"] = torch.cuda.is_available()
    if info["available"]:
        idx = torch.cuda.current_device()
        info["device_index"] = idx
        info["name"] = torch.cuda.get_device_name(idx)
        info["capability"] = torch.cuda.get_device_capability(idx)
        info["memory_total"] = torch.cuda.get_device_properties(idx).total_memory
    return info

def check_sdpa_flash_available(dtype=torch.bfloat16):
    if not torch.cuda.is_available():
        return {"flash_enabled": False, "reason": "cuda_unavailable"}
    enabled = False
    try:
        enabled = torch.backends.cuda.flash_sdp_enabled()
    except Exception:
        enabled = False
    result = {"flash_enabled": bool(enabled)}
    if enabled:
        try:
            q = torch.randn(1, 4, 8, 64, device="cuda", dtype=dtype)
            k = torch.randn(1, 4, 8, 64, device="cuda", dtype=dtype)
            v = torch.randn(1, 4, 8, 64, device="cuda", dtype=dtype)
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
            result["smoke_test"] = True
        except Exception as e:
            result["smoke_test"] = False
            result["error"] = str(e)
    return result

def set_reproducibility(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
