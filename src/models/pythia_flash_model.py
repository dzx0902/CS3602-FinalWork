# models/pythia_flash_model.py
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .flash_attention import use_math_sdpa, use_flash_sdpa
from utils import check_sdpa_flash_available


class NeoXFlashAttentionAdapter(nn.Module):
    """
    适配器：不改 GPT-NeoXAttention 的数学逻辑，
    只通过 sdp_kernel 控制 scaled_dot_product_attention 使用 math 还是 flash kernel。
    """

    def __init__(self, original_module: nn.Module):
        super().__init__()
        self.original = original_module
        # 环境变量决定是否“尝试”使用 flash
        self.use_flash = os.environ.get("FLASH_SDPA", "0") == "1"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
        position_ids=None,
        **kwargs,
    ):
        # 根据环境变量选择 kernel
        if self.use_flash:
            ctx = use_flash_sdpa()
        else:
            ctx = use_math_sdpa()

        with ctx:
            return self.original(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_ids=position_ids,
                **kwargs,
            )


class PythiaFlashModel(nn.Module):
    """
    对外暴露的 Flash 包装：
    - .model: HF 的 AutoModelForCausalLM
    - .tokenizer: 对应 tokenizer
    - attention 层都被 NeoXFlashAttentionAdapter 包裹
    - 在支持的 GPU 上，模型以 bfloat16 运行以匹配 flash kernel 要求
    """

    def __init__(self, model_name: str = "EleutherAI/pythia-70m"):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)

        # 默认用 float32，只有在 GPU 确认支持 flash + bf16 时才用 bf16
        torch_dtype = torch.float32
        info = check_sdpa_flash_available(dtype=torch.bfloat16)
        if info.get("flash_enabled") and info.get("smoke_test", False):
            torch_dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._replace_attention_modules()

    def _replace_attention_modules(self):
        """
        遍历模型，查找 GPTNeoXAttention 模块，并用适配器包裹。
        注意：我们不改动 query_key_value / dense 等参数，只是包一层 forward。
        """
        num_heads = getattr(self.config, "num_attention_heads", None)
        hidden_size = getattr(self.config, "hidden_size", None)

        for name, module in self.model.named_modules():
            has_qkv = hasattr(module, "query_key_value") and isinstance(module.query_key_value, nn.Linear)
            has_out = hasattr(module, "dense") and isinstance(module.dense, nn.Linear)
            if has_qkv and has_out and num_heads and hidden_size:
                parent = self._get_parent_module(name)
                attr = name.split(".")[-1]
                adapter = NeoXFlashAttentionAdapter(module)
                setattr(parent, attr, adapter)

    def _get_parent_module(self, module_name: str):
        parts = module_name.split(".")
        parent = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
