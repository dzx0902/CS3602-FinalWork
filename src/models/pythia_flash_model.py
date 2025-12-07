import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from .flash_attention import use_math_sdpa, use_flash_sdpa

class NeoXFlashAttentionAdapter(nn.Module):
    def __init__(self, original_module: nn.Module):
        super().__init__()
        self.original = original_module
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
    def __init__(self, model_name: str = "EleutherAI/pythia-70m"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._replace_attention_modules()

    def _replace_attention_modules(self):
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
