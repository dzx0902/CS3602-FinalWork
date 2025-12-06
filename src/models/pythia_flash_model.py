import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from .flash_attention import FlashSelfAttention

class NeoXFlashAttentionAdapter(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, rotary_base: float = 10000.0, rotary_pct: float = 1.0, rotary_emb: nn.Module = None):
        super().__init__()
        self.flash = FlashSelfAttention(hidden_size, num_heads, rotary_base=rotary_base, rotary_pct=rotary_pct, rotary_emb=rotary_emb)

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
        out, present = self.flash(hidden_states, past_key_value=layer_past, use_cache=use_cache, position_ids=position_ids, attention_mask=attention_mask)
        if output_attentions:
            return out, present, None
        return out, present

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
                rotary_base = getattr(self.config, "rotary_embedding_base", 10000.0)
                rotary_pct = getattr(self.config, "rotary_pct", 1.0)
                rotary_emb = getattr(module, "rotary_emb", None)
                adapter = NeoXFlashAttentionAdapter(hidden_size, num_heads, rotary_base=rotary_base, rotary_pct=rotary_pct, rotary_emb=rotary_emb)
                adapter.flash.qkv_proj.weight.data.copy_(module.query_key_value.weight.data)
                if module.query_key_value.bias is not None and adapter.flash.qkv_proj.bias is not None:
                    adapter.flash.qkv_proj.bias.data.copy_(module.query_key_value.bias.data)
                adapter.flash.out_proj.weight.data.copy_(module.dense.weight.data)
                if module.dense.bias is not None and adapter.flash.out_proj.bias is not None:
                    adapter.flash.out_proj.bias.data.copy_(module.dense.bias.data)
                parent = self._get_parent_module(name)
                attr = name.split(".")[-1]
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
