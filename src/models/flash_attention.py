import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from torch.backends.cuda import sdp_kernel
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

class FlashSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, rotary_base: float = 10000.0, rotary_pct: float = 1.0, rotary_emb: nn.Module = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.rotary_base = rotary_base
        self.rotary_pct = rotary_pct
        self.rotary_emb = rotary_emb
        self.use_flash = os.environ.get("FLASH_SDPA", "0") == "1"

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        dim = self.head_dim
        dtype = q.dtype
        if self.rotary_emb is None:
            return q, k
        total_seq_len = position_ids.max().item() + 1
        try:
            cos, sin = self.rotary_emb(k, position_ids=position_ids)
        except TypeError:
            try:
                cos, sin = self.rotary_emb(k, seq_len=total_seq_len)
            except TypeError:
                cos, sin = self.rotary_emb(k)
        if cos.size(-1) != q.size(-1):
            tail = q.size(-1) - cos.size(-1)
            cos_tail = torch.ones((*cos.shape[:-1], tail), device=cos.device, dtype=cos.dtype)
            sin_tail = torch.zeros((*sin.shape[:-1], tail), device=sin.device, dtype=sin.dtype)
            cos = torch.cat([cos, cos_tail], dim=-1)
            sin = torch.cat([sin, sin_tail], dim=-1)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        return q_rot.to(dtype), k_rot.to(dtype)

    def forward(self, hidden_states: torch.Tensor, past_key_value=None, use_cache: bool = False, position_ids: torch.Tensor = None, attention_mask: torch.Tensor = None):
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(self.hidden_size, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_key_value is not None:
            pk, pv = past_key_value
            past_len = pk.size(2)

        if position_ids is None:
            start = past_len
            position_ids = torch.arange(start, start + seq_len, device=hidden_states.device).unsqueeze(0).expand(bsz, seq_len)

        q, k_new = self._apply_rope(q, k, position_ids)

        if past_key_value is not None:
            k = torch.cat([pk, k_new], dim=2)
            v = torch.cat([pv, v], dim=2)
        else:
            k = k_new

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if past_len > 0:
                    kv_mask = torch.ones((bsz, past_len), device=attention_mask.device, dtype=attention_mask.dtype)
                    kv_mask = torch.cat([kv_mask, attention_mask], dim=1)
                else:
                    kv_mask = attention_mask
                attn_mask = (kv_mask == 0).view(bsz, 1, 1, -1).expand(bsz, 1, seq_len, kv_mask.size(1))
        if self.use_flash and q.is_cuda:
            orig_dtype = q.dtype
            qf = q
            kf = k
            vf = v
            if orig_dtype == torch.float32:
                if torch.cuda.is_available():
                    qf = q.to(torch.bfloat16)
                    kf = k.to(torch.bfloat16)
                    vf = v.to(torch.bfloat16)
            with use_flash_sdpa():
                attn = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=attn_mask, is_causal=True).to(orig_dtype)
        else:
            with use_math_sdpa():
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, -1, self.hidden_size)
        out = self.out_proj(attn)
        present = None
        if use_cache:
            present = (k, v)
        return out, present

@contextmanager
def use_math_sdpa():
    with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        yield

@contextmanager
def use_flash_sdpa():
    with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        yield
