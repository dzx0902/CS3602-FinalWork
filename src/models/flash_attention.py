import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        dim = self.head_dim
        device = q.device
        dtype = q.dtype
        if self.rotary_emb is None:
            return q, k
        q_rot, k_rot = self.rotary_emb.apply_rotary_pos_emb(q, k, position_ids)
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
                attn_mask = (kv_mask == 0).view(bsz, 1, 1, -1).expand(bsz, 1, q.size(2), kv_mask.size(1))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, -1, self.hidden_size)
        out = self.out_proj(attn)
        present = None
        if use_cache:
            present = (k, v)
        return out, present
