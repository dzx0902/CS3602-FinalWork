import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, rotary_base: float = 10000.0, rotary_pct: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.rotary_base = rotary_base
        self.rotary_pct = rotary_pct

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, seq_positions: torch.Tensor):
        dim = self.head_dim
        device = q.device
        dtype = q.dtype
        rotary_dim = int(dim * self.rotary_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        if rotary_dim <= 0:
            return q, k
        inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / dim))
        t = seq_positions.to(torch.float32).unsqueeze(-1) * inv_freq  # (bsz, seq_len, rotary_dim/2)
        cos = torch.cos(t).to(dtype).unsqueeze(1)  # (bsz, 1, seq_len, rotary_dim/2)
        sin = torch.sin(t).to(dtype).unsqueeze(1)  # (bsz, 1, seq_len, rotary_dim/2)

        q_head = q[..., :rotary_dim]
        k_head = k[..., :rotary_dim]
        q_tail = q[..., rotary_dim:]
        k_tail = k[..., rotary_dim:]
        q_even = q_head[..., ::2]
        q_odd = q_head[..., 1::2]
        k_even = k_head[..., ::2]
        k_odd = k_head[..., 1::2]
        q_rot_even = q_even * cos - q_odd * sin
        q_rot_odd = q_odd * cos + q_even * sin
        k_rot_even = k_even * cos - k_odd * sin
        k_rot_odd = k_odd * cos + k_even * sin
        q_rot = torch.empty_like(q_head)
        k_rot = torch.empty_like(k_head)
        q_rot[..., ::2] = q_rot_even
        q_rot[..., 1::2] = q_rot_odd
        k_rot[..., ::2] = k_rot_even
        k_rot[..., 1::2] = k_rot_odd
        q_out = torch.cat([q_rot, q_tail], dim=-1)
        k_out = torch.cat([k_rot, k_tail], dim=-1)
        return q_out, k_out

    def forward(self, hidden_states: torch.Tensor, past_key_value=None, use_cache: bool = False, position_ids: torch.Tensor = None):
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

        if position_ids is not None:
            # position_ids shape typically (bsz, seq_len)
            seq_positions = position_ids
        else:
            start = past_len
            seq_positions = torch.arange(start, start + seq_len, device=hidden_states.device).unsqueeze(0).expand(bsz, seq_len)

        q, k_new = self._apply_rope(q, k, seq_positions)

        if past_key_value is not None:
            k = torch.cat([pk, k_new], dim=2)
            v = torch.cat([pv, v], dim=2)
        else:
            k = k_new

        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, -1, self.hidden_size)
        out = self.out_proj(attn)
        present = None
        if use_cache:
            present = (k, v)
        return out, present
