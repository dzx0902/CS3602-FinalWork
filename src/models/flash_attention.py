import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from torch.backends.cuda import sdp_kernel
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb


class FlashSelfAttention(nn.Module):
    """
    自己实现的一版 Attention（目前在 PythiaFlashModel 中没有被实际使用，
    现在的 flash 模式是通过 sdp_kernel 包裹原始 GPT-NeoX Attention）。
    保留这个类，以后如果你想换成“完全自定义的 FlashAttention”可以直接复用。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_base: float = 10000.0,
        rotary_pct: float = 1.0,
        rotary_emb: nn.Module | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.rotary_base = rotary_base
        self.rotary_pct = rotary_pct
        self.rotary_emb = rotary_emb

    def _apply_rope(
        self,
        q: torch.Tensor,  # (B, H, L, D)
        k: torch.Tensor,  # (B, H, L, D)
        position_ids: torch.Tensor,  # (B, L)
    ):
        """
        为了和 GPT-NeoX 的实现保持一致，这里复用原模型的 rotary_emb + apply_rotary_pos_emb。
        如果 rotary_emb 不存在，直接返回原 q, k。
        """
        dtype = q.dtype

        if self.rotary_emb is None:
            return q, k

        # 这里 total_seq_len 只是为兼容不同版本的 rotary_emb 接口
        total_seq_len = int(position_ids.max().item()) + 1

        # 尝试兼容两种常见签名：
        #   rotary_emb(k, seq_len=...) 或 rotary_emb(k)
        try:
            cos, sin = self.rotary_emb(k, seq_len=total_seq_len)
        except TypeError:
            try:
                cos, sin = self.rotary_emb(k)
            except TypeError:
                # 实在不匹配就直接放弃 RoPE
                return q, k

        # 某些实现里 cos/sin 的最后一维可能小于 head_dim，这里做一个简单 padding
        if cos.size(-1) != q.size(-1):
            tail = q.size(-1) - cos.size(-1)
            if tail > 0:
                cos_tail = torch.ones((*cos.shape[:-1], tail), device=cos.device, dtype=cos.dtype)
                sin_tail = torch.zeros((*sin.shape[:-1], tail), device=sin.device, dtype=sin.dtype)
                cos = torch.cat([cos, cos_tail], dim=-1)
                sin = torch.cat([sin, sin_tail], dim=-1)

        # 直接复用 HF 的 apply_rotary_pos_emb，保持和原 Attention 一致
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        return q_rot.to(dtype), k_rot.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, L, D)
        past_key_value=None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        bsz, seq_len, _ = hidden_states.shape

        # 线性得到 qkv
        qkv = self.qkv_proj(hidden_states)    # (B, L, 3D)
        q, k, v = qkv.split(self.hidden_size, dim=-1)  # 各 (B, L, D)

        # 变换到 (B, H, L, d)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_key_value is not None:
            pk, pv = past_key_value
            past_len = pk.size(2)  # (B, H, L_past, d)

        # 生成 position_ids（如果没传的话）
        if position_ids is None:
            start = past_len
            position_ids = torch.arange(
                start, start + seq_len, device=hidden_states.device
            ).unsqueeze(0).expand(bsz, seq_len)  # (B, L)

        # 应用 RoPE（使用原始 rotary_emb）
        q, k_new = self._apply_rope(q, k, position_ids)

        # 拼接 KV cache
        if past_key_value is not None:
            k = torch.cat([pk, k_new], dim=2)
            v = torch.cat([pv, v], dim=2)
        else:
            k = k_new

        # 处理 padding mask，转成 SDPA 需要的 bool 掩码
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: (B, L_curr) ==1 有效, 0 padding
            if attention_mask.dim() == 2:
                if past_len > 0:
                    kv_mask = torch.ones(
                        (bsz, past_len),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    kv_mask = torch.cat([kv_mask, attention_mask], dim=1)  # (B, L_total)
                else:
                    kv_mask = attention_mask
                # SDPA 期望形状 (B, 1, L_q, L_kv)
                attn_mask = (kv_mask == 0).view(bsz, 1, 1, -1).expand(
                    bsz, 1, q.size(2), kv_mask.size(1)
                )

        # 这里不强制指定 kernel，真正的 flash / math 选择由外部 sdp_kernel context 决定
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=True,
        )  # (B, H, L, d)

        attn = attn.transpose(1, 2).contiguous().view(bsz, -1, self.hidden_size)
        out = self.out_proj(attn)

        present = None
        if use_cache:
            present = (k, v)
        return out, present


@contextmanager
def use_math_sdpa():
    """
    只启用 math kernel：无论 dtype 是 float32/float16 都能跑，
    不会触发 Flash / memory efficient。
    """
    with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        yield


@contextmanager
def use_flash_sdpa():
    """
    优先尝试 Flash kernel，但允许回退到 math：
    - 当 dtype 是 Half/BFloat16 且其他条件满足 → 用 Flash
    - 否则自动 fallback 到 math（因为 enable_math=True）
    这样就不会再出现 No available kernel 的错误。
    """
    with sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        yield
