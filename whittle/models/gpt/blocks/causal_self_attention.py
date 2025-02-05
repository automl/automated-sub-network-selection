from __future__ import annotations

import math

import torch
import torch.nn as nn
from litgpt import Config
from litgpt.model import KVCache, apply_rope

from whittle.modules import Linear


class CausalSelfAttention(nn.Module):
    """Extension of litgpt's `litgpt.model.CausalSelfAttention` with support to adapt to sub-network dimensionality."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = Linear(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )
        # disabled by default
        self.kv_cache: KVCache | None = None
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None
            and block_idx % config.sliding_window_layer_placing == 0
        )
        self.config = config
        # Set current sub-network to super-network
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_n_head = self.config.n_head
        self.sub_network_head_size = self.config.head_size
        self.sub_network_qkv_shape = (
            self.config.n_head + 2 * self.config.n_query_groups
        ) * self.config.head_size
        self.sub_network_query_groups = self.config.n_query_groups
        self.sub_network_q_per_kv = (
            self.sub_network_n_head // self.sub_network_query_groups
        )
        self.sub_attention_scaler = self.config.attention_scores_scalar
        self.q_per_kv = self.config.n_head // self.config.n_query_groups
        self.qkv_indices = None
        self.proj_indices = None
    
        # intitialize head id to query group id mapping
    def get_qkv_indices(self, sampled_head_indices=None, sampled_query_groups_indices=None, sampled_head_size_indices=None):
        def process_indices(start, num_heads, size_indices, head_indices=None):
            if head_indices:
                num_heads = self.config.n_head // self.config.n_query_groups
                for h in head_indices:
                    qkv_indices.extend(start + h * self.config.head_size + i for i in size_indices)
            else:
                for h in range(num_heads):
                    qkv_indices.extend(start + h*self.config.head_size + i for i in size_indices)
            qkv_indices.extend(start + num_heads * self.config.head_size + i for i in size_indices)
            qkv_indices.extend(start + (num_heads + 1) * self.config.head_size + i for i in size_indices)

        qkv_indices = []
        heads_per_group =  self.config.n_head // self.config.n_query_groups
        if self.config.n_query_groups == 1 and sampled_head_indices:
                start = 0
                process_indices(start, 1, sampled_head_size_indices, sampled_head_indices)
        elif sampled_query_groups_indices and self.config.n_query_groups != self.config.n_head:
            for qg in sampled_query_groups_indices:
                base = qg * (heads_per_group + 2) * self.config.head_size
                if sampled_head_size_indices:
                    if sampled_head_indices:
                        process_indices(base, heads_per_group, sampled_head_size_indices, sampled_head_indices)
                    else:
                        process_indices(base, heads_per_group, sampled_head_size_indices)
                else:
                    if sampled_head_indices:
                        process_indices(base, heads_per_group, range(self.sub_network_head_size), sampled_head_indices)
                    else:
                        process_indices(base, heads_per_group, range(self.sub_network_head_size))

        elif self.config.n_head == self.config.n_query_groups and sampled_head_indices:
            for h in sampled_head_indices:
                base = 3 * h * self.config.head_size
                if sampled_head_size_indices:
                    process_indices(base, 1, sampled_head_size_indices)
                else:
                    process_indices(base, 1, range(self.sub_network_head_size))
        elif sampled_head_size_indices:
            if self.config.n_query_groups != self.config.n_head:
                for g in range(self.sub_network_query_groups):
                    base = g * heads_per_group * self.config.head_size
                    sampled_num_heads = self.q_per_kv
                    process_indices(base, sampled_num_heads, sampled_head_size_indices, [i for i in range(sampled_num_heads)])
            else:
                for h in range(self.sub_network_n_head):
                    base = 3*h * self.config.head_size
                    process_indices(base, 1, sampled_head_size_indices)
        else:
            qkv_indices = None

        return qkv_indices

    def get_proj_indices(self, sampled_head_indices=None, sampled_query_groups_indices=None, sampled_head_size_indices=None):
        def process_proj_indices(start, num_heads, size_indices, head_indices=None):
            if head_indices:
                for h in head_indices:
                    proj_indices.extend(start + h * self.config.head_size + i for i in size_indices)
            else:
                for h in range(num_heads):
                    proj_indices.extend(start + i for i in size_indices)
                    start += self.config.head_size

        proj_indices = []
        heads_per_group =  self.config.n_head // self.config.n_query_groups if self.config.n_query_groups != 1 else self.sub_network_n_head
        if (self.config.n_query_groups == 1 and sampled_head_indices) or (self.config.n_head == self.config.n_query_groups and sampled_head_indices):
            for h in sampled_head_indices:
                proj_base = h * self.config.head_size
                if sampled_head_size_indices:
                    process_proj_indices(proj_base, 1, sampled_head_size_indices)
                else:
                    proj_indices.extend(range(proj_base, proj_base + self.sub_network_head_size))
        elif sampled_query_groups_indices and self.config.n_query_groups != self.config.n_head:
            for qg in sampled_query_groups_indices:
                proj_base = qg * heads_per_group * self.config.head_size
                if sampled_head_size_indices:
                    if sampled_head_indices:
                        process_proj_indices(proj_base, heads_per_group, sampled_head_size_indices, sampled_head_indices)
                    else:
                        process_proj_indices(proj_base, heads_per_group, sampled_head_size_indices)
                else:
                    if sampled_head_indices:
                        process_proj_indices(proj_base, heads_per_group, range(self.sub_network_head_size), sampled_head_indices)
                    else:
                        for h in range(heads_per_group):
                            proj_indices.extend(range(proj_base, proj_base + self.sub_network_head_size))
                            proj_base += self.config.head_size

        elif sampled_head_size_indices:
            if self.config.n_query_groups != self.config.n_head:
                for gg in range(self.sub_network_query_groups):
                    proj_base = gg * heads_per_group * self.config.head_size
                    sampled_num_heads = self.q_per_kv
                    process_proj_indices(proj_base, sampled_num_heads, sampled_head_size_indices)
            else:
                for h in range(self.sub_network_n_head):
                    proj_base = h * self.config.head_size
                    process_proj_indices(proj_base, 1, sampled_head_size_indices)
        else:
            proj_indices = None

        return proj_indices
    
    def set_sub_network(
        self,
        sub_network_n_embd: int,
        sub_network_n_head: int,
        sub_network_query_groups: int,
        sub_network_head_size: int,
        sampled_head_indices: list[int] | None = None,
        sampled_embd_indices: list[int] | None = None,
        sampled_head_size_indices: list[int] | None = None,
        sampled_query_groups_indices: list[int] | None = None,
    ):
        """
        Sets the CausalSelfAttention block to the specified sub-network dimensionality.

        Args:
            sub_network_n_embd: Embedding dimension of the sub-network
            sub_network_n_head: Number of attention heads in the sub-network
            sub_network_query_groups: Number of query groups for grouped-query attention (GQA).
            sub_network_head_size: Size of each attention head in the sub-network.
        """
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_n_head = sub_network_n_head
        self.sub_network_query_groups = sub_network_query_groups
        self.sub_network_head_size = sub_network_head_size if sub_network_head_size else self.config.head_size
        if self.config.n_query_groups == 1:
            self.q_per_kv = self.sub_network_n_head if self.sub_network_n_head else self.config.n_head
            self.sub_network_query_groups = 1
            self.sub_network_n_head = self.q_per_kv
        elif self.config.n_head!=self.config.n_query_groups and self.config.n_query_groups!=1:
            if self.sub_network_query_groups is None:
                self.sub_network_query_groups = self.config.n_query_groups
            self.q_per_kv = self.sub_network_n_head//self.config.n_query_groups if self.sub_network_n_head else self.config.n_head//self.config.n_query_groups
        elif self.config.n_head == self.config.n_query_groups:
            self.q_per_kv = 1
            self.sub_network_query_groups = self.sub_network_n_head
        self.sub_network_qkv_shape = (
            (self.q_per_kv + 2) * self.sub_network_head_size * self.sub_network_query_groups)
        if self.config.n_query_groups == 1 or self.config.n_head != self.config.n_query_groups:
            if sampled_query_groups_indices is None:
                sampled_query_groups_indices = list(range(self.sub_network_query_groups))
            sampled_head_indices = list(range(self.q_per_kv)) if sampled_head_indices is None else sampled_head_indices         
            if sampled_head_size_indices is None:
                sampled_head_size_indices = list(range(self.sub_network_head_size))
        qkv_indices = self.get_qkv_indices(sampled_head_indices=sampled_head_indices, sampled_query_groups_indices=sampled_query_groups_indices, sampled_head_size_indices=sampled_head_size_indices)
        self.attn.set_sub_network(self.sub_network_n_embd, self.sub_network_qkv_shape, sampled_in_indices=sampled_embd_indices, sampled_out_indices=qkv_indices)
        proj_indices = self.get_proj_indices(sampled_head_indices=sampled_head_indices, sampled_query_groups_indices=sampled_query_groups_indices, sampled_head_size_indices=sampled_head_size_indices)
        self.proj.set_sub_network(
            self.sub_network_head_size * self.sub_network_query_groups * self.q_per_kv,
            self.sub_network_n_embd,
            sampled_in_indices=proj_indices,
            sampled_out_indices=sampled_embd_indices,
        )
        self.sub_network_q_per_kv = self.q_per_kv
        if self.config.attention_scores_scalar:
            self.sub_attention_scaler = (
                self.sub_network_n_embd // (self.sub_network_n_head)
            )
        else:
            self.sub_attention_scaler = self.config.attention_scores_scalar
        self.qkv_indices = qkv_indices
        self.proj_indices = proj_indices
        

    def reset_super_network(self):
        """Resets the dimensionality of the current sub-network to the super-network dimensionality."""
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_n_head = self.config.n_head
        self.q_per_kv = self.config.n_head // self.config.n_query_groups
        self.sub_network_head_size = self.config.head_size
        self.sub_network_qkv_shape = (
            self.config.n_head + 2 * self.config.n_query_groups
        ) * self.config.head_size
        self.sub_network_query_groups = self.config.n_query_groups
        self.sub_network_q_per_kv = self.q_per_kv

        self.attn.reset_super_network()
        self.proj.reset_super_network()
        self.sub_attention_scaler = self.config.attention_scores_scalar
        self.qkv_indices = None
        self.proj_indices = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (
            self.sub_network_n_embd is not None
        ), "You need to call `gpt.set_sub_network()"
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.attn(x)
        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        total_qkv = self.q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(
            B,
            T,
            self.sub_network_query_groups,
            total_qkv,
            self.sub_network_head_size,
        )

        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((self.q_per_kv, 1, 1), dim=2)
        
        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.sub_network_query_groups != (self.sub_network_query_groups*self.q_per_kv) and (
            input_pos is None or self.sub_network_query_groups != 1
        ):
            k = k.expand(
                B,
                self.sub_network_query_groups,
                self.q_per_kv,
                T,
                self.sub_network_head_size,
            )
            v = v.expand(
                B,
                self.sub_network_query_groups,
                self.q_per_kv,
                T,
                self.sub_network_head_size,
            )
        q = q.reshape(B, -1, T, self.sub_network_head_size)
        k = k.reshape(B, -1, T, self.sub_network_head_size)
        v = v.reshape(B, -1, T, self.sub_network_head_size)
        rope_n_elem = int(self.config.rotary_percentage*self.sub_network_head_size)
        # cos, sin = build_rope_cache(seq_len=T, n_elem=rope_n_elem,device=q.device)
        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)
        if self.apply_sliding_window_attention:
            """
                  Global Window              Sliding window             Sliding window
                  attention mask      +            bias          =      attention mask
            ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
            │ True False False False │  │ True  True  True True │  │ True  False False False │
            │ True True  False False │  │ True  True  True True │  │ True  True  False False │
            │ True True  True  False │  │ False True  True True │  │ False True  True  False │
            │ True True  True  True  │  │ False False True True │  │ False False True  True  │
            └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
            """
            if mask is None:
                mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), float("-inf"))
            sliding_window_bias = torch.ones_like(mask).tril(
                diagonal=-self.config.sliding_window_size
            )
            sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            mask += sliding_window_bias
        
        y = self.scaled_dot_product_attention(q, k, v, mask)
       
        y = y.reshape(
            B, T, self.sub_network_head_size * self.q_per_kv * self.sub_network_query_groups
        )  # re-assemble all head outputs side by side
        #print("after reshape", y)
        return self.proj(y), [q, k, v, mask]

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.sub_attention_scaler or self.sub_network_head_size)
        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            scale = 1.0 / math.sqrt(
                self.sub_attention_scaler or self.sub_network_head_size
            )
            scores = q @ k.mT * scale
            scores = (
                torch.tanh(scores / self.config.attention_logit_softcapping)
                * self.config.attention_logit_softcapping
            )
            if mask is None:
                mask = torch.ones(
                    q.size(2), q.size(2), dtype=q.dtype, device=q.device
                ).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            scores = scores + mask
            scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(
                dtype=q.dtype
            )
            y = scores @ v
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=scale,
                is_causal=mask is None,
            )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope_n_elem: int | None = None,
    ) -> KVCache:
        heads = 1 if self.sub_network_query_groups == 1 else self.sub_network_n_head
        v_shape = (batch_size, heads, max_seq_length, self.sub_network_head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            rope_n_elem = (
                rope_n_elem if rope_n_elem is not None else self.config.rope_n_elem
            )
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.sub_network_head_size - rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)

# test causal self attention
if __name__ == "__main__":
    config = Config()
    config.n_embd = 8
    config.n_head = 8
    config.head_size = 8
    config.n_layer = 1
    config.n_query_groups = 4
    config.attention_scores_scalar = 1
    config.rotary_percentage = 0.25
    config.max_seq_len = 8
    config.bias = True
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.dtype = torch.float32
    attention = CausalSelfAttention(config, 0)
    sampled_head_indices = None
    sampled_query_groups_indices = None
    sampled_head_size_indices = None
    attention.set_sub_network(sub_network_n_embd=4, sub_network_n_head=8, sub_network_query_groups=2, sub_network_head_size=4, sampled_head_indices=sampled_head_indices, sampled_query_groups_indices=sampled_query_groups_indices, sampled_head_size_indices=sampled_head_size_indices)
    q_per_kv = config.n_head // config.n_query_groups
    indices_test = torch.tensor([i for i in range(config.n_query_groups*(q_per_kv+2)*config.head_size)])
    indices_test = indices_test.view(config.n_query_groups, q_per_kv+2, config.head_size)
    if config.n_query_groups == config.n_head:
        indices = []
        if sampled_query_groups_indices is None:
            sampled_query_groups_indices = sampled_head_indices
        for i in sampled_query_groups_indices:
            indices.extend([i.item() for i in indices_test[i, :, sampled_head_size_indices].reshape(-1)])
    elif config.n_query_groups == 1:
        indices = []
        for h in sampled_head_indices:
            indices.extend([i.item() for i in indices_test[0, h, sampled_head_size_indices].reshape(-1)])
        indices.extend([i.item() for i in indices_test[0, -2, sampled_head_size_indices].reshape(-1)])
        indices.extend([i.item() for i in indices_test[0, -1, sampled_head_size_indices].reshape(-1)])
    else:
        indices = []
        if sampled_query_groups_indices is None:
            sampled_query_groups_indices = list(range(attention.sub_network_query_groups))
        if sampled_head_indices is None:
            sampled_head_indices = list(range(attention.sub_network_n_head//attention.config.n_query_groups))
        if sampled_head_size_indices is None:
            sampled_head_size_indices = list(range(attention.sub_network_head_size))
        for g in sampled_query_groups_indices:
            for h in sampled_head_indices:
                indices.extend([i.item() for i in indices_test[g, h, sampled_head_size_indices].reshape(-1)])
            indices.extend([i.item() for i in indices_test[g, -2, sampled_head_size_indices].reshape(-1)])
            indices.extend([i.item() for i in indices_test[g, -1, sampled_head_size_indices].reshape(-1)])
    print(indices)
    print(attention.qkv_indices)
    print(attention.proj_indices)
    input = torch.rand(8, 8, 4)
    cos = torch.rand(8, 8)
    sin = torch.rand(8, 8)
    out = attention(input, cos, sin)
    #print(out.size())
    