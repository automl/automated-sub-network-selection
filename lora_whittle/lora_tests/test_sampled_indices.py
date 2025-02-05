from __future__ import annotations

import torch
from litgpt import Config as LitConfig
from lora_whittle.config import LoRAConfig as Config
from litgpt.model import GPT as LitGPT
import random
from lora_whittle.lora_gpt import GPT


def test_gpt_sampled():
    torch.manual_seed(0)
    config = Config()
    config.padded_vocab_size = 128
    config.n_embd = 64
    config.intermediate_size = 64 * 4
    config.n_head = 8
    config.n_query_groups = 4
    config.head_size = 8
    config.n_layer = 3
    config.block_size = 128
    config.rotary_percentage = 0.25
    config.norm_class_name = "RMSNorm"
    config.mlp_class_name = "LLaMAMLP"
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    config.norm_eps = 1e-5
    config.lm_head_bias = True
    config.fix_head_size = False
    litconfig = LitConfig()
    litconfig.padded_vocab_size = 128
    litconfig.n_embd = 64
    litconfig.intermediate_size = 64 * 4
    litconfig.n_head = 8
    litconfig.n_query_groups = 4
    litconfig.head_size = 8
    litconfig.n_layer = 3
    litconfig.block_size = 128
    litconfig.rotary_percentage = 0.25
    litconfig.norm_class_name = "RMSNorm"
    litconfig.mlp_class_name = "LLaMAMLP"
    litconfig.rope_n_elem = int(litconfig.rotary_percentage * litconfig.head_size)
    litconfig.norm_eps = 1e-5
    litconfig.lm_head_bias = True
    litconfig.fix_head_size = False
    gpt = GPT(config)
    gpt.transformer.wte.embedding.weight.data = torch.randn_like(gpt.transformer.wte.embedding.weight.data)
    gpt.lm_head.linear.weight.data = torch.randn_like(gpt.lm_head.linear.weight.data)
    gpt.lm_head.linear.bias.data = torch.randn_like(gpt.lm_head.linear.bias.data)
    gpt.transformer.ln_f.weight.data = torch.randn_like(
        gpt.transformer.ln_f.weight.data
    )

    for block in gpt.transformer.h:
        block.attn.attn.linear.weight.data = torch.randn_like(block.attn.attn.linear.weight.data)
        block.attn.attn.linear.bias.data = torch.randn_like(block.attn.attn.linear.bias.data)
        block.attn.proj.linear.bias.data = torch.randn_like(block.attn.proj.linear.bias.data)
        block.attn.proj.linear.weight.data = torch.randn_like(block.attn.proj.linear.weight.data)
        block.mlp.fc_1.linear.weight.data = torch.randn_like(block.mlp.fc_1.linear.weight.data)
        block.mlp.fc_1.linear.bias.data = torch.randn_like(block.mlp.fc_1.linear.bias.data)
        block.mlp.fc_2.linear.weight.data = torch.randn_like(block.mlp.fc_2.linear.weight.data)
        block.mlp.fc_2.linear.bias.data = torch.randn_like(block.mlp.fc_2.linear.bias.data)
        block.mlp.proj.linear.weight.data = torch.randn_like(block.mlp.proj.linear.weight.data)
        block.mlp.proj.linear.bias.data = torch.randn_like(block.mlp.proj.linear.bias.data)
        block.norm_1.weight.data = torch.randn_like(block.norm_1.weight.data)
        block.norm_2.weight.data = torch.randn_like(block.norm_2.weight.data)

    gpt.reset_super_network()
    input = torch.randint(0, 128, (1,128))
    out_large = gpt(input)
    assert out_large.shape == (1, 128, 128)

    lit_gpt = LitGPT(litconfig)
    lit_gpt.lm_head.weight.data = gpt.lm_head.linear.weight.data
    lit_gpt.lm_head.bias.data = gpt.lm_head.linear.bias.data
    lit_gpt.transformer.wte.weight.data = gpt.transformer.wte.embedding.weight.data
    lit_gpt.transformer.ln_f.weight.data = gpt.transformer.ln_f.weight.data
    for i, block in enumerate(lit_gpt.transformer.h):
        block_orig = gpt.transformer.h[i]
        block.attn.attn.weight.data = block_orig.attn.attn.linear.weight.data
        block.attn.attn.bias.data = block_orig.attn.attn.linear.bias.data
        block.attn.proj.bias.data = block_orig.attn.proj.linear.bias.data
        block.attn.proj.weight.data = block_orig.attn.proj.linear.weight.data
        block.mlp.fc_1.weight.data = block_orig.mlp.fc_1.linear.weight.data
        block.mlp.fc_1.bias.data = block_orig.mlp.fc_1.linear.bias.data
        block.mlp.fc_2.weight.data = block_orig.mlp.fc_2.linear.weight.data
        block.mlp.fc_2.bias.data = block_orig.mlp.fc_2.linear.bias.data
        block.mlp.proj.weight.data = block_orig.mlp.proj.linear.weight.data
        block.mlp.proj.bias.data = block_orig.mlp.proj.linear.bias.data
        block.norm_1.weight.data = block_orig.norm_1.weight.data
        block.norm_2.weight.data = block_orig.norm_2.weight.data

    out_lit_large = lit_gpt(input)
    assert torch.allclose(out_lit_large, out_large, atol=1e-3)
    gpt.set_sub_network(
        sub_network_n_embd=32,
        sub_network_intermediate_size=32 * 4,
        sub_network_num_heads=4,
        sub_network_n_layers=2,
        sub_network_query_groups=2,
        sub_network_head_size=4,
        sampled_intermediate_indices = list(random.sample(range(gpt.config.intermediate_size), 32*4)),
        sampled_embd_indices = list(random.sample(range(gpt.config.n_embd), 32)),
        sampled_query_group_indices= list(random.sample(range(gpt.config.n_query_groups), 2)),
        sampled_layer_indices= list(sorted(random.sample(range(gpt.config.n_layer), 2))),
        sampled_head_size_indices= list(random.sample(range(gpt.config.head_size), 4))
    )
    out_small = gpt(input)
    assert out_small.shape == (1, 128, 128)
    litconfig.n_embd = 32
    litconfig.n_head = 2
    litconfig.rotary_percentage = 0.25
    litconfig.n_query_groups = 2
    litconfig.head_size = 4
    litconfig.intermediate_size = 32 * 4
    litconfig.rope_n_elem = int(config.rotary_percentage * litconfig.head_size)
    litconfig.n_layer = 2
    lit_gpt_small = LitGPT(litconfig)
    lit_gpt_small.lm_head.weight.data = gpt.lm_head.linear.weight.data[
        : gpt.lm_head.sub_network_out_features, gpt.sampled_embd_indices]
    lit_gpt_small.lm_head.bias.data = gpt.lm_head.linear.bias.data[:]
    lit_gpt_small.transformer.wte.weight.data = gpt.transformer.wte.embedding.weight.data[
        :, gpt.sampled_embd_indices
    ]
    lit_gpt_small.transformer.ln_f.weight.data = gpt.transformer.ln_f.weight.data[
        gpt.sampled_embd_indices
    ]

    for i, block in enumerate(lit_gpt_small.transformer.h):
        block_orig = gpt.transformer.h[gpt.sampled_layer_indices[i]]
        block.attn.attn.weight.data = block_orig.attn.attn.linear.weight.data[block_orig.attn.qkv_indices,:][:,gpt.sampled_embd_indices]
        block.attn.attn.bias.data = block_orig.attn.attn.linear.bias.data[
            block_orig.attn.qkv_indices
        ]
        block.attn.proj.bias.data = block_orig.attn.proj.linear.bias.data[
            gpt.sampled_embd_indices
        ]
        block.attn.proj.weight.data = block_orig.attn.proj.linear.weight.data[
            gpt.sampled_embd_indices,:][:,block_orig.attn.proj_indices]
        block.mlp.fc_1.weight.data = block_orig.mlp.fc_1.linear.weight.data[
            gpt.sampled_intermediate_indices,:][:,gpt.sampled_embd_indices]
        block.mlp.fc_1.bias.data = block_orig.mlp.fc_1.linear.bias.data[
            gpt.sampled_intermediate_indices
        ]
        block.mlp.fc_2.weight.data = block_orig.mlp.fc_2.linear.weight.data[
            gpt.sampled_intermediate_indices,:][:,gpt.sampled_embd_indices]
        block.mlp.fc_2.bias.data = block_orig.mlp.fc_2.linear.bias.data[
            gpt.sampled_intermediate_indices
        ]
        block.mlp.proj.weight.data = block_orig.mlp.proj.linear.weight.data[
            gpt.sampled_embd_indices,:][:,gpt.sampled_intermediate_indices]
        block.mlp.proj.bias.data = block_orig.mlp.proj.linear.bias.data[
            gpt.sampled_embd_indices
        ]
        block.norm_1.weight.data = block_orig.norm_1.weight.data[
            gpt.sampled_embd_indices
        ]
        block.norm_2.weight.data = block_orig.norm_2.weight.data[
            gpt.sampled_embd_indices
        ]
    out_lit_small = lit_gpt_small(input)
    print(out_lit_small)
    print(out_small)
    print(torch.sum(out_lit_small - out_small))
    assert torch.allclose(out_lit_small, out_small, atol=1e-3)

