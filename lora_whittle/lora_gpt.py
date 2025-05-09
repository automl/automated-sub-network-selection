from whittle.models.gpt.model import GPT as BaseModel
from typing import Any, Optional, Tuple
from typing_extensions import Self

import torch
import torch.nn as nn

from lora_whittle.config import LoRAConfig as Config
from litgpt.model import build_rope_cache

# from whittle.models.gpt.blocks import Block
from whittle.modules.linear import Linear
from whittle.modules.rmsnorm import RMSNorm
from whittle.modules.layernorm import LayerNorm
from lora_whittle.lora_linear import LoRALinear
from lora_whittle.lora_embedding import LoRAEmbedding
from lora_whittle.lora_block import LoRABlock as Block
from typing import Union, Dict, Tuple, Any, List
from litgpt.utils import map_old_state_dict_weights


class GPT(BaseModel):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lm_head = LoRALinear(
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
            r=(config.lora_r if config.lora_head else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=LoRAEmbedding(
                    config.padded_vocab_size,
                    config.n_embd,
                    r=(config.lora_r if config.lora_emb else 0),
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                ),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=self.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_layer = config.n_layer
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

        # Set current sub-network to super-network
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_intermediate_size = self.config.intermediate_size
        self.sub_network_num_heads = self.config.n_head
        self.sub_network_n_layers = self.config.n_layer
        self.sub_network_head_size: int | None = self.config.head_size
        self.sub_network_query_groups: int | None = self.config.n_query_groups
        self.sub_network_rope_n_elem = self.config.rope_n_elem
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        self.config.is_encoder_decoder = False
        self.main_input_name = "input_pos"
        self._supports_cache_class = True
        self.sub_network_head_size = None
        self.sampled_embd_indices = None
        self.sampled_intermediate_indices = None
        self.sampled_head_indices = None
        self.sampled_layer_indices = None
        self.sampled_query_groups_indices = None
        self.sampled_head_size_indices = None

    def forward(
        self,
        idx: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        lm_head_chunk_size: int = 0,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.sub_network_n_embd**0.5, dtype=x.dtype)
        if self.sampled_layer_indices is not None:
         for j in self.sampled_layer_indices:
            block = self.transformer.h[j]
            cos, sin = self.cos.to(idx.device), self.sin.to(idx.device)
            cos, sin, mask = self.process_rope_cache(cos, sin, input_pos, T)
            (
                x,
                _,
                _,
                _,
                _,
                _,
               _,
            ) = block(x, cos, sin, mask, input_pos)
        else:
            for i in range(self.sub_network_n_layers):
                block = self.transformer.h[i]

                cos, sin = self.cos.to(idx.device), self.sin.to(idx.device)
                cos, sin, mask = self.process_rope_cache(cos, sin, input_pos, T)
                (
                    x,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = block(x, cos, sin, mask, input_pos)

        x = self.transformer.ln_f(x)
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [self.lm_head(x_i) for x_i in x.split(lm_head_chunk_size, dim=1)]
        x = self.lm_head(x)  # (b, t, vocab_size)
        if self.config.final_logit_softcapping is not None:
            x = (
                torch.tanh(x / self.config.final_logit_softcapping)
                * self.config.final_logit_softcapping
            )
        return x
        # return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, LoRALinear):
            module.reset_parameters()

    def _load_from_state_dict(
        self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "lm_head.weight": "lm_head.linear.weight",
            "lm_head.bias": "lm_head.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, LoRALinear):
            module.reset_parameters()

    def _load_from_state_dict(
        self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "lm_head.weight": "lm_head.linear.weight",
            "lm_head.bias": "lm_head.linear.bias",
            "transformer.wte.weight": "transformer.wte.embedding.weight",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
