from __future__ import annotations


import litgpt
import torch
from litgpt import Config

from whittle.modules import Linear


class GptNeoxMLP(litgpt.model.GptNeoxMLP):
    """An extension of litgpt's `litgpt.model.GptNeoxMLP` with support to adapt to sub-network dimensionality."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.fc = Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config
        self.in_features = config.n_embd
        self.intermediate_size = config.intermediate_size

        # Set current sub-network to super-network
        self.sub_network_n_embd = self.in_features
        self.sub_network_intermediate_size = self.intermediate_size

    def set_sub_network(
        self, sub_network_n_embd: int, sub_network_intermediate_size: int, sampled_intermediate_indices: list[int] | None = None, sampled_embd_indices: list[int] | None = None
    ):
        """
        Sets the dimensionality of the current sub-network MLP layers.

        Args:
           sub_network_n_embd: Input and output embedding dimension of the sub-network.
           sub_network_intermediate_size: Hidden layer dimension of the sub-network MLP.
        """
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_intermediate_size = sub_network_intermediate_size

        self.fc.set_sub_network(
            self.sub_network_n_embd, self.sub_network_intermediate_size, sampled_in_indices=sampled_embd_indices, sampled_out_indices=sampled_intermediate_indices
        )
        self.proj.set_sub_network(
            self.sub_network_intermediate_size, self.sub_network_n_embd, sampled_in_indices=sampled_intermediate_indices, sampled_out_indices=sampled_embd_indices
        )

    def reset_super_network(self):
        """Resets the MLP dimensions to the original super-network dimensionality."""
        self.sub_network_n_embd = self.in_features
        self.sub_network_intermediate_size = self.intermediate_size

        self.fc.reset_super_network()
        self.proj.reset_super_network()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc = self.fc(x)
        # print("Sum after fc", torch.sum(x_fc))
        x = torch.nn.functional.gelu(x_fc, approximate=self.config.gelu_approximate)
        # print("Sum after proj", torch.sum(self.proj(x)))
        return self.proj(x), x_fc

class LLaMAMLP(litgpt.model.LLaMAMLP):
    """An extension of litgp's `litgpt.model.LLaMAMLP` with support to adapt to sub-network dimensionality."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.fc_1 = Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.in_features = config.n_embd
        self.intermediate_size = config.intermediate_size
        self.sub_network_n_embd: int | None = None
        self.sub_network_intermediate_size: int | None = None
        self.config = config

    def set_sub_network(
        self, sub_network_n_embd: int, sub_network_intermediate_size: int, sampled_intermediate_indices: list[int] | None = None, sampled_embd_indices: list[int] | None = None
    ):
        """
        Sets the dimensionality of the current sub-network MLP layers.

        Args:
            sub_network_n_embd: Input and output embedding dimension of the sub-network.
            sub_network_intermediate_size: Hidden layer dimension of the sub-network MLP.
        """
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_intermediate_size = sub_network_intermediate_size

        self.fc_1.set_sub_network(
            self.sub_network_n_embd, self.sub_network_intermediate_size, sampled_in_indices=sampled_embd_indices, sampled_out_indices=sampled_intermediate_indices
        )
        self.fc_2.set_sub_network(
            self.sub_network_n_embd, self.sub_network_intermediate_size, sampled_in_indices=sampled_embd_indices, sampled_out_indices=sampled_intermediate_indices
        )
        self.proj.set_sub_network(
            self.sub_network_intermediate_size, self.sub_network_n_embd, sampled_in_indices=sampled_intermediate_indices, sampled_out_indices=sampled_embd_indices
        )

    def reset_super_network(self):
        """Reset the input dimensionality of the current sub-network to the super-network dimensionality."""
        self.sub_network_n_embd = self.in_features
        self.sub_network_intermediate_size = self.intermediate_size

        self.fc_1.reset_super_network()
        self.fc_2.reset_super_network()
        self.proj.reset_super_network()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x), x_fc_1

class GemmaMLP(LLaMAMLP):
    """Implementation of the forward pass of LLaMAMLP network."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = (
            torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate)
            * x_fc_2
        )
        return self.proj(x), x_fc_1
