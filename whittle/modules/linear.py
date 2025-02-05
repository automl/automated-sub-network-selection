from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    """An extension of PyTorch's torch.nn.Linear with flexible input and output dimensionality corresponding to sub-network"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)

        # Set the current sub-network dimensions equal to super-network
        self.sub_network_in_features = in_features
        self.sub_network_out_features = out_features
        self.use_bias = bias
        self.sampled_in_indices : list[int] | None = None
        self.sampled_out_indices : list[int] | None = None

    def set_sub_network(
        self, sub_network_in_features: int, sub_network_out_features: int, sampled_in_indices: list[int] | None = None, sampled_out_indices: list[int] | None = None
    ):
        """Set the linear transformation dimensions of the current sub-network."""
        self.sub_network_in_features = sub_network_in_features
        self.sub_network_out_features = sub_network_out_features
        self.sampled_in_indices = sampled_in_indices
        self.sampled_out_indices = sampled_out_indices

    def reset_super_network(self):
        """Reset the linear transformation dimensions of the current sub-network to the super-network dimensionality."""
        self.sub_network_in_features = self.in_features
        self.sub_network_out_features = self.out_features
        self.sampled_in_indices = None
        self.sampled_out_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            if self.sampled_in_indices is not None or self.sampled_out_indices is not None:
                if self.sampled_out_indices is not None and self.sampled_in_indices is not None:
                    return F.linear(
                        x,
                        self.weight[
                            self.sampled_out_indices,:][:, self.sampled_in_indices],
                        self.bias[self.sampled_out_indices],
                    )
                elif self.sampled_in_indices is not None:
                    return F.linear(
                        x,
                        self.weight[
                            : self.sub_network_out_features, self.sampled_in_indices
                        ],
                        self.bias[: self.sub_network_out_features],
                    )
                else:
                    return F.linear(
                        x,
                        self.weight[
                            self.sampled_out_indices, : self.sub_network_in_features
                        ],
                        self.bias[self.sampled_out_indices],
                    )
            else:
                return F.linear(
                    x,
                    self.weight[
                        : self.sub_network_out_features, : self.sub_network_in_features
                    ],
                    self.bias[: self.sub_network_out_features],
                )
        else:
            if self.sampled_in_indices is not None or self.sampled_out_indices is not None:
                if self.sampled_out_indices is not None and self.sampled_in_indices is not None:
                    return F.linear(
                        x,
                        self.weight[
                            self.sampled_out_indices,:][:, self.sampled_in_indices],
                    )
                elif self.sampled_in_indices is not None:
                    return F.linear(
                        x,
                        self.weight[
                            : self.sub_network_out_features, self.sampled_in_indices
                        ],
                    )
                else:
                    return F.linear(
                        x,
                        self.weight[
                            self.sampled_out_indices, : self.sub_network_in_features
                        ],
                    )
            else:

                return F.linear(
                    x,
                    self.weight[
                        : self.sub_network_out_features, : self.sub_network_in_features
                    ],
                )
