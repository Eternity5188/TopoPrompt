import torch
import torch.nn as nn
import torch.nn.init as init


class BatchGCNConv(nn.Module):
    """
    Simple batched GCN layer, inspired by Kipf & Welling (ICLR 2017): 
    "Semi-Supervised Classification with Graph Convolutional Networks" (arXiv:1609.02907).

    Supports two modes:
        - `gcn=True`:  standard GCN (only neighborhood aggregation)
        - `gcn=False`: adds a separate self-loop transformation (`x @ W_self`)
    
    Input:
        x: [B, N, in_channels]  – node features for B graphs (same topology)
        adj: [N, N]             – shared normalized adjacency matrix

    Output:
        out: [B, N, out_channels]
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, gcn: bool = True):
        super(BatchGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear transformation for aggregated neighbors
        self.weight_neigh = nn.Linear(in_channels, out_channels, bias=bias)

        # Optional separate transformation for self-features (used when gcn=False)
        if not gcn:
            self.weight_self = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.register_parameter('weight_self', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize parameters using default PyTorch strategies."""
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, N, in_channels] – batched node features
            adj: [N, N] – normalized adjacency matrix (shared across batch)

        Returns:
            output: [B, N, out_channels]
        """
        # Aggregate neighbors: adj @ x → [B, N, in_channels]
        # Note: torch.matmul broadcasts adj over the batch dimension
        input_x = torch.matmul(adj, x)  # [N, N] × [B, N, in_channels] → [B, N, in_channels]

        # Transform aggregated features
        output = self.weight_neigh(input_x)  # [B, N, out_channels]

        # Add self-feature transformation if enabled
        if self.weight_self is not None:
            output += self.weight_self(x)  # [B, N, out_channels]

        return output