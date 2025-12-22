import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv import BatchGCNConv

class TopoPrompt(nn.Module):
    """
    TopoPrompt: A model based on baseline, using low-rank adaptive prompts
    for spatio-temporal forecasting. The prompt is expanded when new nodes appear.
    """

    def __init__(self, args):
        super(TopoPrompt, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank  # Low-rank dimension for node prompt
        self.num_nodes = args.base_node_size  # Initial number of nodes

        # GCN layers (spatial encoder)
        self.gcn1 = BatchGCNConv(
            in_channels=args.gcn["in_channel"],
            out_channels=args.gcn["hidden_channel"],
            bias=True,
            gcn=False
        )
        self.gcn2 = BatchGCNConv(
            in_channels=args.gcn["hidden_channel"],
            out_channels=args.gcn["out_channel"],
            bias=True,
            gcn=False
        )

        # TCN layer (temporal modeling)
        kernel_size = args.tcn["kernel_size"]
        dilation = args.tcn["dilation"]
        padding = (kernel_size - 1) * dilation // 2
        self.tcn1 = nn.Conv1d(
            in_channels=args.tcn["in_channel"],
            out_channels=args.tcn["out_channel"],
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )

        # Output projection
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

        # Low-rank node prompt: U ∈ ℝ^{N × r}, V ∈ ℝ^{r × d}
        self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))  # [N, r]
        self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))  # [r, d]

        # Low-rank edge prompt: U_edge ∈ ℝ^{N × r_edge}, V_edge ∈ ℝ^{N × r_edge}
        self.U_edge = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))  # [N, r_edge]
        self.V_edge = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))  # [N, r_edge]

        # Edge prompt scale parameter
        self.edge_prompt_scale = nn.Parameter(torch.tensor(0.5))
        
        # Node importance embedding
        self.node_importance_emb = nn.Parameter(
            torch.zeros(args.base_node_size, args.gcn["in_channel"])
        )

        # Importance estimator
        self.importance_mlp = nn.Sequential(
            nn.Linear(args.gcn["in_channel"], args.gcn["in_channel"] // 2),
            nn.ReLU(),
            nn.Linear(args.gcn["in_channel"] // 2, 1)
        )

        # Node Importance Gate scale (trainable)
        self.importance_scale = nn.Parameter(torch.tensor(0.1))

        # Feature mask MLP
        self.feat_mask_mlp = nn.Sequential(
            nn.Linear(args.gcn["in_channel"], args.gcn["in_channel"]),
            nn.ReLU(),
            nn.Linear(args.gcn["in_channel"], args.gcn["in_channel"])
        )

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def forward(self, data, adj):
        """
        Forward pass.

        Args:
            data: Batch object with attribute `x` of shape [B * N, T_in]
            adj: Normalized adjacency matrix of shape [N, N]

        Returns:
            output: Forecast tensor of shape [B * N, T_out]
        """
        N = adj.shape[0]  # Number of nodes in current graph
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))  # [B, N, T_in]
        B = x.shape[0]  # Batch size

        # Apply feature mask MLP
        x = x + self.feat_mask_mlp(x)

        # Apply low-rank adaptive prompt
        adaptive_prompt = torch.mm(self.U[:N, :], self.V)  # [N, d]
        edge_prompt = self.edge_prompt_scale * torch.mm(self.U_edge[:N, :], self.V_edge[:N, :].t())  # [N, d_edge]
        # edge_prompt = torch.mm(self.U_edge[:N, :], self.V_edge[:N, :].t())  # [N, d_edge]

        adj = adj + edge_prompt
        x = x + adaptive_prompt.unsqueeze(0) 
        
        # Node Importance Gate (MLP-based) 
        node_emb = self.node_importance_emb[:N]          # [N, d]
        importance = self.importance_mlp(node_emb)       # [N, 1]
        importance = torch.sigmoid(importance)           # [N, 1]

        # Residual importance calibration (VERY important)
        x = x * (1.0 + self.importance_scale * importance.unsqueeze(0))

        # First GCN layer
        x = F.relu(self.gcn1(x, adj))  # [B, N, hidden_dim]

        # Temporal convolution (reshape for Conv1d)
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))  # [B * N, 1, hidden_dim]
        x = self.tcn1(x)  # [B * N, 1, hidden_dim]
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))  # [B, N, hidden_dim]

        # Second GCN layer
        x = self.gcn2(x, adj)  # [B, N, out_dim]

        # Reshape and add residual connection
        x = x.reshape((-1, self.args.gcn["out_channel"]))  # [B * N, out_dim]
        x = x + data.x  # Residual: [B * N, out_dim]

        # Final projection and dropout
        x = self.fc(self.activation(x))  # [B * N, T_out]
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def expand_adaptive_params(self, new_num_nodes: int):
        """
        Expand the node-specific prompt matrix U when new nodes are added.

        Args:
            new_num_nodes (int): The new total number of nodes.
        """
        if new_num_nodes > self.num_nodes:
            # Sample new rows for U corresponding to new nodes
            new_U_rows = torch.empty(
                new_num_nodes - self.num_nodes,
                self.rank,
                dtype=self.U.dtype,
                device=self.U.device
            ).uniform_(-0.1, 0.1)

            # Concatenate old and new parameters
            self.U = nn.Parameter(torch.cat([self.U, new_U_rows], dim=0))  # [new_N, r]

            new_U_edge_rows = torch.empty(
                new_num_nodes - self.num_nodes,
                self.rank,
                dtype=self.U_edge.dtype,
                device=self.U_edge.device
            ).uniform_(-0.01, 0.01)

            new_V_edge_rows = torch.empty(
                new_num_nodes - self.num_nodes,
                self.rank,
                dtype=self.V_edge.dtype,
                device=self.V_edge.device
            ).uniform_(-0.01, 0.01)

            new_node_importance_rows = torch.zeros(
                new_num_nodes - self.num_nodes,
                self.node_importance_emb.shape[1],
                dtype=self.node_importance_emb.dtype,
                device=self.node_importance_emb.device
            )
            self.U_edge = nn.Parameter(torch.cat([self.U_edge, new_U_edge_rows], dim=0))
            self.V_edge = nn.Parameter(torch.cat([self.V_edge, new_V_edge_rows], dim=0))
            self.node_importance_emb = nn.Parameter(torch.cat([self.node_importance_emb, new_node_importance_rows], dim=0))

            self.num_nodes = new_num_nodes