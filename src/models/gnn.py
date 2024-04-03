from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Linear
from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from torch_geometric.nn import (
    GCNConv,
    GINConv,
)
from torch_geometric.nn import JumpingKnowledge


class SimpleGNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        n_classes: int = 1,
        n_layers: int = 2,
        p_dropout: float = 0.5,
        pool: Literal["mean", "add", "max"] = "mean",
    ):
        super(SimpleGNN, self).__init__()
        self.lin_in = Linear(input_dim, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for layer in range(n_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.out = Linear(hidden_channels, n_classes)
        self.p_dropout = p_dropout
        self.pool = global_mean_pool
        if pool == "max":
            self.pool = global_max_pool
        elif pool == "add":
            self.pool = global_add_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.lin_in(x.float())

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out(x)

        return x


class AdvancedGNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        n_classes: int = 1,
        n_layers: int = 2,
        p_dropout: float = 0.5,
        pool: Literal["mean", "add", "max"] = "mean",
    ):
        super(AdvancedGNN, self).__init__()
        self.lin_in = Linear(input_dim, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(n_layers):
            self.convs.append(
                GINConv(
                    torch.nn.Sequential(
                        Linear(hidden_channels, hidden_channels),
                        BatchNorm1d(hidden_channels),
                        torch.nn.ReLU(),
                        Linear(hidden_channels, hidden_channels),
                    )
                )
            )
            self.batch_norms.append(BatchNorm1d(hidden_channels))

        self.out = Linear(hidden_channels, n_classes)
        self.p_dropout = p_dropout

        # Pooling function
        self.pool = global_mean_pool
        if pool == "max":
            self.pool = global_max_pool
        elif pool == "add":
            self.pool = global_add_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.lin_in(x.float())

        for conv, bn in zip(self.convs, self.batch_norms):
            skip = x  # Save for the skip connection
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = x + skip  # Add the skip connection
            x = F.dropout(x, p=self.p_dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out(x)

        return x


class JKAdvancedGNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        n_classes: int = 1,
        n_layers: int = 2,
        p_dropout: float = 0.5,
        pool: Literal["mean", "add", "max"] = "mean",
        jk_mode: Literal["cat", "max", "lstm"] = "cat",
    ):
        super(JKAdvancedGNN, self).__init__()
        self.lin_in = Linear(input_dim, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.layer_outputs = []  # To store outputs of each layer for JK

        for layer in range(n_layers):
            self.convs.append(
                GINConv(
                    torch.nn.Sequential(
                        Linear(hidden_channels, hidden_channels),
                        BatchNorm1d(hidden_channels),
                        torch.nn.ReLU(),
                        Linear(hidden_channels, hidden_channels),
                    )
                )
            )
            self.batch_norms.append(BatchNorm1d(hidden_channels))

        self.jk = JumpingKnowledge(
            mode=jk_mode, channels=hidden_channels, num_layers=n_layers
        )
        self.jk_lin = Linear(
            hidden_channels * n_layers if jk_mode == "cat" else hidden_channels,
            n_classes,
        )

        self.p_dropout = p_dropout

        # Pooling function
        self.pool = global_mean_pool
        if pool == "max":
            self.pool = global_max_pool
        elif pool == "add":
            self.pool = global_add_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin_in(x.float())
        self.layer_outputs = []

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)
            self.layer_outputs.append(x)

        # Use Jumping Knowledge to combine layer outputs
        x = self.jk(self.layer_outputs)
        x = self.pool(x, batch)
        x = self.jk_lin(x)

        return x


class GATGNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        n_classes: int = 1,
        n_layers: int = 2,
        n_heads: int = 1,
        p_dropout: float = 0.5,
        pool: Literal["mean", "add", "max"] = "mean",
    ):
        super(GATGNN, self).__init__()
        self.lin_in = Linear(input_dim, hidden_channels)

        # Intermediate GAT layers
        self.convs = torch.nn.ModuleList()
        for _ in range(1, n_layers - 1):
            self.convs.append(
                GATConv(
                    hidden_channels * n_heads,
                    hidden_channels,
                    heads=n_heads,
                    dropout=p_dropout,
                    concat=True,
                )
            )

        # Final GAT layer
        self.conv_out = GATConv(
            hidden_channels * n_heads, hidden_channels, heads=1, concat=False
        )

        # Output layer
        self.out = Linear(hidden_channels, n_classes)

        # Pooling function
        self.pool = global_mean_pool
        if pool == "max":
            self.pool = global_max_pool
        elif pool == "add":
            self.pool = global_add_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.lin_in(x.float())

        for conv in self.convs:
            # skip = x  # Save for the skip connection
            x = conv(x, edge_index)
            # x = bn(x)
            x = F.relu(x)
            # x = x + skip  # Add the skip connection
            # x = F.dropout(x, p=self.p_dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out(x)

        return x

        # # Initial GAT layer
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = F.elu(self.conv1(x, edge_index))
        #
        # # Intermediate GAT layers
        # for conv in self.convs:
        #     x = F.dropout(x, p=self.p_dropout, training=self.training)
        #     x = F.elu(conv(x, edge_index))
        #
        # # Final GAT layer
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = self.conv_out(x, edge_index)
        #
        # # Pooling
        # x = self.pool(x, batch)
        #
        # # Output layer
        # x = self.out(x)
        #
        # return x
