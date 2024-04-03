import torch.nn as nn
import torch.nn.functional as F


class BaselineClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        n_classes: int,
        n_layers: int,
        p_dropout: float,
    ):
        super(BaselineClassifier, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_channels))

        for _ in range(1, n_layers):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.Dropout(p_dropout))

        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_channels, n_classes)

    def forward(self, data):
        x = data["x"]
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        return self.output(x)
