import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(
            self, 
            input_size,
            output_size,
            n_units,
            networks_depth,
            act_func=None, 
            bias=True, 
            dropout=0,
            n_classes=2, 
            device=torch.device('cpu')
        ):

        super().__init__()

        if networks_depth == 3:  # TODO: too much hardcoding
            sizes = [input_size, n_units, n_units, n_units, output_size]

        if act_func == 'relu':
            self.act_func = nn.ReLU()

        LayerCollections = []
        for in_f, out_f in zip(sizes, sizes[1:]):
            LayerCollections.append(
                nn.Linear(in_f, out_f, bias=bias)
            )
        self.layers = nn.ModuleList(LayerCollections)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i_l, layer in enumerate(self.layers):
            x = layer(x)
            # if linear act / final layer, no act_func
            if self.act_func and not layer == self.layers[-1]:
                if i_l > 0:
                    x = self.dropout(x)
                x = self.act_func(x)

        return x
