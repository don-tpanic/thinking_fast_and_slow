import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(
        self, config, device=torch.device('cpu')):
        super().__init__()

        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.n_units = config['n_units']
        self.networks_depth = config['networks_depth']
        self.act_func = config['act_func']
        self.bias = True 
        self.dropout = config['dropout']
        self.n_classes= 2

        if self.networks_depth == 3:  # TODO: too much hardcoding
            sizes = [
                self.input_size, 
                self.n_units, 
                self.n_units, 
                self.n_units, 
                self.output_size
            ]

        if self.act_func == 'relu':
            self.act_func = nn.ReLU()

        LayerCollections = []
        for in_f, out_f in zip(sizes, sizes[1:]):
            LayerCollections.append(
                nn.Linear(in_f, out_f, bias=self.bias)
            )
        self.layers = nn.ModuleList(LayerCollections)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        for i_l, layer in enumerate(self.layers):
            x = layer(x)
            # if linear act / final layer, no act_func
            if self.act_func and not layer == self.layers[-1]:
                if i_l > 0:
                    x = self.dropout(x)
                x = self.act_func(x)

        return x
