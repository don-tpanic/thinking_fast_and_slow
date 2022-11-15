#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:25:46 2022

@author: robert.mok
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, sizes, act_func=nn.ReLU(), bias=True, dropout=0,
                 n_classes=2, device=torch.device('cpu')):
        super().__init__()
        self.act_func = act_func
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f, bias=bias)
             for in_f, out_f in zip(sizes, sizes[1:])])
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
