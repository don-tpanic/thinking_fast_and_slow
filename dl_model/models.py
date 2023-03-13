import os
import numpy as np
from collections import OrderedDict

import torch
from torch import nn 

try:
    import layers
except ModuleNotFoundError:
    from dl_model import layers


class DistrLearner(nn.Module):
    def __init__(self, config):
        super(DistrLearner, self).__init__()
        self.model_type = config['model_type']
        self.max_nunits = config['max_nunits']
        self.n_dims = config['n_dims']
        self.nn_sizes = [config['max_nunits'], 2]  # only association ws
        self.unit_recruit_mech = config['unit_recruit_type']
        self.Phi = config['Phi']
        self.attn_weighting = config['attn_weighting']
        # self.softmax = nn.Softmax(dim=0)
        # self.attn_trace = []
        # self.units_pos_trace = []
        # self.units_act_trace = []
        # self.recruit_unit_trl = []
        # self.fc1_w_trace = []
        # self.fc1_act_trace = []

        self.MultiVariateAttn = layers.MultiVariateAttention(
            n_dims=self.n_dims,
            max_nunits=self.max_nunits,
            attn_weighting=self.attn_weighting,
        )

        self.MaskNonRecruit = layers.Mask(
            in_features=self.max_nunits, 
            out_features=self.max_nunits,
            bias=False, trainable=False
        )

        self.ClsLayer = layers.Classfication(
            in_features=self.max_nunits, 
            out_features=2,
            bias=False, Phi=self.Phi
        )

        self.custom_lr_list = []

        self.custom_lr_list.append(
            {'params': \
                self.MultiVariateAttn.attn,
             'lr': \
                config['lr_attn']}
        )

        self.custom_lr_list.append(
            {'params': \
                self.MultiVariateAttn.units,
             'lr': \
                config['lr_units']}
        )

        self.custom_lr_list.append(
            {'params': \
                self.ClsLayer.parameters(),
             'lr': \
                config['lr_nn']}
        )

        if config['optim'] == 'sgd':
            self.optim = torch.optim.SGD(
                self.custom_lr_list, momentum=config['momentum'])
        if config['loss_fn'] == 'bcelogits':
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def need_recruit(self, y_logits, y_true):
        """
        Check if need recruit units.
        Conditions:
            Either makes an incorrect prediction or output is all 0s
            which suggests it is the first trial and no units are active
        """
        recruit = False
        if ((torch.argmax(y_logits).item() != torch.argmax(y_true).item()) or
               (torch.all(y_logits.data == 0))):  # if incorrect
            recruit = True
        return recruit
    
    def recruit_units(self, x, recruit):
        # Recruit cluster, and update model
        if self.unit_recruit_mech == 'feedback':
            if recruit and any(self.MaskNonRecruit.active_units == 0):  # in case recruit too many
                new_unit_ind = np.nonzero(self.MaskNonRecruit.active_units == 0)[0]
                self.MaskNonRecruit.active_units[new_unit_ind] = 1.
                self.MultiVariateAttn.units.data[new_unit_ind] = x  # place at curr stim
                # model.mask1[:, new_unit_ind] = True  # new clus weights
                # model.recruit_unit_trl.append(itrl)

    def make_prediction(self, x):
        act = self.MultiVariateAttn(x)
        act_masked = self.MaskNonRecruit(act)
        y_logits = self.ClsLayer(act_masked)
        return y_logits

    def forward(self, x, epoch, i, signature, y_true):
        y_logits = self.make_prediction(x)

        if self.need_recruit(y_logits, y_true):
            self.recruit_units(x, recruit=True)
            y_logits = self.make_prediction(x)
        
        return y_logits

