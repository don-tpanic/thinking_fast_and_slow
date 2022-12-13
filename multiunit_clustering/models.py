import os
import numpy as np
from collections import OrderedDict

import torch
from torch import nn 

try:
    import layers
except ModuleNotFoundError:
    from multiunit_clustering import layers


class MultiUnitCluster(nn.Module):
    def __init__(self, config):

        super(MultiUnitCluster, self).__init__()
        self.attn_type = config['attn_type']
        self.n_units = config['n_units']
        self.n_dims = config['n_dims']
        self.r = config['r']
        self.c = config['c']
        self.n_classes = config['n_classes']
        self.phi = config['phi']
        self.lr_nn = config['lr_nn']
        self.lr_clusters = config['lr_clusters']
        self.lr_attn = config['lr_attn']
        self.k = config['k']
        self.noise = config['noise']
        self.lesion = config['lesion']

        # # history
        # self.attn_trace = []
        # self.units_pos_trace = []
        # self.units_pos_bothupd_trace = []
        # self.units_act_trace = []
        # self.recruit_units_trl = []
        # self.fc1_w_trace = []
        # self.fc1_act_trace = []

        self.Distance = layers.Distance(n_units=self.n_units, n_dims=self.n_dims, r=self.r)
        self.DimensionWiseAttnLayer = layers.DimWiseAttention(n_dims=self.n_dims, trainable=True)  # TODO: better semantic for trainable
        self.ClusterActvLayer = layers.ClusterActivation(q=1, r=self.r, specificity=self.c)
        self.MaskLayer = layers.Mask(n_units=self.n_units, topk=self.k)
        self.ClsLayer = layers.Classfication(n_units=self.n_units, n_classes=self.n_classes, phi=self.phi, bias=False)
        
        # adjust asso lr
        self.custom_lr_list = []
        self.custom_lr_list.append(
            {'params': self.ClsLayer.parameters(), 
            'lr': self.lr_nn}
        )

        # set optimizer and loss fn
        if config['optim'] == 'sgd':
            self.optim = torch.optim.SGD(self.custom_lr_list,)
        if config['loss_fn'] == 'bcelogits':
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        if config['loss_fn'] == 'crossentropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()

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

    def recruit_units(self, x, act, epoch, i, recruit, win_ind, y_true):

        # Recruit cluster, and update model
        les_units = [] # TODO: complete later
        if (torch.tensor(recruit)
            and torch.sum(self.ClsLayer.weight == 0) > 0
            and len(les_units) < self.n_units):  # if no more units, stop

            # 1st trial - select closest k inactive units
            if epoch == 0 and i == 0:
                _, recruit_ind = ( torch.topk( act, int(self.n_units * self.k) ) )

                # since topk takes top even if all 0s, remove the 0 acts
                if torch.any(act[recruit_ind] == 0):
                    recruit_ind = recruit_ind[act[recruit_ind] != 0]

            # recruit and REPLACE k units that mispredicted
            else:
                if len(win_ind):  # else all lesioned
                    mispred_units = (
                        torch.argmax(
                            self.ClsLayer.weight.data[win_ind, :], dim=1) \
                                != torch.argmax(y_true)
                    )
                else:
                    mispred_units = torch.tensor(0)
                
                # select closest n_mispredicted inactive units
                n_mispred_units = mispred_units.sum()
                act[(self.MaskLayer.active_units == 1)] = 0  # REMOVE all active units
                act[les_units] = 0  # REMOVE all lesioned units

                # find closest units excluding the active units to recruit
                _, recruit_ind = ( torch.topk(act, n_mispred_units) )

                # since topk takes top even if all 0s, remove the 0 acts
                if torch.any(act[recruit_ind] == 0):
                    recruit_ind = recruit_ind[act[recruit_ind] != 0]

            # recruit n_mispredicted units
            self.MaskLayer.active_units.data[recruit_ind] = 1  # set ws to active
            self.MaskLayer.winning_units.data[:] = 0  # clear
            self.MaskLayer.winning_units.data[recruit_ind] = 1

            # keep units that predicted correctly
            if i > 0 and len(win_ind):  # else all lesioned
                self.MaskLayer.winning_units.data[win_ind[~mispred_units]] = 1

            self.Distance.weight.data[recruit_ind] = x  # place at curr stim            
              
    def _norm_diff(self, a, b):
        return a-b  # no norm

    def make_prediction(self, x):
        dim_dist = self.Distance(x)
        attn_dim_dist = self.DimensionWiseAttnLayer(dim_dist)
        act = self.ClusterActvLayer(attn_dim_dist)
        act_win, win_ind = self.MaskLayer(act)
        y_logits = self.ClsLayer(act_win)
        return y_logits, act, win_ind

    # def update_assoc(self, y_logits, y_true, x):
    #     self.optim.zero_grad()
    #     loss_value = self.loss_fn(y_logits, y_true)
    #     loss_value.backward(retain_graph=True)            # TODO: double check
    #     self.DimensionWiseAttnLayer.weight.grad[:] = 0    # TODO: double check
    #     self.optim.step()

    def update_units(self, x, win_ind):
        # update units - double update rule
        # - step 1 - winners update towards input
        update = (
            (x - self.Distance.weight.data[win_ind])
            * self.lr_clusters
        )

        # add noise to updates
        if self.noise:
            NotImplementedError()
        #     update += (
        #         torch.tensor(
        #             norm.rvs(loc=noise['update1'][0],
        #                         scale=noise['update1'][1],
        #                         size=(len(update), self.n_dims)))
        #         * self.params['lr_clusters']
        #             )

        self.Distance.weight.data[win_ind] += update

        # # store unit positions after both upds
        # self.units_pos_bothupd_trace.append(
        #     model.units_pos.detach().clone())

        # - step 2 - winners update towards self
        winner_mean = torch.mean(self.Distance.weight.data[win_ind], axis=0)
        update = (
            (winner_mean - self.Distance.weight.data[win_ind])
            * self.lr_clusters
        )

        # # add noise to 2nd update?
        # if noise:
        #     update += (
        #         torch.tensor(
        #             norm.rvs(loc=noise['update2'][0],
        #                         scale=noise['update2'][1],
        #                         size=(len(update), model.n_dims)))
        #         * model.params['lr_clusters_group']
        #             )

        self.Distance.weight.data[win_ind] += update

        # # save updated unit positions
        # model.units_pos_trace.append(model.units_pos.detach().clone())
        # model.units_pos_bothupd_trace.append(
        #     model.units_pos.detach().clone())  # store both upds

    def update_attn(self, act, win_ind):
        if self.attn_type[-5:] == 'local':
            win_ind = (self.MaskLayer.winning_units == 1)
            lose_ind = (win_ind == 0) & (self.MaskLayer.active_units == 1)

            act_1 = self._norm_diff(torch.sum(act[win_ind]), torch.sum(act[lose_ind]))
            # compute gradient
            act_1.backward(retain_graph=True)
            # divide grad by n active units (scales to any n_units)
            self.DimensionWiseAttnLayer.weight.data += (
                self.lr_attn
                * (self.DimensionWiseAttnLayer.weight.grad / self.MaskLayer.active_units.sum()))

        # ensure attention are non-negative
        attn_weights = self.DimensionWiseAttnLayer.weight.data
        attn_weights = torch.clamp(attn_weights, min=0.)
        # sum attention weights to 1
        self.DimensionWiseAttnLayer.weight.data = (attn_weights / torch.sum(attn_weights))
    
    def forward(self, x, epoch, i, y_true):
        """
        Forward pass logic (param update is now in train.py)
        This logic only includes forward evaluation and 
        optionally recruit units.
        """
        y_logits, act, win_ind = self.make_prediction(x)
        
        if self.need_recruit(y_logits, y_true):
            self.recruit_units(
                x=x, act=act, 
                epoch=epoch, i=i, 
                recruit=True, 
                win_ind=win_ind, 
                y_true=y_true
            )
            # once recruit, recompute activations
            y_logits, act, win_ind = self.make_prediction(x)
        
        return y_logits, act, win_ind