#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:31:43 2023

Distribution learner
-multiunit version

@author: robert.mok
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DistrLearnerMU(nn.Module):
    def __init__(self, model_info, params=False, fitparams=False,
                 start_params=None):
        super(DistrLearnerMU, self).__init__()
        self.model_type = model_info['model_type']
        self.max_nunits = model_info['max_nunits']
        self.n_dims = model_info['n_dims']
        self.nn_sizes = [model_info['max_nunits'], 2]  # only association ws
        self.unit_recruit_mech = model_info['unit_recruit_type']
        self.softmax = nn.Softmax(dim=0)
        self.attn_trace = []
        self.units_pos_trace = []
        self.units_act_trace = []
        self.recruit_unit_trl = []
        self.fc1_w_trace = []
        self.fc1_act_trace = []

        if params:
            self.params = params
        else:  # defaults
            self.params = {
                'phi': 1,  # .1 # response parameter, non-negative
                'lr_attn': .002,  # .001
                'lr_nn': .025,  # .01
                'lr_units': .002, #.05, # .005,  # .01
                'k': .01
                }

        # mask for active clusters
        self.active_units = torch.zeros(self.max_nunits, dtype=torch.bool)

        # set up by model type
        # if self.model_type == 'cluster':
        self.units = torch.nn.Parameter(
            torch.zeros([self.max_nunits, self.n_dims], dtype=torch.float))

        # elif self.model_type[0:15] == 'cluster_kohonen':
        #     self.units = torch.rand([self.max_nunits, self.n_dims],
        #                             dtype=torch.float)

        # mask for NN (cluster acts in->output weights)
        # self.mask1 = torch.zeros([self.nn_sizes[1], self.max_nunits],
        #                          dtype=torch.bool)

        # use cholesky decomposed covmat to learn, so start with it too
        attn = torch.linalg.cholesky(torch.eye(self.n_dims) * .02)  # .05
        self.attn = torch.nn.Parameter(attn.repeat(self.max_nunits, 1, 1))

        self.fc1 = nn.Linear(self.max_nunits, self.nn_sizes[1], bias=False)
        # set weights to zero
        self.fc1.weight = (
            torch.nn.Parameter(torch.zeros([self.nn_sizes[1],
                                            self.max_nunits])))

    def forward(self, x):
        # compute activations of clusters here. stim x clusterpos x attn

        # mvn - can have multiple distributions here
        mvn1 = torch.distributions.MultivariateNormal(
            self.units, scale_tril=torch.tril(self.attn))

        act = torch.exp(mvn1.log_prob(x))

        # mask with active clusters
        units_output = act * self.active_units

        # association weights
        out = self.fc1(units_output)

        # convert to response probability
        pr = self.softmax(self.params['phi'] * out)

        out = self.params['phi'] * out

        return out, pr

def train_distr_mu(model, inputs, output, n_epochs, shuffle=False):

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters(), 'lr': model.params['lr_nn']}
    p_attn = {'params': [model.attn], 'lr': model.params['lr_attn']}
    p_clusters = {'params': [model.units],
                  'lr': model.params['lr_units']}
    params = [p_fc1, p_clusters, p_attn]

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(params, momentum=0)

    # save accuracy
    itrl = 0
    trial_acc = torch.zeros(len(inputs) * n_epochs)
    epoch_acc = torch.zeros(n_epochs)
    trial_ptarget = torch.zeros(len(inputs) * n_epochs)
    trial_p = torch.zeros([len(inputs) * n_epochs, len(torch.unique(output))])
    epoch_ptarget = torch.zeros(n_epochs)
    
    lc = np.zeros(n_epochs)
    ct = 0
    
    # model.train()
    np.random.seed(999)
    for epoch in range(n_epochs):
        if shuffle:
            # shuffle_ind = torch.randperm(len(inputs))
            # inputs_ = inputs[shuffle_ind]
            # output_ = output[shuffle_ind]
            shuffled_indices = np.random.permutation(len(inputs))
            inputs_ = inputs[shuffled_indices]
            output_ = output[shuffled_indices]
        else:
            inputs_ = inputs
            output_ = output

        i = 0
        for x, target in zip(inputs_, output_):
            # testing
            # x=inputs_[itrl]
            # target=output_[itrl]

            # learn
            optimizer.zero_grad()
            out, pr = model.forward(x)
            if target == 0:
                t = torch.tensor([1, 0], dtype=torch.float)
            else:
                t = torch.tensor([0, 1], dtype=torch.float)
            loss = criterion(out.unsqueeze(0), t.unsqueeze(0))
            i += 1
            loss.backward()
            # zero out gradient for masked connections
            # with torch.no_grad():
            #     model.fc1.weight.grad.mul_(model.mask1)

            # Recruitment
            # if incorrect, recruit
            if model.unit_recruit_mech == 'feedback':
                if ((not torch.argmax(out.data) == target) or
                    (torch.all(out.data == 0))):  # if incorrect
                    recruit = True
                else:
                    recruit = False

            # if not recruit, update model
            if recruit:
                pass
            else:
                optimizer.step()

            # save acc per trial
            trial_acc[itrl] = torch.argmax(out.data) == target
            trial_ptarget[itrl] = pr[target]
            trial_p[itrl] = pr

            item_proberror = 1. - torch.max(pr * t)
            lc[epoch] += item_proberror
            # if epoch == 0:
            #     print('item_proberror', item_proberror, 'x', x, 'target', target)
            # else:
            #     exit()
            ct += 1

            # Recruit cluster, and update model
            if recruit and any(model.active_units == 0):  # in case recruit too many
                # select closest k inactive units
                # - cant select mispred units only as not WTA, all involved
                mvn1 = torch.distributions.MultivariateNormal(
                    model.units, scale_tril=torch.tril(model.attn))
                act = torch.exp(mvn1.log_prob(x))

                act[model.active_units] = 0  # REMOVE all active units
                # find closest units excluding the active units to recruit
                _, recruit_ind = (
                    torch.topk(act, int(model.max_nunits * model.params['k'])))
                # since topk takes top even if all 0s, remove the 0 acts
                if torch.any(act[recruit_ind] == 0):
                    recruit_ind = recruit_ind[act[recruit_ind] != 0]

                # recruit n_units
                model.active_units[recruit_ind] = True  # set ws to active
                with torch.no_grad():
                    model.units[recruit_ind] = x  # place at curr stim
                # model.mask1[:, recruit_ind] = True  # new clus weights
                # model.recruit_unit_trl.append(itrl)


                # go through update again after cluster added
                optimizer.zero_grad()
                out, pr = model.forward(x)
                if target == 0:
                    t = torch.tensor([1, 0], dtype=torch.float)
                else:
                    t = torch.tensor([0, 1], dtype=torch.float) 
                loss = criterion(out.unsqueeze(0), t.unsqueeze(0))
                loss.backward()
                # with torch.no_grad():
                #     model.fc1.weight.grad.mul_(model.mask1)

                # print(loss)
                # exit()
                optimizer.step()

            itrl += 1

        # save epoch acc (itrl needs to be -1, since it was updated above)
        epoch_acc[epoch] = trial_acc[itrl-len(inputs):itrl].mean()
        epoch_ptarget[epoch] = trial_ptarget[itrl-len(inputs):itrl].mean()

    lc = lc / (1 * len(inputs_))
    print(lc)
    return model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget, trial_p


