#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:07:02 2023

KDE (covariance matrix per category)

TODO - Multi-unit version?
- fine for 1 cov mat for all
- but pernode - might want one per recruited k units? else way too many. 

@author: robert.mok
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ExemplarModelKDE(nn.Module):
    def __init__(self, model_info, params=False, fitparams=False,
                 start_params=None):
        super(ExemplarModelKDE, self).__init__()
        self.model_type = model_info['model_type']
        self.attn_type = model_info['attn_type']
        self.max_nunits = model_info['max_nunits']
        self.n_dims = model_info['n_dims']
        self.nn_sizes = [model_info['max_nunits'], 2]  # only association ws
        self.softmax = nn.Softmax(dim=0)
        self.attn_trace = []
        self.units_act_trace = []
        self.recruit_unit_trl = []
        self.fc1_w_trace = []
        self.fc1_act_trace = []

        if params:
            self.params = params
        else:  # defaults
            self.params = {
                'r': 2,  # 1=city-block, 2=euclid
                'c': 6,
                'p': 1,  # p=1 exp, p=2 gauss
                'phi': 1,  # .1# response parameter, non-negative
                'lr_attn': .01,  # .001; .00025,  # . w/ mom=0-.1. 00005 w/ mom=.3-.5
                'lr_nn': .025,  # .01
                }

        # mask for active clusters
        self.active_units = torch.zeros(self.max_nunits, dtype=torch.bool)

        # exemplar - postions not trainable
        self.units = (
            torch.zeros([self.max_nunits, self.n_dims], dtype=torch.float)
            )

        # mask for NN (cluster acts in->output weights)
        self.mask1 = torch.zeros([self.nn_sizes[1], self.max_nunits],
                                 dtype=torch.bool)

        # for KDE
        # use cholesky decomposed covmat to learn, so start with it too
        attn = torch.cholesky(torch.eye(self.n_dims) * .02)
        # one cov mat for all stim
        if model_info['attn_type'] == 'single':
            self.attn = torch.nn.Parameter(attn)
        # if one cov mat per node
        elif model_info['attn_type'] == 'pernode':
            self.attn = torch.nn.Parameter(attn.repeat(self.max_nunits, 1, 1))

        self.fc1 = nn.Linear(self.max_nunits, self.nn_sizes[1], bias=False)
        # set weights to zero
        self.fc1.weight = (
            torch.nn.Parameter(torch.zeros([self.nn_sizes[1],
                                            self.max_nunits])))

    def forward(self, x):
        # compute activations here. stim x clusterpos x attn

        # looks like this works for single or pernode cov
        mvn1 = torch.distributions.MultivariateNormal(
            self.units, scale_tril=torch.tril(self.attn))
        act = torch.exp(mvn1.log_prob(x))

        # mask with active clusters
        units_output = act * self.active_units

        # association weights
        out = self.fc1(units_output)

        # convert to response probability
        pr = self.softmax(self.params['phi'] * out)

        # save acts, attn ws, assoc ws
        self.units_act_trace.append(units_output.detach().clone())
        self.attn_trace.append(self.attn.detach().clone())
        self.fc1_w_trace.append(self.fc1.weight.detach().clone())
        self.fc1_act_trace.append(out.detach().clone())

        return out, pr


def train_exemplar_kde(model, inputs, output, n_epochs, shuffle=False):

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters()}
    p_attn = {'params': [model.attn], 'lr': model.params['lr_attn']}
    params = [p_fc1, p_attn]

    criterion = nn.CrossEntropyLoss()  # loss

    optimizer = optim.SGD(params, lr=model.params['lr_nn'])  # , momentum=0.3)

    # save accuracy
    itrl = 0
    trial_acc = torch.zeros(len(inputs) * n_epochs)
    epoch_acc = torch.zeros(n_epochs)
    trial_ptarget = torch.zeros(len(inputs) * n_epochs)
    trial_p = torch.zeros([len(inputs) * n_epochs, len(torch.unique(output))])
    epoch_ptarget = torch.zeros(n_epochs)
    
    model.train()
    # torch.manual_seed(5)
    for epoch in range(n_epochs):
        if shuffle:
            shuffle_ind = torch.randperm(len(inputs))
            inputs_ = inputs[shuffle_ind]
            output_ = output[shuffle_ind]
        else:
            inputs_ = inputs
            output_ = output

        for x, target in zip(inputs_, output_):
            # testing
            # x=inputs_[itrl]
            # target=output_[itrl]

            # Forward pass (no updating for exemplar model)
            optimizer.zero_grad()
            out, pr = model.forward(x)

            # save acc per trial
            trial_acc[itrl] = torch.argmax(out.data) == target
            trial_ptarget[itrl] = pr[target]
            trial_p[itrl] = pr

            # Check if stimulus v similar any exemplar node, if not, recruit
            # - compute dist and define a threshold
            x_dists = abs(x - model.units[model.active_units])
            edists = torch.sum(x_dists**2, axis=1) ** (1/2)
            if torch.any(edists < 0.01):  # thresh; .01 v sim to 0, .05 liberal, 0.1 too high
                recruit = False
            else:
                recruit = True

            # Recruit
            if recruit and any(model.active_units == 0):  # in case recruit too many
                new_unit_ind = np.nonzero(model.active_units == 0)[0]
                model.active_units[new_unit_ind] = True
                model.units.data[new_unit_ind] = x  # place at curr stim
                model.mask1[:, new_unit_ind] = True  # new clus weights
                model.recruit_unit_trl.append(itrl)

            # update weights
            optimizer.zero_grad()
            out, pr = model.forward(x)
            loss = criterion(out.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            with torch.no_grad():
                model.fc1.weight.grad.mul_(model.mask1)

            optimizer.step()

            itrl += 1

        # save epoch acc (itrl needs to be -1, since it was updated above)
        epoch_acc[epoch] = trial_acc[itrl-len(inputs):itrl].mean()
        epoch_ptarget[epoch] = trial_ptarget[itrl-len(inputs):itrl].mean()

    return model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget, trial_p