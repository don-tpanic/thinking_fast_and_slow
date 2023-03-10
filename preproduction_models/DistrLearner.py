#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:15:22 2023

Distribution learner
- basic: learn mean and covariance matrix end-to-end, recruit after error
- basic + WTA (only winning distribution outputs and updates)


Next:
- recruitment: luo et al., 2023 style
- recruitment: threshold on loss (prob not that great)
- recruitment: threshold on mutual info - if new distr recruited will overlap
too much with current, don't recruit

- local learning: learn positions and/or covariance matrix by gradient ascent
to distribution activations

Later:
- competition across distributions (beta param)


@author: robert.mok
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DistrLearner(nn.Module):
    def __init__(self, model_info, params=False, fitparams=False,
                 start_params=None):
        super(DistrLearner, self).__init__()
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
                'r': 2,  # 1=city-block, 2=euclid
                'c': 6,
                'p': 1,  # p=1 exp, p=2 gauss
                # 'beta': 2,  # if cluster inhibition
                'phi': 1,  # .1 # response parameter, non-negative
                # 'a': .01,  # recruiting by loss thresh - TODO
                'lr_attn': .002,  # .001
                'lr_nn': .025,  # .01
                'lr_units': .005  # .01
                }

        # mask for active clusters
        self.active_units = torch.zeros(self.max_nunits, dtype=torch.bool)

        # set up by model type
        # if self.model_type == 'cluster':
        self.units = torch.nn.Parameter(
            torch.zeros([self.max_nunits, self.n_dims], dtype=torch.float))

        # elif self.model_type[0:15] == 'cluster_kohonen':
        #     self.units = torch.zeros([self.max_nunits, self.n_dims],
        #                                 dtype=torch.float)

        # mask for NN (cluster acts in->output weights)
        self.mask1 = torch.zeros([self.nn_sizes[1], self.max_nunits],
                                 dtype=torch.bool)

        # use cholesky decomposed covmat to learn, so start with it too
        attn = torch.cholesky(torch.eye(self.n_dims) * .02)  # .05
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

        # save cluster pos and acts, attn ws, assoc ws
        self.units_pos_trace.append(self.units.detach().clone())
        self.units_act_trace.append(units_output.detach().clone())
        self.attn_trace.append(self.attn.detach().clone())
        self.fc1_w_trace.append(self.fc1.weight.detach().clone())
        self.fc1_act_trace.append(out.detach().clone())

        return out, pr


def train_distr(model, inputs, output, n_epochs, shuffle=False):

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters()}
    p_attn = {'params': [model.attn], 'lr': model.params['lr_attn']}
    p_clusters = {'params': [model.units],
                  'lr': model.params['lr_units']}
    params = [p_fc1, p_clusters, p_attn]

    criterion = nn.CrossEntropyLoss()  # loss

    optimizer = optim.SGD(params, lr=model.params['lr_nn'], momentum=0.05)

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

            # learn
            optimizer.zero_grad()
            out, pr = model.forward(x)
            loss = criterion(out.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            # zero out gradient for masked connections
            with torch.no_grad():
                model.fc1.weight.grad.mul_(model.mask1)

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

            # Recruit cluster, and update model
            if recruit and any(model.active_units == 0):  # in case recruit too many
                new_unit_ind = np.nonzero(model.active_units == 0)[0]
                model.active_units[new_unit_ind] = True
                model.units.data[new_unit_ind] = x  # place at curr stim
                model.mask1[:, new_unit_ind] = True  # new clus weights
                model.recruit_unit_trl.append(itrl)

                # go through update again after cluster added
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


