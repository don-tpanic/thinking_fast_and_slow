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
            print('using default params')
            self.params = {
                # 'beta': 2,  # if cluster inhibition
                'phi': 1,  # .1 # response parameter, non-negative
                'lr_attn': 0.002,  # .001
                'lr_nn': 0.025,  # .01
                'lr_units': 0.002
                }

        # mask for active clusters
        self.active_units = torch.zeros(self.max_nunits, dtype=torch.bool)

        # set up by model type
        # if self.model_type == 'cluster':
        self.units = torch.nn.Parameter(
            torch.zeros([self.max_nunits, self.n_dims], dtype=torch.float))

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
        # print('x = ', x)
        # print('self.units = ', self.units[:2, :])
        # print('act = ', act)

        # mask with active clusters
        units_output = act * self.active_units

        # association weights
        out = self.fc1(units_output)   # logits

        # convert to response probability
        pr = self.softmax(self.params['phi'] * out)  # softmax( logits * phi )

        out = self.params['phi'] * out  # logits weighted like in mine

        return out, pr


def train_distr(model, inputs, output, n_epochs, shuffle=False):

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters(), 'lr': model.params['lr_nn']}
    p_attn = {'params': [model.attn], 'lr': model.params['lr_attn']}
    p_clusters = {'params': [model.units], 'lr': model.params['lr_units']}
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
    # print(inputs)
    # exit()
    np.random.seed(999)
    for epoch in range(n_epochs):
        if shuffle:
            # print(f'shuffled data for epoch {epoch}')
            # shuffle_ind = torch.randperm(len(inputs))
            # inputs_ = inputs[shuffle_ind]
            # output_ = output[shuffle_ind]
            shuffled_indices = np.random.permutation(len(inputs))
            inputs_ = inputs[shuffled_indices]
            output_ = output[shuffled_indices]
            # print(inputs_)
            # exit()
        else:
            inputs_ = inputs
            output_ = output

        i = 0
        for x, target in zip(inputs_, output_):
            # if i > 2:
            #     continue
            # print('\n\n\n')

            # learn
            optimizer.zero_grad()
            out, pr = model.forward(x)
            # loss = criterion(out.unsqueeze(0), target.unsqueeze(0))
            # print(out.unsqueeze(0), target.unsqueeze(0))
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

            # if target == 0:
            #     target = torch.tensor([1, 0], dtype=torch.float)
            # else:
            #     target = torch.tensor([0, 1], dtype=torch.float)
            item_proberror = 1. - torch.max(pr * t)
            lc[epoch] += item_proberror
            ct += 1

            # Recruit cluster, and update model
            if recruit and any(model.active_units == 0):  # in case recruit too many
                
                new_unit_ind = np.nonzero(model.active_units == 0)[0]

                model.active_units[new_unit_ind] = True
                model.units.data[new_unit_ind] = x  # place at curr stim
                # model.mask1[:, new_unit_ind] = True  # new clus weights
                # model.recruit_unit_trl.append(itrl)

                # go through update again after cluster added
                optimizer.zero_grad()
                out, pr = model.forward(x)
                if target == 0:
                    t = torch.tensor([1, 0], dtype=torch.float)
                else:
                    t = torch.tensor([0, 1], dtype=torch.float) 
                loss = criterion(out.unsqueeze(0), t.unsqueeze(0))
                # print('loss after recruit', loss)
                loss.backward()
                # with torch.no_grad():
                #     model.fc1.weight.grad.mul_(model.mask1)

                optimizer.step()

            # print(f'out = ', out, 'y_true = ', t, 'x = ', x)
            # print('loss = ', loss)
            # check grads 
            # for param in model.parameters():
            #     if param.shape == torch.Size([50, 2, 2]):
            #         pass
            #     else:
            #         print('grads = ', param.grad)
            itrl += 1

        # save epoch acc (itrl needs to be -1, since it was updated above)
        epoch_acc[epoch] = trial_acc[itrl-len(inputs):itrl].mean()
        epoch_ptarget[epoch] = trial_ptarget[itrl-len(inputs):itrl].mean()

    lc = lc / (1 * len(inputs_))
    print(lc)
    return model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget, trial_p


