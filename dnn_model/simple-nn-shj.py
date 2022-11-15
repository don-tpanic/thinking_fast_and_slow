#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:17:32 2022


@author: robert.mok
"""
# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
sys.path.append(
    '/Users/robert.mok/Documents/GitHub/multilayer-shj-transfer')

# from NeuralNetworkSimple import NeuralNetwork
from models import NeuralNetwork

resdir = os.path.join(maindir, 'shj-transfer-nn')

# gpu if available
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# %% set up stimuli and models

# =============================================================================
# shepard's problems (stimuli and category (last column))
# =============================================================================

six_problems = [[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0],
                 [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 1],
                 [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
                 [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
                 [1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
                 [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],

                [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0],
                 [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],

                ]

fig, ax = plt.subplots()

for problem in [0, 1, 2, 3, 4, 5]:
    # problem = 1  # 0-5
    print('problem', problem)
    stim = six_problems[problem]
    stim = torch.tensor(stim, dtype=torch.float)
    inputs = stim[:, 0:-1]
    output = stim[:, -1].long()  # integer
    input_size = inputs.shape[1]
    output_size = len(output)

    # %%
    savesims = False

    n_sims = 50  # number of sims

    n_epochs = 350

    # =============================================================================
    # set up model params
    # =============================================================================
    act_func = nn.ReLU()  # nn.Identity() or nn.ReLU()

    bias = True  # False
    lr = .0075
    n_units = [25]  # [25, 50]  # can have multiple but not sure if that works now
    mom = 0.3
    dropout = 0

    # organize network sizes - no. of hidden units, no. of layers
    networks_depth = 3

    # equal number of units
    hidden_size_conds = [[i_units] * networks_depth for i_units in n_units]

    networks_shapes = []
    for inet in range(len(hidden_size_conds)):
        if networks_depth == 1:  # 1 hidden layer
            networks_shapes.append([input_size, hidden_size_conds[inet][0],
                                    output_size])
        elif networks_depth == 2:  # 2 hidden layers
            networks_shapes.append([input_size,
                                    hidden_size_conds[inet][0],
                                    hidden_size_conds[inet][1], output_size])
        elif networks_depth == 3:  # 3 hidden layers
            networks_shapes.append([input_size,
                                    hidden_size_conds[inet][0],
                                    hidden_size_conds[inet][1],
                                    hidden_size_conds[inet][2], output_size])

    criterion = nn.CrossEntropyLoss()

    print('networks_shapes', networks_shapes)

    # =============================================================================
    # set up model, tasks
    # =============================================================================
    for inet in range(len(networks_shapes)):

        print('len(networks_shapes)', len(networks_shapes))   # len == 1

        n_trials = len(inputs) * n_epochs
        trial_acc = torch.zeros([n_trials, n_sims])
        epoch_acc = np.zeros([n_epochs, n_sims])
        loss_trial = torch.zeros([n_trials, n_sims])
        loss_train = torch.zeros([n_epochs, n_sims])

        # trained_models = {}  # initalize dict to save models
        problem_seq = []  # if shuffle, save the sequences

        for isim in range(n_sims):

            # networks_shapes[inet] = [input_size, 25, 25, 25, output_size]

            model = NeuralNetwork(networks_shapes[inet], act_func=act_func,
                                device=device).to(device)
    

            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
            model.train()

    # =============================================================================
    # train
    # =============================================================================
            eval_acts = []  # temp - append this to eval_acts_all after all sets

            itrl = 0
            for epoch in range(n_epochs):

                # shuffle trials
                shuffle_ind = torch.randperm(len(inputs))
                input_ = inputs[shuffle_ind].to(device)
                output_ = output[shuffle_ind].to(device)  # ordered alr

                print(f'output_', output_)

                model.train()
                for x, target in zip(input_, output_):

                    optimizer.zero_grad()

                    out = model(x.unsqueeze(0))
                    print('x.unsqueeze(0)', x.unsqueeze(0))
                    
                    loss = criterion(out, target.unsqueeze(0))

                    # save loss
                    loss_trial[itrl, isim] = loss.detach().clone()

                    # backprop
                    loss.backward()

                    optimizer.step()

                    # save trial-wise acc
                    trial_acc[itrl, isim] = (
                        torch.argmax(out.data) == target
                        )

                    itrl += 1

                # save accuracy during training
                model.eval()  # set so doesn't save trace
                with torch.no_grad():
                    pred_out = model(inputs.float().to(device)).to(device)

                epoch_acc[epoch, isim] = (
                    torch.sum(torch.argmax(pred_out.data, dim=1)
                            == output.to(device))
                    / len(pred_out))

                # compute epoch loss
                loss_train[epoch, isim] = (
                    loss_trial[itrl-8:itrl, isim].mean()
                    )

            # # save layer-wise activations after training
            # model.eval()  # set so doesn't save trace
            # model.eval_acts = []  # clear before predict to save acts
            # with torch.no_grad():
            #     pred_out = model(inputs.float().to(device)).to(device)
            # # save over sets - copy else it keeps updating
            # eval_acts.append(model.eval_acts.copy())

        # # save model
        # model.to('cpu')
        # trained_models['sim{}_model'.format(isim)] = (
        #     copy.deepcopy(model.state_dict())
        #     )


    # # plot to check
    # plt.plot(epoch_acc.mean(axis=-1))  # average over sims
    # plt.ylim([0, 1.05])
    # # plt.show()
    # plt.savefig('fig1.png')

    # plt.plot(loss_trial.mean(axis=-1))
    # # plt.show()
    # plt.savefig('fig2.png')

    # plt.plot(loss_train.mean(axis=-1))
    # # plt.show()
    # plt.savefig('fig3.png')
    ax.plot(epoch_acc.mean(axis=-1), label=f'problem {problem}')

plt.legend()
plt.savefig('fig.png')