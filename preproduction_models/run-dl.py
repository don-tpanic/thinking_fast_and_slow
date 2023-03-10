#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:34:16 2023

@author: robert.mok
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import itertools as it

sys.path.append('/Users/robert.mok/Documents/GitHub/distribution-learner')

from DistrLearner import (DistrLearner, train_distr)
from ExemplarModel import (ExemplarModel, train_exemplar)

from DistrLearnerMU import (DistrLearnerMU, train_distr_mu)
from ExemplarModelMU import (ExemplarModelMU, train_exemplar_mu)

from DistrLearnerWta import (DistrLearnerWta, train_distr_wta)
from DistrLearnerWtaMU import (DistrLearnerWtaMU, train_distr_wta_mu)

from ExemplarModelKDE import (ExemplarModelKDE, train_exemplar_kde)

# paths
maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'  # macbook
# wd = os.path.join(maindir, 'typicality-web/')

# %% stim distribution

np.random.seed(111)

# 0-1
mu1 = [.35, .7]
var1 = [.009, .006]
cov1 = .004
mu2 = [.65, .5]
var2 = [.009, .006]
cov2 = -.004

x, y = np.mgrid[0:1:.01, 0:1:.01]
pos = np.dstack((x, y))
rv1 = multivariate_normal([mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]])
rv2 = multivariate_normal([mu2[0], mu2[1]], [[var2[0], cov2], [cov2, var2[1]]])

# % sampling
npoints = 50
x1 = np.random.multivariate_normal(
    [mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]], npoints)
x2 = np.random.multivariate_normal(
    [mu2[0], mu2[1]], [[var2[0], cov2], [cov2, var2[1]]], npoints)

# round
x1 = np.around(x1, decimals=2)
x2 = np.around(x2, decimals=2)
fig, ax = plt.subplots(1, 1)
ax.contour(x, y, rv1.pdf(pos), cmap='Blues')
ax.scatter(x1[:, 0], x1[:, 1], c='blue', s=1)
ax.contour(x, y, rv2.pdf(pos), cmap='Greens')
ax.scatter(x2[:, 0], x2[:, 1], c='green', s=1)
ax.set_box_aspect(1)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_facecolor((.5, .5, .5))
plt.show()

shuffle = False  # set False within model

# shuffle here
inputs = torch.cat([torch.tensor(x1, dtype=torch.float32),
                    torch.tensor(x2, dtype=torch.float32)])
output = torch.cat([torch.zeros(npoints, dtype=torch.long),
                    torch.ones(npoints, dtype=torch.long)])

shuffle_ind = torch.randperm(len(inputs))
inputs = inputs[shuffle_ind]
output = output[shuffle_ind]

# %% run

saveplots = False

model_type = 'cluster'

# trials, etc.
n_epochs = 10

# distribution learner
unit_recruit_type = 'feedback'
model_info = {
    'model_type': model_type,
    'max_nunits': 50,  # max nclusters
    'n_dims': inputs.shape[1],
    'unit_recruit_type': unit_recruit_type,
    }

model_distr = DistrLearner(model_info)
model_distr, epoch_acc, trial_acc, epoch_ptarget_distr, trial_ptarget, trial_p = (
    train_distr(model_distr, inputs, output, n_epochs, shuffle=shuffle)
    )

# distribution learner MU
unit_recruit_type = 'feedback'
model_info = {
    'model_type': model_type,
    'max_nunits': 500,  # max nclusters
    'n_dims': inputs.shape[1],
    'unit_recruit_type': unit_recruit_type,
    'k': .01
    }

model_distr_mu = DistrLearnerMU(model_info)
model_distr_mu, epoch_acc, trial_acc, epoch_ptarget_distr, trial_ptarget, trial_p = (
    train_distr_mu(model_distr_mu, inputs, output, n_epochs, shuffle=shuffle)
    )

# distribution learner wta
unit_recruit_type = 'feedback'
model_info = {
    'model_type': model_type,
    'max_nunits': 50,  # max nclusters
    'n_dims': inputs.shape[1],
    'unit_recruit_type': unit_recruit_type,
    }

model_distr_wta = DistrLearnerWta(model_info)
model_distr_wta, epoch_acc, trial_acc, epoch_ptarget_distr, trial_ptarget, trial_p = (
    train_distr_wta(model_distr_wta, inputs, output, n_epochs, shuffle=shuffle)
    )

# distribution learner k-WTA MU
# - atm most unstable (recruits many), not sure why; might have a bug
unit_recruit_type = 'feedback'
model_info = {
    'model_type': model_type,
    'max_nunits': 500,  # max nclusters
    'n_dims': inputs.shape[1],
    'unit_recruit_type': unit_recruit_type,
    'k': .01
    }
model_distr_wta_mu = DistrLearnerWtaMU(model_info)
model_distr_wta_mu, epoch_acc, trial_acc, epoch_ptarget_distr, trial_ptarget, trial_p = (
    train_distr_wta_mu(model_distr_wta_mu, inputs, output, n_epochs, shuffle=shuffle)
    )


# exemplar model
# -  'attn_type': 'none', 'dim'
model_info = {
    'model_type': model_type,
    'max_nunits': 500,  # max nclusters
    'n_dims': inputs.shape[1],
    'attn_type': 'dim'  # none, dim
    }

model_ex = ExemplarModel(model_info)
model_ex, epoch_acc, trial_acc, epoch_ptarget_ex, trial_ptarget, trial_p = (
    train_exemplar(model_ex, inputs, output, n_epochs, shuffle=shuffle)
    )

# exemplar model MU
model_info = {
    'model_type': model_type,
    'max_nunits': 500,  # max nclusters
    'n_dims': inputs.shape[1],
    'attn_type': 'dim',  # none, dim
    'k': .01
    }

model_ex_mu = ExemplarModelMU(model_info)
model_ex_mu, epoch_acc, trial_acc, epoch_ptarget_ex, trial_ptarget, trial_p = (
    train_exemplar_mu(model_ex_mu, inputs, output, n_epochs, shuffle=shuffle)
    )

# exemplar model KDE
model_info = {
    'model_type': model_type,
    'max_nunits': 500,  # max nclusters
    'n_dims': inputs.shape[1],
    'attn_type': 'pernode'  # 'single' cov for all , or 'pernode': one per node
    }

model_kde = ExemplarModelKDE(model_info)
model_kde, epoch_acc, trial_acc, epoch_ptarget_ex, trial_ptarget, trial_p = (
    train_exemplar_kde(model_kde, inputs, output, n_epochs, shuffle=shuffle)
    )

# %% plot a bit

# print(epoch_acc)
# print(epoch_ptarget_distr)
# plt.plot(1 - epoch_ptarget_distr.detach())
# plt.show()

# print(epoch_acc)
# print(epoch_ptarget_ex)
# plt.plot(1 - epoch_ptarget_ex.detach())
# plt.show()

# plt.plot(torch.stack(model.units_act_trace, dim=0))
# # plt.plot(torch.log(torch.stack(model.units_act_trace, dim=0)))
# plt.show()

# plt.plot(torch.stack(model_distr.units_pos_trace)[:,0,0])
# plt.show()

plt.plot(torch.stack(model_kde.attn_trace)[:, 0])
plt.plot(torch.stack(model_kde.attn_trace)[:, 1])
plt.show()


# %% plot data and results (distr learner)
clw = .65  # contour linewidths
ca = .7  # contour alpha
sa = .3  # scatter alpha

fig, ax = plt.subplots(1, 2, dpi=300)

# input
rv1 = multivariate_normal([mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]])
rv2 = multivariate_normal([mu2[0], mu2[1]], [[var2[0], cov2], [cov2, var2[1]]])
ax[0].contour(x, y, rv1.pdf(pos), cmap='Blues', alpha=ca, zorder=-1)
ax[0].contour(x, y, rv2.pdf(pos), cmap='Greens', alpha=ca, zorder=-1)
ax[0].scatter(x1[:, 0], x1[:, 1], c='blue', s=.5, alpha=sa)
ax[0].scatter(x2[:, 0], x2[:, 1], c='green', s=.5, alpha=sa)
ax[0].set_facecolor((.65, .65, .65))
ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])
ax[0].set_box_aspect(1)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Underlying Concept'
                '\n'
                ' Distributions')

# results
cov_res = model_distr.attn.detach().clone()
cov_res = torch.bmm(cov_res, torch.transpose(cov_res, 1, 2))  # recover covmat
rv1 = multivariate_normal(model_distr.units[0].detach(), cov_res[0])
rv2 = multivariate_normal(model_distr.units[1].detach(), cov_res[1])
ax[1].contour(x, y, rv1.pdf(pos), cmap='Greys', alpha=ca, zorder=-1)
ax[1].contour(x, y, rv2.pdf(pos), cmap='Greys', alpha=ca, zorder=-1)
ax[1].scatter(x1[:, 0], x1[:, 1], c='blue', s=.5, alpha=sa)
ax[1].scatter(x2[:, 0], x2[:, 1], c='green', s=.5, alpha=sa)
ax[1].set_facecolor((.65, .65, .65))
ax[1].set_xlim([0, 1])
ax[1].set_ylim([0, 1])
ax[1].set_box_aspect(1)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Model Learnt'
                '\n'
                'Distributions')

# %% plot categorization

# - takes time - plotting a large grid - can make faster probs

grid = torch.arange(0,1.01,.01)

pr_distr = torch.zeros(len(grid), len(grid))
pr_ex = torch.zeros(len(grid), len(grid))

for i, j in it.product(range(len(grid)), range(len(grid))):

    out, pr = model_distr.forward(torch.tensor([grid[i], grid[j]]))
    pr_distr[i, j] = pr[0]
    
    out, pr = model_ex.forward(torch.tensor([grid[i], grid[j]]))
    pr_ex[i, j] = pr[0]


vmin = [.4, .6]
plt.imshow(torch.rot90(pr_distr.detach(), 1, [0, 1]), vmin=vmin[0], vmax=vmin[1])
plt.colorbar()
plt.show()

plt.imshow(torch.rot90(pr_ex.detach(), 1, [0, 1]), vmin=vmin[0], vmax=vmin[1])
plt.colorbar()
plt.show()

plt.imshow(torch.rot90(pr_distr.detach()-pr_ex.detach(), 1, [0, 1]))
plt.colorbar()
plt.show()
