import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import OrderedDict
import torch

import models
from utils import load_config
plt.rcParams.update({'font.size': 8})


def examine_lc(config_version, 
                problem_types=[1, 2, 3, 4, 5, 6], 
                plot_learn_curves=True):
    """
    Follow sustain impl, we examine learning curves (y-axis is proberror)

    return:
    -------
        trapz_areas: An array of scalars which are the areas under the learning curves computed
                     using the trapzoidal rule.
        figure (optional): If `plot_learn_curves=True` will plot the learning curves.
    """
    config = load_config(config_version)
    num_runs = config['n_sims']
    num_blocks = config['n_epochs']

    colors = ['blue', 'orange', 'black', 'green', 'red', 'cyan']
    if plot_learn_curves:
        fig, ax = plt.subplots()

    trapz_areas = np.empty(len(problem_types))
    for idx in range(len(problem_types)):
        problem_type = problem_types[idx]

        lc_file = f'results/{config_version}/lc_type{problem_type}.npy'
        lc = np.load(lc_file)[:num_blocks]

        trapz_areas[idx] = np.round(np.trapz(lc), 3)
        if plot_learn_curves:
            ax.errorbar(
                range(lc.shape[0]), 
                lc, 
                color=colors[idx],
                label=f'Type {problem_type}',
            )
    
    print(f'[Results] {config_version} trapzoidal areas = ', trapz_areas)
    if plot_learn_curves:
        plt.legend()
        plt.title(f'{trapz_areas}')
        plt.xlabel('epochs')
        plt.ylabel('average probability of error')
        plt.tight_layout()
        plt.savefig(f'results/{config_version}/lc.png')
        plt.close()
    return trapz_areas
            

if __name__ == '__main__':
    examine_lc(config_version='config5')