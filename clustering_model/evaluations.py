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
    num_runs = config['num_runs']
    num_blocks = config['num_blocks']

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
            
            
def examine_recruited_clusters_n_attn(config_version, canonical_runs_only=True):
    """
    Record the runs that produce canonical solutions
    for each problem type. 
    Specificially, we check the saved `mask_non_recruit`
    """
    config = load_config(config_version)
    in_features=config['in_features']
    out_features=config['out_features']
    r=config['r']
    q=config['q']
    specificity=config['specificity']
    high_attn_constraint=config['high_attn_constraint']
    high_attn_regularizer=config['high_attn_regularizer']
    high_attn_reg_strength=config['high_attn_reg_strength']
    beta=config['beta']
    temp_competition=config['temp_competition']
    temp_softwta=config['temp_softwta']
    Phi=config['Phi']
    lr=config['lr_clustering']
    asso_lr_multiplier=config['asso_lr_multiplier']
    center_lr_multiplier=config['center_lr_multiplier']
    high_attn_lr_multiplier=config['high_attn_lr_multiplier']
    random_seed=config['random_seed']
    num_blocks=config['num_blocks']
    num_clusters=config['num_clusters']
    num_runs=config['num_runs']
    thr=config['thr']
    results_path = f'results/{config_version}'

    num_types = 6
    results_path = f'results/{config_version}'
    num_runs = config['num_runs']
    num_dims = 3
    type2cluster = {
        1: 2, 2: 4,
        3: 6, 4: 6, 5: 6,
        6: 8}

    from collections import defaultdict
    canonical_runs = defaultdict(list)

    attn_weights = []
    for z in range(num_types):
        problem_type = z + 1
        
        all_runs_attn = np.empty((num_runs, num_dims))
        for run in range(num_runs):            
            model_path = os.path.join(results_path, f'type{problem_type}_run{run}.pth.tar')
            model_state_dict = torch.load(model_path)['state_dict']
            mask_non_recruit = model_state_dict['MaskNonRecruit.weight']
            dim_wise_attn_weights = model_state_dict['DimensionWiseAttnLayer.weight']            
            all_runs_attn[run, :] = dim_wise_attn_weights

            num_nonzero = len(np.nonzero(mask_non_recruit.detach().numpy()[0])[0])
            if canonical_runs_only:
                if num_nonzero == type2cluster[problem_type]:
                    canonical_runs[z].append(run)
            else:
                canonical_runs[z].append(run)

        per_type_attn_weights = np.round(
            np.mean(all_runs_attn, axis=0), 3
        )
        attn_weights.append(per_type_attn_weights)

    proportions = []
    for z in range(num_types):
        print(f'Type {z+1}, has {len(canonical_runs[z])}/{num_runs} canonical solutions')
        proportions.append(
            np.round(
                len(canonical_runs[z]) / num_runs 
            )
        )
    
    print(attn_weights)
    return proportions, attn_weights


if __name__ == '__main__':
    examine_lc(config_version='config1')
    examine_recruited_clusters_n_attn(config_version='config1')