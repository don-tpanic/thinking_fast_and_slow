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
                problem_types=[1, 2, 3, 4, 5, 6]):
    """
    Follow sustain impl, we examine learning curves (y-axis is proberror)
    """
    config = load_config(config_version)
    num_runs = config['num_runs']
    num_blocks = config['num_blocks']
    model_types = ['total', 'fast', 'slow']
    colors = ['blue', 'orange', 'black', 'green', 'red', 'cyan']
    fig, axes = plt.subplots(3, 1)

    for subplot_idx in range(len(model_types)):
        model_type = model_types[subplot_idx]

        for idx in range(len(problem_types)):
            problem_type = problem_types[idx]

            lc_file = f'results/{config_version}/lc_{model_type}_type{problem_type}.npy'
            lc = np.load(lc_file)[:num_blocks]

            axes[subplot_idx].errorbar(
                range(lc.shape[0]), 
                lc, 
                color=colors[idx],
                label=f'Type {problem_type}',
            )        

        axes[subplot_idx].set_ylim(0, 0.7)
        axes[subplot_idx].set_ylabel('proberror')
        axes[-1].set_xlabel('epochs')
        axes[-1].legend()

        if model_type == 'fast':
            config_ = config['fast_config']
            lr_attn = config_['lr_clustering'] * config_['high_attn_lr_multiplier']
            lr_asso = config_['lr_clustering'] * config_['asso_lr_multiplier']
            lr_center = config_['lr_clustering'] * config_['center_lr_multiplier']
            axes[subplot_idx].set_title(f'{model_type}, lr attn: {lr_attn:.2f}, asso: {lr_asso:.2f}, center: {lr_center:.2f}')

        elif model_type == 'slow':
            config_ = config['slow_config']
            lr_dnn = config_['lr_dnn']
            axes[subplot_idx].set_title(f'{model_type}, lr dnn: {lr_dnn:.2f}')
    
    plt.tight_layout()
    plt.savefig(f'results/{config_version}/lc.png')
    return plt
            

def examine_loss(config_version, 
                problem_types=[1, 2, 3, 4, 5, 6]):
    """
    Instead of plotting proberror, we show raw loss.
    """
    config = load_config(config_version)
    num_runs = config['num_runs']
    num_blocks = config['num_blocks']
    model_types = ['total', 'fast', 'slow']
    colors = ['blue', 'orange', 'black', 'green', 'red', 'cyan']
    fig, axes = plt.subplots(3, 1)

    for subplot_idx in range(len(model_types)):
        model_type = model_types[subplot_idx]

        for idx in range(len(problem_types)):
            problem_type = problem_types[idx]

            lc_file = f'results/{config_version}/loss_{model_type}_type{problem_type}.npy'
            lc = np.load(lc_file)[:num_blocks]

            axes[subplot_idx].errorbar(
                range(lc.shape[0]), 
                lc, 
                color=colors[idx],
                label=f'Type {problem_type}',
            )
            axes[subplot_idx].set_title(f'{model_type}')
            axes[subplot_idx].set_ylabel('loss')
            axes[-1].set_xlabel('epochs')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{config_version}/loss.png')
    plt.close()


def examine_recruited_clusters_n_attn(config_version, canonical_runs_only=True):
    """
    Record the runs that produce canonical solutions
    for each problem type. 
    Specificially, we check the saved `mask_non_recruit`
    """
    config = load_config(config_version)
    num_runs=config['num_runs']
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
            mask_non_recruit = model_state_dict['ClusteringModel.MaskNonRecruit.weight']
            dim_wise_attn_weights = model_state_dict['ClusteringModel.DimensionWiseAttnLayer.weight']            
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
    config_version = 'config3'
    examine_lc(config_version)
    # examine_loss(config_version)
    # examine_recruited_clusters_n_attn(config_version)