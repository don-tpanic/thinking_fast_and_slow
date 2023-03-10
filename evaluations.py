import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import OrderedDict
import torch

import models
from utils import load_config
plt.rcParams.update({'font.size': 8})


def examine_lc(config_version):
    """
    Follow sustain impl, we examine learning curves (y-axis is proberror)
    """
    config = load_config(config_version)
    num_runs = config['num_runs']
    num_blocks = config['num_blocks']
    model_types = ['total', 'fast', 'slow']
    # colors = ['blue', 'orange', 'black', 'green', 'red', 'cyan']
    fig, axes = plt.subplots(3, 1)

    for subplot_idx in range(len(model_types)):
        model_type = model_types[subplot_idx]

        lc_file = f'results/{config_version}/lc_{model_type}.npy'
        lc = np.load(lc_file)[:num_blocks]

        axes[subplot_idx].errorbar(
            range(lc.shape[0]), 
            lc, 
            color='b',
        )        

        axes[subplot_idx].set_ylim(0, 0.7)
        axes[subplot_idx].set_ylabel('proberror')
        axes[-1].set_xlabel('epochs')
        axes[-1].legend()

        if model_type == 'fast':
            config_ = config['fast_config']

            if config['fast'] == 'clustering':
                lr_attn = config_['lr_clustering'] * config_['high_attn_lr_multiplier']
                lr_asso = config_['lr_clustering'] * config_['asso_lr_multiplier']
                lr_center = config_['lr_clustering'] * config_['center_lr_multiplier']
                axes[subplot_idx].set_title(f'{model_type}, lr attn: {lr_attn:.2f}, asso: {lr_asso:.2f}, center: {lr_center:.2f}')
            elif config['fast'] == 'multiunit_clustering':
                pass # TODO: what to track during sweep?
        
        elif model_type == 'slow':
            config_ = config['slow_config']
            lr_dnn = config_['lr_dnn']
            axes[subplot_idx].set_title(f'{model_type}, lr dnn: {lr_dnn:.2f}')
        
        else:
            axes[subplot_idx].set_title(f'{model_type}')
    
    plt.tight_layout()
    plt.savefig(f'results/{config_version}/lc.png')
    return plt
            

def examine_recruited_clusters_n_attn(config_version):
    """
    Check num of recruited clusters and attn weights
    """
    config = load_config(config_version)
    num_runs=config['num_runs']
    results_path = f'results/{config_version}'
    num_runs = config['num_runs']
    num_dims = 2
    all_runs_attn = np.empty((num_runs, num_dims))
    all_runs_num_recruited_clusters = np.empty(num_runs)

    for run in range(num_runs):            
        model_path = os.path.join(results_path, f'run{run}.pth.tar')
        model_state_dict = torch.load(model_path)['state_dict']
        mask_non_recruit = model_state_dict['FastModel.MaskNonRecruit.weight']
        dim_wise_attn_weights = model_state_dict['FastModel.DimensionWiseAttnLayer.weight']            
        all_runs_attn[run, :] = dim_wise_attn_weights
        num_recruited_clusters = len(np.nonzero(mask_non_recruit.detach().numpy()[0])[0])
        all_runs_num_recruited_clusters[run] = num_recruited_clusters

    print(all_runs_attn, '\n\n')
    print(all_runs_num_recruited_clusters)


if __name__ == '__main__':
    config_version = 'config_n1'
    examine_lc(config_version)
    examine_recruited_clusters_n_attn(config_version)