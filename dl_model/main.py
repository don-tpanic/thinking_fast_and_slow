import os
import time

import numpy as np
import torch 
from torch import nn

import models
from train import fit
from utils import load_config, load_new_stimuli
torch.autograd.set_detect_anomaly(True)

"""
Main executation script.
"""

def train_model(config_version):
    config = load_config(config_version)
    random_seed=config['random_seed']
    num_blocks=config['num_blocks']
    num_runs=config['num_runs']
    results_path = f'results/{config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.random.seed(random_seed)
    # --------------------------------------------------------------------------
    lc = np.zeros(num_blocks)
    ct = 0
    for run in range(num_runs):
        
        model = models.DistrLearner(config=config)

        for epoch in range(num_blocks):
            # load and shuffle data
            dataset = load_new_stimuli()
            shuffled_indices = np.random.permutation(len(dataset))
            shuffled_dataset = dataset[shuffled_indices]
            # each epoch trains on all items
            for i in range(len(shuffled_dataset)):
                dp = shuffled_dataset[i]
                x = torch.Tensor(dp[0])
                y_true = torch.Tensor(dp[1])
                signature = dp[2]
                model, item_proberror = fit(
                    model=model,
                    x=x,
                    y_true=y_true,
                    signature=signature,
                    epoch=epoch, 
                    i=i,
                )
                lc[epoch] += item_proberror
                ct += 1
        
        # save run-level model per problem type
        ckpt_data = {}
        ckpt_data['state_dict'] = model.state_dict()
        torch.save(
            ckpt_data, 
            os.path.join(results_path,
            f'run{run}.pth.tar')
        )
        del model
    
    assert num_runs * num_blocks * len(dataset) == ct, f'got incorrect ct = {ct}'
    lc = lc / (num_runs * len(dataset))
    print(lc)
    np.save(os.path.join(results_path, f'lc.npy'), lc)


if __name__ == '__main__':
    start_time = time.time()
    config_version = 'config1'

    train_model(config_version=config_version)

    duration = time.time() - start_time
    print(f'duration = {duration}s')