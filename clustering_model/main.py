import os
import time

import numpy as np
import torch 
from torch import nn

import models
from train import fit
from utils import load_config, load_data
torch.autograd.set_detect_anomaly(True)

"""
Main executation script.
"""

def train_model(problem_type, config_version):
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
        print(f'= problem_type {problem_type} Run {run} ========================================')
        
        model = models.ClusteringModel(config=config)

        for epoch in range(num_blocks):
            # load and shuffle data
            dataset = load_data(problem_type)
            run2indices = np.load(f'run2indices_num_runs={num_runs}.npy')
            shuffled_indices = run2indices[run][epoch]
            shuffled_dataset = dataset[shuffled_indices]
            # print('[Check] shuffled_indices', shuffled_indices)
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
            f'type{problem_type}_run{run}.pth.tar')
        )
        del model
    
    assert num_runs * num_blocks * len(dataset) == ct, f'got incorrect ct = {ct}'
    lc = lc / (num_runs * len(dataset))
    np.save(os.path.join(results_path, f'lc_type{problem_type}.npy'), lc)


if __name__ == '__main__':
    start_time = time.time()
    num_processes = 6
    problem_types = [1, 2, 3, 4, 5, 6]
    config_version = 'config1'
    
    train_model(problem_type=1, config_version=config_version)
    import multiprocessing
    with multiprocessing.Pool(num_processes) as pool:
        for problem_type in problem_types:
            results = pool.apply_async(
                    train_model, 
                    args=[problem_type, config_version]
                )
        pool.close()
        pool.join()
        
    duration = time.time() - start_time
    print(f'duration = {duration}s')