import os
import time
import numpy as np
import multiprocessing
import torch 
from torch import nn

import models
from train import fit
from evaluations import examine_lc
from utils import load_config, load_data
torch.autograd.set_detect_anomaly(True)

import wandb

"""
Main execution script.
"""

def train_model(problem_type, config_version):
    config = load_config(config_version)
    print(f'[Check] config: {config_version}')
    random_seed=config['random_seed']
    num_blocks=config['num_blocks']
    num_runs=config['num_runs']
    results_path = f'results/{config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.random.seed(random_seed)

    lc_fast = np.zeros(num_blocks)
    lc_slow = np.zeros(num_blocks)
    lc_total = np.zeros(num_blocks)
    loss_fast = np.zeros(num_blocks)
    loss_slow = np.zeros(num_blocks)
    loss_total = np.zeros(num_blocks)
    ct = 0
    for run in range(num_runs):
        print(f'= problem_type {problem_type} Run {run} ========================================')
        
        model = models.FastSlow(config=config)

        for epoch in range(num_blocks):
            dataset = load_data(problem_type)
            shuffled_indices = np.random.permutation(len(dataset))
            shuffled_dataset = dataset[shuffled_indices]

            for i in range(len(shuffled_dataset)):
                dp = shuffled_dataset[i]
                x = torch.Tensor(dp[0])
                y_true = torch.Tensor(dp[1])
                signature = dp[2]
                model, \
                    item_proberror_fast, item_proberror_slow, item_proberror_total, \
                    loss_value_fast, loss_value_slow, loss_value = \
                        fit(
                            model=model, 
                            x=x,
                            y_true=y_true,
                            signature=signature,
                            epoch=epoch,
                            i=i,
                        )
                lc_fast[epoch] += item_proberror_fast
                lc_slow[epoch] += item_proberror_slow
                lc_total[epoch] += item_proberror_total
                loss_fast[epoch] += loss_value_fast
                loss_slow[epoch] += loss_value_slow
                loss_total[epoch] += loss_value
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
    lc_fast = lc_fast / (num_runs * len(dataset))
    lc_slow = lc_slow / (num_runs * len(dataset))
    lc_total = lc_total / (num_runs * len(dataset))
    loss_fast = loss_fast / (num_runs * len(dataset))
    loss_slow = loss_slow / (num_runs * len(dataset))
    loss_total = loss_total / (num_runs * len(dataset))
    np.save(os.path.join(results_path, f'lc_fast_type{problem_type}.npy'), lc_fast)
    np.save(os.path.join(results_path, f'lc_slow_type{problem_type}.npy'), lc_slow)
    np.save(os.path.join(results_path, f'lc_total_type{problem_type}.npy'), lc_total)
    np.save(os.path.join(results_path, f'loss_fast_type{problem_type}.npy'), loss_fast)
    np.save(os.path.join(results_path, f'loss_slow_type{problem_type}.npy'), loss_slow)
    np.save(os.path.join(results_path, f'loss_total_type{problem_type}.npy'), loss_total)


def train_model_across_types(config_version):
    config = load_config(config_version)
    if not disable_wandb:
        run = wandb.init(
            project="thinking_fast_and_slow",
            entity="robandken",
            config=config,
            reinit=True,
        )
        wandb.run.name = f'{config_version}'

    for problem_type in problem_types:
        train_model(problem_type, config_version)
    
    # log results to wandb
    log_results(config_version)
    run.finish()


def log_results(config_version):
    print(f'[Check] logging results..')
    # will save locally.
    plt = examine_lc(config_version)

    # log figures to wandb
    wandb.log({"lc": plt})
    print(f'[Check] done logging results.')


if __name__ == '__main__':
    start_time = time.time()
    problem_types = [6]
    disable_wandb = False
    single_config = False
    multiple_configs = True

    if single_config:
        num_processes = 6
        config_version = 'config6'
        config = load_config(config_version)

        if not disable_wandb:
            wandb.init(
                project="thinking_fast_and_slow",
                entity="robandken",
                config=config,
            )
            wandb.run.name = f'{config_version}'

        import multiprocessing
        with multiprocessing.Pool(num_processes) as pool:
            for problem_type in problem_types:
                results = pool.apply_async(
                        train_model, 
                        args=[problem_type, config_version]
                    )
            print(results.get())
            pool.close()
            pool.join()
        
        # log results to wandb
        if not disable_wandb:
            log_results(config_version)

    elif multiple_configs:
        # one process is one config over 6 types
        num_processes = 20
        # config_versions = [f'config{i}' for i in range(6, 23)]
        config_versions = ['config13']
        with multiprocessing.Pool(num_processes) as pool:
            for config_version in config_versions:
                results = pool.apply_async(
                        train_model_across_types, 
                        args=[config_version]
                    )

            pool.close()
            pool.join()
        
    duration = time.time() - start_time
    print(f'duration = {duration}s')