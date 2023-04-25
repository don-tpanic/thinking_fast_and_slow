import os
import time
import argparse
import numpy as np
import multiprocessing
import torch 
from torch import nn

import models
from train import fit
from evaluations import examine_lc
from utils import load_config, str2bool, load_new_stimuli
torch.autograd.set_detect_anomaly(True)

import wandb

"""
Main execution script.
"""

def train_model(config_version):

    torch.set_num_threads(1)

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

        model = models.FastSlow(config=config)

        for epoch in range(num_blocks):
            print(f'[Check] run {run}, epoch {epoch}')
            dataset = load_new_stimuli()
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
        
        # save run-level model task
        ckpt_data = {}
        ckpt_data['state_dict'] = model.state_dict()
        torch.save(
            ckpt_data, 
            os.path.join(results_path,
            f'run{run}.pth.tar')
        )
        del model
    
    assert num_runs * num_blocks * len(dataset) == ct, f'got incorrect ct = {ct}'
    lc_fast = lc_fast / (num_runs * len(dataset))
    lc_slow = lc_slow / (num_runs * len(dataset))
    lc_total = lc_total / (num_runs * len(dataset))
    loss_fast = loss_fast / (num_runs * len(dataset))
    loss_slow = loss_slow / (num_runs * len(dataset))
    loss_total = loss_total / (num_runs * len(dataset))
    np.save(os.path.join(results_path, f'lc_fast.npy'), lc_fast)
    np.save(os.path.join(results_path, f'lc_slow.npy'), lc_slow)
    np.save(os.path.join(results_path, f'lc_total.npy'), lc_total)
    np.save(os.path.join(results_path, f'loss_fast.npy'), loss_fast)
    np.save(os.path.join(results_path, f'loss_slow.npy'), loss_slow)
    np.save(os.path.join(results_path, f'loss_total.npy'), loss_total)


def log_results(config_version):
    print(f'[Check] logging results..')
    # will save locally.
    plt = examine_lc(config_version)

    # log figures to wandb
    wandb.log({"lc": plt})
    print(f'[Check] done logging results.')


def train_model_multiproc(config_version):
    config = load_config(config_version)
    if args.logging:
        run = wandb.init(
            project="thinking_fast_and_slow",
            entity="robandken",
            config=config,
            reinit=True,
        )
        wandb.run.name = f'{config_version}'

    train_model(config_version)
    
    # log results to wandb
    log_results(config_version)
    run.finish()


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging', dest='logging', default=True, type=str2bool)
    parser.add_argument('-c', '--config', dest='config', type=str, default=None)
    parser.add_argument('-b', '--begin', dest='begin', type=int, default=None)
    parser.add_argument('-e', '--end', dest='end', type=int, default=None)
    args = parser.parse_args()

    # just run a single config
    if args.config:
        config_version = args.config
        config = load_config(config_version)
        if args.logging:
            wandb.init(
                project="thinking_fast_and_slow",
                entity="robandken",
                config=config,
                reinit=True,
            )
            wandb.run.name = f'{config_version}'

        train_model(config_version)
        
        # log results to wandb
        if args.logging:
            log_results(config_version)

    # run a range of configs (hparams sweep)
    elif args.begin and args.end:
        # one process is one config
        config_versions = [f'config_dlMU_dnn_{i}' for i in range(args.begin, args.end+1)]
        num_processes = 40
        with multiprocessing.Pool(num_processes) as pool:
            for config_version in config_versions:
                results = pool.apply_async(
                        train_model_multiproc, 
                        args=[config_version]
                    )

            pool.close()
            pool.join()
        
    duration = time.time() - start_time
    print(f'duration = {duration}s')