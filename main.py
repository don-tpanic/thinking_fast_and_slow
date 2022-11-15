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

    # --- clustering config ---
    max_num_clusters = config['max_num_clusters']
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
    lr_clustering=config['lr_clustering']
    asso_lr_multiplier=config['asso_lr_multiplier']
    center_lr_multiplier=config['center_lr_multiplier']
    high_attn_lr_multiplier=config['high_attn_lr_multiplier']
    random_seed=config['random_seed']
    num_blocks=config['num_blocks']
    num_clusters=config['num_clusters']
    num_runs=config['num_runs']
    thr=config['thr']

    # --- dnn config ---
    input_size=config['input_size']
    output_size=config['output_size']
    networks_depth=config['networks_depth']
    n_units=config['n_units']
    act_func=config['act_func']
    bias=config['bias']
    dropout=config['dropout']
    lr_dnn=config['lr_dnn']
    n_classes=config['n_classes']

    results_path = f'results/{config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.random.seed(random_seed)
    # --------------------------------------------------------------------------
    lc_fast = np.empty(num_blocks)
    lc_slow = np.empty(num_blocks)
    lc_total = np.empty(num_blocks)
    ct = 0
    for run in range(num_runs):
        print(f'= problem_type {problem_type} Run {run} ========================================')
        model = models.FastSlow(
            max_num_clusters=max_num_clusters,
            in_features=in_features,
            out_features=out_features,
            r=r,
            q=q,
            specificity=specificity,
            high_attn_constraint=high_attn_constraint,
            beta=beta,
            temp_competition=temp_competition,
            temp_softwta=temp_softwta,
            Phi=Phi,
            thr=thr,
            input_size=input_size,
            output_size=output_size,
            n_units=n_units,
            networks_depth=networks_depth,
            act_func=act_func,
            bias=bias, 
            dropout=dropout,
            n_classes=n_classes,
        )

        custom_lr_list = []
        # adjust center lr
        for cluster_index in range(max_num_clusters):
            custom_lr_list.append(
                {'params': \
                    model.ClusteringModel.DistanceLayerCollections[f'cluster{cluster_index}'].parameters(), 
                 'lr': \
                    lr_clustering * center_lr_multiplier}
            )

        # adjust attn lr
        custom_lr_list.append(
            {'params': model.ClusteringModel.DimensionWiseAttnLayer.parameters(), 
             'lr': lr_clustering * high_attn_lr_multiplier}
        )

        # adjust asso lr
        custom_lr_list.append(
            {'params': model.ClusteringModel.ClsLayer.parameters(), 
             'lr': lr_clustering * asso_lr_multiplier}
        )

        # adjust dnn lr
        custom_lr_list.append(
            {'params': model.NeuralNetwork.parameters(), 
             'lr': lr_dnn}
        )

        optimizer = torch.optim.SGD(custom_lr_list,)
        loss_fn = nn.BCEWithLogitsLoss()

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
                model, item_proberror_fast, item_proberror_slow, item_proberror = \
                    fit(
                        model=model, 
                        x=x,
                        y_true=y_true,
                        signature=signature,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epoch=epoch,
                        i=i,
                    )
                lc_fast[epoch] += item_proberror_fast
                lc_slow[epoch] += item_proberror_slow
                lc_total[epoch] += item_proberror
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
    np.save(os.path.join(results_path, f'lc_fast_type{problem_type}.npy'), lc_fast)
    np.save(os.path.join(results_path, f'lc_slow_type{problem_type}.npy'), lc_slow)
    np.save(os.path.join(results_path, f'lc_total_type{problem_type}.npy'), lc_total)


if __name__ == '__main__':
    start_time = time.time()
    num_processes = 6
    problem_types = [1, 2, 3, 4, 5, 6]
    config_version = 'config1'

    # train_model(problem_types[0], config_version)
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