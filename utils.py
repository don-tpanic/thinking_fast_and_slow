import multiprocessing
import argparse
import os
import yaml
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_config(config_version):
    with open(os.path.join(f'configs', f'{config_version}.yaml')) as f:
        config = yaml.safe_load(f)
    return config


def load_new_stimuli():
    """
    Create synthetic data-points or load them if they exist.
    (to avoid double-seeding)
    """
    if os.path.exists('new_stimuli.npy'):
        data = np.load('new_stimuli.npy', allow_pickle=True)
    else:
        np.random.seed(999)

        # 0-1
        mu1 = [.35, .7]
        var1 = [.009, .006]
        cov1 = .004
        mu2 = [.65, .5]
        var2 = [.009, .006]
        cov2 = -.004
        
        # % sampling
        npoints = 50
        x1 = np.random.multivariate_normal(
            [mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]], npoints)
        x2 = np.random.multivariate_normal(
            [mu2[0], mu2[1]], [[var2[0], cov2], [cov2, var2[1]]], npoints)

        # round
        x1 = np.around(x1, decimals=2)
        x2 = np.around(x2, decimals=2)

        # inputs = torch.cat([torch.tensor(x1, dtype=torch.float32),
        #                     torch.tensor(x2, dtype=torch.float32)])
        # output = torch.cat([torch.zeros(npoints, dtype=torch.long),
        #                     torch.ones(npoints, dtype=torch.long)])
        
        # concat x1 and x2 into a single array of data-points
        inputs = np.concatenate((x1, x2))

        # create one-hot binary labels for each data-point
        output = np.zeros((npoints*2, 2))
        output[:npoints, 0] = 1
        output[npoints:, 1] = 1

        # group by data-point and add signature
        data = []
        for i in range(len(inputs)):
            data.append([[inputs[i]], [output[i]], i])

        data = np.array(data, dtype=object)
        np.save('new_stimuli.npy', data)
    return data


def load_shepard(problem_type):
    """
    Shepard six problems

    Each data-point has three parts:
        [features, label, signature]
    i.e.[x, y_true, signature]
    """
    if problem_type == 1:
        dp0 = [[[0, 0, 0]], [[1., 0.]], 0]
        dp1 = [[[0, 0, 1]], [[1., 0.]], 1]
        dp2 = [[[0, 1, 0]], [[1., 0.]], 2]
        dp3 = [[[0, 1, 1]], [[1., 0.]], 3]
        dp4 = [[[1, 0, 0]], [[0., 1.]], 4]
        dp5 = [[[1, 0, 1]], [[0., 1.]], 5]
        dp6 = [[[1, 1, 0]], [[0., 1.]], 6]
        dp7 = [[[1, 1, 1]], [[0., 1.]], 7]

    if problem_type == 2:
        dp0 = [[[0, 0, 0]], [[1., 0.]], 0]
        dp1 = [[[0, 0, 1]], [[1., 0.]], 1]
        dp2 = [[[0, 1, 0]], [[0., 1.]], 2]
        dp3 = [[[0, 1, 1]], [[0., 1.]], 3]
        dp4 = [[[1, 0, 0]], [[0., 1.]], 4]
        dp5 = [[[1, 0, 1]], [[0., 1.]], 5]
        dp6 = [[[1, 1, 0]], [[1., 0.]], 6]
        dp7 = [[[1, 1, 1]], [[1., 0.]], 7]
    
    if problem_type == 3:
        dp0 = [[[0, 0, 0]], [[0., 1.]], 0]
        dp1 = [[[0, 0, 1]], [[0., 1.]], 1]
        dp2 = [[[0, 1, 0]], [[0., 1.]], 2]
        dp3 = [[[0, 1, 1]], [[1., 0.]], 3]
        dp4 = [[[1, 0, 0]], [[1., 0.]], 4]
        dp5 = [[[1, 0, 1]], [[0., 1.]], 5]
        dp6 = [[[1, 1, 0]], [[1., 0.]], 6]
        dp7 = [[[1, 1, 1]], [[1., 0.]], 7]
    
    if problem_type == 4:
        dp0 = [[[0, 0, 0]], [[0., 1.]], 0]
        dp1 = [[[0, 0, 1]], [[0., 1.]], 1]
        dp2 = [[[0, 1, 0]], [[0., 1.]], 2]
        dp3 = [[[0, 1, 1]], [[1., 0.]], 3]
        dp4 = [[[1, 0, 0]], [[0., 1.]], 4]
        dp5 = [[[1, 0, 1]], [[1., 0.]], 5]
        dp6 = [[[1, 1, 0]], [[1., 0.]], 6]
        dp7 = [[[1, 1, 1]], [[1., 0.]], 7]
    
    if problem_type == 5:
        dp0 = [[[0, 0, 0]], [[0., 1.]], 0]
        dp1 = [[[0, 0, 1]], [[0., 1.]], 1]
        dp2 = [[[0, 1, 0]], [[0., 1.]], 2]
        dp3 = [[[0, 1, 1]], [[1., 0.]], 3]
        dp4 = [[[1, 0, 0]], [[1., 0.]], 4]
        dp5 = [[[1, 0, 1]], [[1., 0.]], 5]
        dp6 = [[[1, 1, 0]], [[1., 0.]], 6]
        dp7 = [[[1, 1, 1]], [[0., 1.]], 7]
    
    if problem_type == 6:
        dp0 = [[[0, 0, 0]], [[0., 1.]], 0]
        dp1 = [[[0, 0, 1]], [[1., 0.]], 1]
        dp2 = [[[0, 1, 0]], [[1., 0.]], 2]
        dp3 = [[[0, 1, 1]], [[0., 1.]], 3]
        dp4 = [[[1, 0, 0]], [[1., 0.]], 4]
        dp5 = [[[1, 0, 1]], [[0., 1.]], 5]
        dp6 = [[[1, 1, 0]], [[0., 1.]], 6]
        dp7 = [[[1, 1, 1]], [[1., 0.]], 7]

    # print(np.array([dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7], dtype=object))
    return np.array([dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7], dtype=object)


if __name__ == '__main__':
    load_new_stimuli()
    # load_shepard(1)