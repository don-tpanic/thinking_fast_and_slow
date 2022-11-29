import multiprocessing
import argparse
import os
import yaml
import numpy as np


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


def load_data(problem_type):
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
    return np.array([dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7], dtype=object)
