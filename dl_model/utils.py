import multiprocessing
import os
import yaml
import numpy as np


def load_config(config_version):
    with open(os.path.join('configs', f'{config_version}.yaml')) as f:
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
