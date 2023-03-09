import numpy as np 


def load_data():
    np.random.seed(999)
    data = [1, 2, 3, 4]
    data = np.random.choice(data, size=len(data), replace=False)

    # data = np.array([1, 2, 3, 4])
    print('\ndata sampled', data)
    return data


def training_loop():
    # np.random.seed(999)
    num_runs = 3
    for _ in range(num_runs):
        data = load_data()
        shuflled_indices = np.random.permutation(len(data))
        shuffled_data = data[shuflled_indices]
        print('data shuffled', shuffled_data)


if __name__ == '__main__':
    training_loop()


"""
This test is to understand the behaviour of data shuffling in the training loop,
in main.py 
    1. The correct behaviour should be that the sampled data stays the same all
        the time but the shuffled data changes for every epoch; but also the shuffling
        should be reproducible for restarts.
    2. However, if we have 2 random seeding, one for shuffling, one for sampling,
        the seeding for sampling (which happens at epoch level) will subsequently
        seed the shuffling so that the data points will always have the same 
        order for each epoch which is not what we want.
    3. The solution is to seed shuffling; but in sampling, we only sample 
        once at the very first time, and we save the sampled data (given a seed),
        so next time, we load the data instead of sampling again, which avoids seeding
        the sampling process, which avoids seeding shuffling at epoch level.
"""