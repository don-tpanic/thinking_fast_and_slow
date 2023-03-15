import numpy as np 

# lc_total = np.load('results/config_dl_dnn_126/lc_total.npy')
# lc_fast = np.load('results/config_dl_dnn_126/lc_fast.npy')
# lc_slow = np.load('results/config_dl_dnn_126/lc_slow.npy')

# # print(lc_total)
# # print(lc_fast)
# # print(lc_slow)

# loss_total = np.load('results/config_dl_dnn_126/loss_total.npy')
# loss_fast = np.load('results/config_dl_dnn_126/loss_fast.npy')
# loss_slow = np.load('results/config_dl_dnn_126/loss_slow.npy')

# print(loss_total)
# print(loss_fast)
# print(loss_slow)

investigation = \
    np.round(np.load('results/config_dl_dnn_126_investigate/investigation.npy'), 2)[0, 250:251]


print(investigation.shape)

for epoch in range(investigation.shape[0]):
    for data_ind in range(investigation.shape[1]):
        print(data_ind, investigation[epoch, data_ind, :])
