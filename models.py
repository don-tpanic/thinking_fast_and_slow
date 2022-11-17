import torch 
from torch import nn

from clustering_model.models import ClusteringModel 
from dnn_model.models import NeuralNetwork


class FastSlow(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        # initializing models
        if config['fast'] == 'clustering':
            self.FastModel = ClusteringModel(config=config)

        elif config['fast'] == 'local':
            NotImplementedError()

        if config['slow'] == 'dnn':
            self.SlowModel = NeuralNetwork(config=config)
        
        # initializing optimizer and loss function
        # NOTE: model specific lr are handled internally in 
        # each model component.
        if config['optim'] == 'sgd':
            custom_lr_list = \
                self.FastModel.custom_lr_list + self.SlowModel.custom_lr_list
            self.optim = torch.optim.SGD(custom_lr_list,)

        if config['loss_fn'] == 'bcelogits':
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inp, epoch, i, signature, y_true):
        y_pred_fast = self.FastModel(inp, epoch, i, signature, y_true)
        y_pred_slow = self.SlowModel(inp)
        y_pred_total = torch.add(y_pred_fast, y_pred_slow)
        return y_pred_fast, y_pred_slow, y_pred_total