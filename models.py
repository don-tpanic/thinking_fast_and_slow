import torch 
from torch import nn

from clustering_model.models import ClusteringModel 
from dnn_model.models import NeuralNetwork
from multiunit_clustering.models import MultiUnitCluster


class FastSlow(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # initializing models
        if config['fast'] == 'clustering':
            self.FastModel = ClusteringModel(config=config['fast_config'])

        elif config['fast'] == 'multiunit_clustering':
            self.FastModel = MultiUnitCluster(config=config['fast_config'])

        if config['slow'] == 'dnn':
            self.SlowModel = NeuralNetwork(config=config['slow_config'])
        
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
        if self.config['fast'] == 'clustering':
            y_logits_fast = self.FastModel(inp, epoch, i, signature, y_true)
            extra_stuff = ()

        elif self.config['fast'] == 'multiunit_clustering':
            y_logits_fast, act, win_ind = self.FastModel(inp, epoch, i, y_true)
            extra_stuff = (act, win_ind)

        y_logits_slow = self.SlowModel(inp)
        y_logits_total = torch.add(y_logits_fast, y_logits_slow)

        # TODO: better way handle extra outputs?
        return y_logits_fast, y_logits_slow, y_logits_total, extra_stuff