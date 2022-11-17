import torch 
from torch import nn

from clustering_model.models import ClusteringModel 
from dnn_model.models import NeuralNetwork


class FastSlow(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        # initializing models
        if config['fast'] == 'clustering':
            self.FastModel = \
                ClusteringModel(
                    max_num_clusters=config['max_num_clusters'], 
                    in_features=config['in_features'],
                    out_features=config['out_features'],
                    r=config['r'],
                    q=config['q'],
                    specificity=config['specificity'],
                    high_attn_constraint=config['high_attn_constraint'],
                    beta=config['beta'],
                    temp_competition=config['temp_competition'],
                    temp_softwta=config['temp_softwta'],
                    Phi=config['Phi'],
                    thr=config['thr'],
                )
        elif config['fast'] == 'local':
            NotImplementedError()

        if config['slow'] == 'dnn':
            self.SlowModel = \
                NeuralNetwork(
                    input_size=config['input_size'],
                    output_size=config['output_size'],
                    n_units=config['n_units'],
                    networks_depth=config['networks_depth'],
                    act_func=config['act_func'],
                    bias=config['bias'],
                    dropout=config['dropout'],
                    n_classes=config['n_classes'],
                )
        
        # setting up optimizer and lr
        custom_lr_list = []
        # adjust center lr
        for cluster_index in range(config['max_num_clusters']):
            custom_lr_list.append(
                {'params': \
                    self.FastModel.DistanceLayerCollections[f'cluster{cluster_index}'].parameters(), 
                 'lr': \
                    config['lr_clustering'] * config['center_lr_multiplier']}
            )

        # adjust attn lr
        custom_lr_list.append(
            {'params': self.FastModel.DimensionWiseAttnLayer.parameters(), 
             'lr': config['lr_clustering'] * config['high_attn_lr_multiplier']}
        )

        # adjust asso lr
        custom_lr_list.append(
            {'params': self.FastModel.ClsLayer.parameters(), 
             'lr': config['lr_clustering'] * config['asso_lr_multiplier']}
        )

        # adjust dnn lr
        custom_lr_list.append(
            {'params': self.SlowModel.parameters(), 
             'lr': config['lr_dnn']}
        )
        
        if config['optim'] == 'sgd':
            self.optim = torch.optim.SGD(custom_lr_list,)
        if config['loss_fn'] == 'bcelogits':
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inp, epoch, i, signature, y_true):
        y_pred_fast = self.FastModel(inp, epoch, i, signature, y_true)
        y_pred_slow = self.SlowModel(inp)
        y_pred_total = torch.add(y_pred_fast, y_pred_slow)
        return y_pred_fast, y_pred_slow, y_pred_total