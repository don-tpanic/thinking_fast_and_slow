import torch 
from torch import nn

from clustering_model.models import ClusteringModel 
from dnn_model.models import NeuralNetwork


class FastSlow(nn.Module):

    def __init__(
        self,
        max_num_clusters, 
        in_features, 
        out_features, 
        r, 
        q, 
        specificity,
        high_attn_constraint,
        beta,
        temp_competition,
        temp_softwta,
        Phi,
        thr,
        input_size,
        output_size,
        n_units,
        networks_depth,
        act_func, 
        bias, 
        dropout,
        n_classes, 
        ):
        super().__init__()

        self.ClusteringModel = \
            ClusteringModel(
                max_num_clusters, 
                in_features, 
                out_features, 
                r, 
                q, 
                specificity,
                high_attn_constraint,
                beta,
                temp_competition,
                temp_softwta,
                Phi,
                thr,
            )

        self.NeuralNetwork = \
            NeuralNetwork(
                input_size,
                output_size,
                n_units,
                networks_depth,
                act_func, 
                bias, 
                dropout,
                n_classes, 
            )
    
    def forward(self, inp, epoch, i, signature, y_true):
        y_pred_fast = self.ClusteringModel(inp, epoch, i, signature, y_true)
        y_pred_slow = self.NeuralNetwork(inp)
        y_pred = torch.add(y_pred_fast, y_pred_slow)
        return y_pred_fast, y_pred_slow, y_pred