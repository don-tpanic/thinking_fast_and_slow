import os
from collections import OrderedDict

import torch
from torch import nn 

try:
    import layers
except ModuleNotFoundError:
    from clustering_model import layers

"""
Model definitions.
"""

class ClusteringModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.max_num_clusters = config['max_num_clusters']
        self.in_features = config['in_features']
        self.out_features = config['out_features']
        self.r = config['r'] 
        self.q = config['q']
        self.specificity = config['specificity']
        self.high_attn_constraint = config['high_attn_constraint']
        self.beta = config['beta']
        self.temp_competition = config['temp_competition']
        self.temp_softwta = config['temp_softwta']
        self.Phi = config['Phi']
        self.thr = config['thr']
        self.lr_clustering = config['lr_clustering']
        self.center_lr_multiplier = config['center_lr_multiplier']
        self.high_attn_lr_multiplier = config['high_attn_lr_multiplier']
        self.asso_lr_multiplier = config['asso_lr_multiplier']

        self.Distance = layers.Distance(
            self.in_features,
            self.out_features,
            self.max_num_clusters,
            r=self.r,
            bias=False,
        )

        self.DimensionWiseAttnLayer = layers.DimWiseAttention(
            self.in_features, self.out_features, bias=False,
            high_attn_constraint=self.high_attn_constraint,
        )

        self.ClusterActvLayer = layers.ClusterActivation(
            self.in_features, self.out_features, 
            r=self.r, q=self.q, specificity=self.specificity,
        )

        self.MaskNonRecruit = layers.Mask(
            in_features=self.max_num_clusters, 
            out_features=self.max_num_clusters,
            bias=False, trainable=False
        )

        self.ClusterCompetition = layers.ClusterSoftmax(
            in_features=self.max_num_clusters,
            out_features=self.max_num_clusters, 
            bias=False, trainable=False, 
            temp=self.temp_competition, beta=self.beta
        )

        self.SoftWTA = layers.ClusterSoftmax(
            in_features=self.max_num_clusters, 
            out_features=self.max_num_clusters, 
            bias=False, trainable=False, 
            temp=self.temp_softwta, beta=None
        )

        self.ClusterSupport = layers.ClusterSupport(
            bias=False, trainable=False
        )

        self.ClsLayer = layers.Classfication(
            in_features=self.max_num_clusters, 
            out_features=2,          # TODO: only if SHJ
            bias=False, Phi=self.Phi
        )

        self.custom_lr_list = []

        self.custom_lr_list.append(
            {'params': \
                self.Distance.parameters(), 
             'lr': \
                self.lr_clustering * self.center_lr_multiplier}
        )

        # adjust attn lr
        self.custom_lr_list.append(
            {'params': self.DimensionWiseAttnLayer.parameters(), 
             'lr': self.lr_clustering * self.high_attn_lr_multiplier}
        )

        # adjust asso lr
        self.custom_lr_list.append(
            {'params': self.ClsLayer.parameters(), 
            'lr': self.lr_clustering * self.asso_lr_multiplier}
        )

        # set optimizer and loss fn
        if config['optim'] == 'sgd':
            self.optim = torch.optim.SGD(self.custom_lr_list,)
        if config['loss_fn'] == 'bcelogits':
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def recruit_cluster(self, center, signature):
        """
        Recruit a new cluster by centering on that item.
        
        The mask for cluster recruitment has the corresponding
        unit permanently set to 1.
        
        inputs:
        -------
            center: current trial item.
            signature: current trial item unique ID.
        """
        print(f'[Check] recruit cluster signature={signature}')
        # Center on that item
        # self.DistanceLayerCollections[f'cluster{signature}'].weight.data = center
        self.Distance.weight.data[signature, :] = center
        # A cluster is permanently recruited -- set mask to 1.
        self.MaskNonRecruit.weight.data[:, signature] = 1.

    def make_prediction(self, inp):
        """
        Given some input, make a prediction, and output
        cluster activations for support evaluation if neccessary.
        """
        # (8, 3)
        dim_dist = self.Distance(inp)
        attn_dim_dist = self.DimensionWiseAttnLayer(dim_dist)
        H_concat = self.ClusterActvLayer(attn_dim_dist)

        # mask out non-recruit clusters
        clusters_actv = self.MaskNonRecruit(H_concat)

        # pass on recruited clusters actvs for competition.
        clusters_actv_competed, nonzero_clusters_indices = self.ClusterCompetition(clusters_actv)
       
        # pass on competed clusters actvs for soft wta.
        clusters_actv_softwta, _ = self.SoftWTA(clusters_actv_competed)
        print(f'clusters_actv : {clusters_actv}, clusters_actv_competed : {clusters_actv_competed}, clusters_actv_softwta : {clusters_actv_softwta}')

        # produce output probabilities
        y_pred = self.ClsLayer(clusters_actv_softwta)
        return y_pred, nonzero_clusters_indices, clusters_actv_softwta

    def evaluate_support(self, nonzero_clusters_indices, clusters_actv_softwta, y_true):
        """
        Compute support.
        """
        totalSupport = 0
        print(f'[Check] Evaluating totalSupport')
        assoc_weights = self.ClsLayer.weight.data
        totalSupport = 0
        # NOTE: nonzero_clusters_indices is 2D (batch, actv)
        for cluster_index in nonzero_clusters_indices[-1]:
            support = self.ClusterSupport(
                cluster_index, assoc_weights, y_true
            )
            single_cluster_actv = clusters_actv_softwta[:, cluster_index]
            totalSupport += support * single_cluster_actv

        totalSupport = totalSupport / torch.sum(clusters_actv_softwta)
        print(f'[Check] totalSupport = {totalSupport}')
        return totalSupport

    def forward(self, inp, epoch, i, signature, y_true):
        if epoch == 0 and i == 0:
            self.recruit_cluster(center=inp, signature=signature)
        else:
            y_pred, nonzero_clusters_indices, clusters_actv_softwta = self.make_prediction(inp)
            totalSupport = self.evaluate_support(nonzero_clusters_indices, clusters_actv_softwta, y_true)
            if totalSupport < self.thr:
                self.recruit_cluster(center=inp, signature=signature)

        y_pred, _, _ = self.make_prediction(inp)
        return y_pred