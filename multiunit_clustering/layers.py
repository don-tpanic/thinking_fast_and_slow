import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter


class Distance(nn.Module):
    """
    Trainable:
    ----------
        - self.weight, meaning: cluster centers
    """
    def __init__(self, n_dims: int, n_units, bias: bool = True,
                 device=None, dtype=None, 
                 r: int = 2,
                 trainable: bool = False,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Distance, self).__init__()

        self.n_dims = n_dims
        self.n_units = n_units
        self.r = r
        self.weight = Parameter(torch.empty((n_units, n_dims), **factory_kwargs))
        if not trainable:
            self.weight.requires_grad = False
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialization of a cluster's dimensional centers.
        """
        # TODO: doule check consistency
        # nn.init.uniform_(self.weight, 0, 1)
        torch.manual_seed(999)
        data = torch.rand([self.n_units, self.n_dims], dtype=torch.float)
        self.weight.data = data

    def forward(self, input: Tensor) -> Tensor:
        """
        Computes dimensional eucledian distance between input and a cluster (r=2)
        Output shape is the same as input shape.
        """
        # output = \
        #     torch.clip(
        #         (input - self.weight).pow(self.r),
        #         min=1e-6)
        output = torch.pow(abs(input - self.weight), self.r)
        return output


class DimWiseAttention(nn.Module):    
    """
    Trainable:
    ----------
        - self.weight, meaning: dim-wise attention weights
    """
    def __init__(self, n_dims: int, bias: bool = True,
                 device=None, dtype=None, 
                 high_attn_constraint="sumtoone",
                 trainable: bool = False,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(DimWiseAttention, self).__init__()
        self.n_dims = n_dims
        self.weight = Parameter(torch.ones((1, n_dims), **factory_kwargs))
        self.high_attn_constraint = high_attn_constraint
        if not trainable:
            self.weight.requires_grad = False
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Dimensional attention weights are initialized to 1/3.
        """
        nn.init.constant_(self.weight, 1/self.n_dims)

    def forward(self, input: Tensor) -> Tensor:
        """
        Dimensional multiplication between distances and attention weights,
        output shape is the same as input shape.
        """
        # if self.high_attn_constraint == "sumtoone":     # TODO: double check if this is correct
        #     # first make sure weight is nonneg
        #     self.weight.data = torch.clip(self.weight.data, min=1e-6)
        #     # then apply sum-to-one constraint
        #     self.weight.data = self.weight.data / self.weight.data.sum()

        # input shape (num_units, 3)
        # weight shape (1, 3)
        output = torch.multiply(input, self.weight)
        return output


class ClusterActivation(nn.Module):
    """
    Compute overall activation of a cluster after attention.

    Trainable:
    ----------
        - No trainable parameters, operation is deterministic.
    """
    def __init__(self, bias: bool = True,
                 device=None, dtype=None, 
                 r: int = 2,
                 q: int = 1,
                 specificity: float = 1,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(ClusterActivation, self).__init__()
        self.register_parameter('bias', None)
        self.r = r
        self.q = q
        self.c = specificity

    def reset_parameters(self) -> None:
        NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        # dimension-wise summation
        sum_of_distances = torch.sum(input, axis=1)

        # q/r powered sum and multiply specificity
        qr_powered_sum = torch.pow(sum_of_distances, self.q / self.r)
        c_sum = torch.multiply(qr_powered_sum, self.c)

        # get cluster activation.
        cluster_actv = self.c * torch.exp(-c_sum)
        return cluster_actv


class Mask(nn.Module):
    """
    Masking out non-winning units' activations.

    inputs:
    -------
        An array of unit (i.e. cluster) activations (dim collapsed)

    params:
    -------
        Default weights are zeros at initialisation. 

    returns:
    --------
        Same shape as input, but masked.
    """
    def __init__(self, n_units: int, topk, bias: bool = True,
                 device=None, dtype=None, 
                 trainable: bool = False,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(Mask, self).__init__()
        self.n_units = n_units
        self.topk = topk
        self.winning_units = Parameter(torch.zeros(n_units, **factory_kwargs))
        self.active_units = Parameter(torch.zeros(n_units, **factory_kwargs))
        if not trainable:
            self.winning_units.requires_grad = False
            self.active_units.requires_grad = False
        self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        NotImplementedError()

    def forward(self, act: Tensor) -> Tensor:
        """
        Take unit actv as inputs, and mask out non-winning units' activations.
        """
        # act[~self.active_units] = 0  # not connected, no act

        act = torch.multiply(act, self.active_units)

        # get top k winners
        _, win_ind = torch.topk(act,
                                int(self.n_units
                                    * self.topk))

        # since topk takes top even if all 0s, remove the 0 acts
        if torch.any(act[win_ind] == 0):
            win_ind = win_ind[act[win_ind] != 0]

        # self.winning_units[win_ind] = True
        self.winning_units.data[:] = 0
        self.winning_units.data[win_ind] = 1

        # return masked act 
        act = torch.multiply(act, self.winning_units)
        return act, win_ind


class Classfication(nn.Module):
    """
    Compute output probabilities for each class.
    """
    def __init__(self, n_units: int, n_classes: int, bias: bool = True,
                 device=None, dtype=None,
                 phi: float = 1.0,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(Classfication, self).__init__()
        self.n_units = n_units
        self.n_classes = n_classes
        self.register_parameter('bias', None)
        self.weight = Parameter(torch.zeros((n_units, n_classes), **factory_kwargs))
        self.reset_parameters()
        self.phi = phi

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 0)

    def forward(self, input: Tensor) -> Tensor:
        """
        Matrix multiply between cluster activations and association weights and
        apply softmax to get output probabilities.
        """
        input = torch.unsqueeze(input, 0)  # TODO: ungly..

        logits = torch.matmul(input, self.weight)
        
        # Output logits for `BCEWithLogitsLoss`
        output = torch.multiply(logits, self.phi)
        return output
