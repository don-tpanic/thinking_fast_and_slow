import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter

"""
Custom model layer definitions
"""

class MultiVariateAttention(nn.Module):    
    """
    """
    # __constants__ = ['in_features', 'out_features']
    # in_features: int
    # out_features: int
    units: Tensor
    attn: Tensor

    def __init__(self, n_dims, max_nunits, attn_weighting=0.02, bias: bool = True,
                 device=None, dtype=None, 
                 trainable: bool = True,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(MultiVariateAttention, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features

        # New stuff
        self.n_dims = n_dims
        self.max_nunits = max_nunits
        self.attn_weighting = attn_weighting

        # self.units = torch.nn.Parameter(torch.zeros([self.max_nunits, self.n_dims], dtype=torch.float))
        self.units = Parameter(torch.zeros([self.max_nunits, self.n_dims], dtype=torch.float))

        attn = torch.linalg.cholesky(torch.eye(self.n_dims) * attn_weighting)  # .05
        # self.attn = torch.nn.Parameter(attn.repeat(self.max_nunits, 1, 1))
        self.attn = Parameter(attn.repeat(self.max_nunits, 1, 1))
        # ----------------

        self.register_parameter('bias', None)
        # self.reset_parameters()

    # def reset_parameters(self) -> None:
    #     """
    #     Dimensional attention weights are initialized to 1/3.
    #     """
    #     nn.init.constant_(self.weight, 1/3)

    def forward(self, input: Tensor) -> Tensor:
        """
        """
        # diagonal of self.attn.data cannot be less than 1e-6
        # first extract the diagonal
        diag = torch.diagonal(self.attn, dim1=-2, dim2=-1)
        # second clip min=1e-6
        diag = torch.clamp(diag, min=1e-6)
        # third update the diagonal
        self.attn.data = torch.diag_embed(diag)

        mvn1 = torch.distributions.MultivariateNormal(
            self.units, 
            scale_tril=torch.tril(self.attn)
        )

        act = torch.exp(mvn1.log_prob(input))
        return act


class Mask(nn.Module):
    """
    This Mask is used in two different locations in 
    the network:

        1. Zero out clusters that have not been recruited.
        This is to make sure both cluster inhibition and
        competition only happen within the recruited clusters.

        2. Zero out cluster activations to the decision unit
        except for the winner cluster. When there is a winner, 
        the unit corresponds to winner index will have weight flipped to 1.

    inputs:
    -------
        An array of cluster_actv from ClusterActivation

    params:
    -------
        Default weights are zeros at initialisation. 

    returns:
    --------
        Same shape as input, but masked.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    active_units: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, 
                 trainable: bool = False,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(Mask, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.active_units = Parameter(torch.zeros(in_features, **factory_kwargs))
        # if not trainable:
        #     self.active_units.requires_grad = False
        self.active_units = torch.zeros(in_features, **factory_kwargs)
        self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        """
        Initialization of mask weights to zeros because no clusters recruited yet.
        """
        # nn.init.constant_(self.active_units, 0)  
        NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        """
        Mask out clusters that have not been recruited.
        """
        output = torch.multiply(input, self.active_units)
        return output


class Classfication(nn.Module):
    """
    Compute output probabilities for each class.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None,
                 Phi: float = 1.0,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(Classfication, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter('bias', None)
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        self.reset_parameters()
        self.Phi = Phi

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 0)

    def forward(self, input: Tensor) -> Tensor:
        """
        Matrix multiply between cluster activations and association weights and
        apply softmax to get output probabilities.
        """
        input = torch.unsqueeze(input, 0)  # TODO: ungly..
        logits = torch.matmul(input, self.weight)

        # # apply softmax
        # output = torch.nn.functional.softmax(
        #     torch.multiply(
        #         logits, self.Phi
        #     ), dim=1
        # )
        
        # Output logits for `BCEWithLogitsLoss`
        output = torch.multiply(logits, self.Phi)

        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        