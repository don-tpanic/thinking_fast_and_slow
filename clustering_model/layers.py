import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter

"""
Custom model layer definitions
"""

class Distance(nn.Module):
    """
    Trainable:
    ----------
        - self.weight, meaning: cluster centers
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, 
                 r: int = 2,
                 trainable: bool = True,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Distance, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((1, in_features), **factory_kwargs))
        self.register_parameter('bias', None)
        self.reset_parameters()
        self.r = r

    def reset_parameters(self) -> None:
        """
        Initialization of a cluster's dimensional centers. Do not matter here
        because cluster recruitment will replace whatever the initializations are.
        """
        nn.init.constant_(self.weight, 0.5)   # 0.5 is really just placeholder for cluster centers

    def forward(self, input: Tensor) -> Tensor:
        """
        Computes dimensional eucledian distance between input and a cluster (r=2)
        Output shape is the same as input shape.
        """
        output = \
            torch.clip(
                (input - self.weight).pow(self.r),
                min=1e-6)
        assert output.shape == input.shape
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class DimWiseAttention(nn.Module):    
    """
    Apply dimension-wise attention to distances between input and cluster centers.

    Trainable:
    ----------
        - self.weight, meaning: dim-wise attention weights
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, 
                 high_attn_constraint="sumtoone",
                #  high_attn_regularizer="entropy",
                #  high_attn_reg_strength=0,
                 trainable: bool = True,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(DimWiseAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((1, in_features), **factory_kwargs))
        self.register_parameter('bias', None)
        self.reset_parameters()
        self.high_attn_constraint = high_attn_constraint
        # self.high_attn_regularizer = high_attn_regularizer
        # self.high_attn_reg_strength = high_attn_reg_strength

    def reset_parameters(self) -> None:
        """
        Dimensional attention weights are initialized to 1/3.
        """
        nn.init.constant_(self.weight, 1/3)

    def forward(self, input: Tensor) -> Tensor:
        """
        Dimensional multiplication between distances and attention weights,
        output shape is the same as input shape.
        """
        if self.high_attn_constraint == "sumtoone":     # TODO: double check if this is correct
            # first make sure weight is nonneg
            self.weight.data = torch.clip(self.weight.data, min=1e-6)
            # then apply sum-to-one constraint
            self.weight.data = self.weight.data / self.weight.data.sum()

        output = torch.multiply(input, self.weight)
        assert output.shape == input.shape
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ClusterActivation(nn.Module):
    """
    Compute overall activation of a cluster after attention.

    Trainable:
    ----------
        - No trainable parameters, operation is deterministic.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, 
                 r: int = 2,
                 q: int = 1,
                 specificity: float = 1,
                 trainable: bool = False,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(ClusterActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter('bias', None)
        self.r = r
        self.q = q
        self.c = specificity

    def reset_parameters(self) -> None:
        NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        # dimension-wise summation
        sum_of_distances = torch.sum(input, axis=1, keepdim=True)
        # print(f"sum_of_distances.shape: {sum_of_distances.shape}")

        # q/r powered sum and multiply specificity
        qr_powered_sum = torch.pow(sum_of_distances, self.q / self.r)
        c_sum = torch.multiply(qr_powered_sum, self.c)
        # print(f'c_sum.shape: {c_sum.shape}')

        # get cluster activation.
        cluster_actv = torch.exp(-c_sum)
        # print(f'cluster_actv.shape: {cluster_actv.shape}')

        return cluster_actv

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


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
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, 
                 trainable: bool = False,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(Mask, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((1, in_features), **factory_kwargs))
        if not trainable:
            self.weight.requires_grad = False
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialization of mask weights to zeros because no clusters recruited yet.
        """
        nn.init.constant_(self.weight, 0)  

    def forward(self, input: Tensor) -> Tensor:
        """
        Mask out clusters that have not been recruited.
        """
        output = torch.multiply(input, self.weight)
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ClusterSoftmax(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None,
                 temp=None, beta=None,
                 trainable: bool = False,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(ClusterSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter('bias', None)
        self.temp = temp
        self.beta = beta

    def reset_parameters(self) -> None:
        NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        """
        If it's the first softmax, this layer perform cluster competition
        If it's the second softmax, this layer perform soft winner take all
        """
        # NOTE: careful nonzero returns a tuple (batch_dim, actv_dim)
        nonzero_clusters_indices = torch.nonzero(input, as_tuple=True)
        clusters_actv_nonzero = input[nonzero_clusters_indices]
        
        # NOTE: ugly conversion from beta to temp (kept because used in TF impl)
        if self.temp == 'equivalent':
            temp = clusters_actv_nonzero / (
                self.beta * torch.log(clusters_actv_nonzero)
            )
            # TODO: interesting, self.temp override will cause error.

        else:
            temp = self.temp

        # softmax probabilities and flatten as required
        nom = torch.exp(clusters_actv_nonzero / temp)
        denom = torch.sum(nom)
        softmax_proba = nom / denom
        softmax_weights = torch.zeros(input.shape[-1])
        softmax_weights[nonzero_clusters_indices[-1]] = softmax_proba
        
        clusters_actv_softmax = torch.multiply(input, softmax_weights)
        assert clusters_actv_softmax.shape == input.shape
        return clusters_actv_softmax, nonzero_clusters_indices

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ClusterSupport(nn.Module):
    def __init__(self, bias: bool = True,
                 device=None, dtype=None,
                 trainable: bool = False,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(ClusterSupport, self).__init__()
        self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        NotImplementedError()

    def forward(self, 
                cluster_index: Tensor, 
                assoc_weights: Tensor, 
                y_true: Tensor
                ) -> Tensor:
        """
        Compute single cluster's support for the right model response.
        """
        # get assoc weights of a cluster
        cluster_assoc_weights = assoc_weights[cluster_index, :]

        # print(f'cluster_index: {cluster_index}')
        # print(f'assoc_weights: {assoc_weights}')
        # print(f'cluster_assoc_weights: {cluster_assoc_weights}')

        # based on y_true
        if y_true[0][0] == 0:
            w_correct = cluster_assoc_weights[1]
            w_incorrect = cluster_assoc_weights[0]
        else:
            w_correct = cluster_assoc_weights[0]
            w_incorrect = cluster_assoc_weights[1]

        support = (w_correct - w_incorrect) / (
            torch.abs(w_correct) + torch.abs(w_incorrect) + 1e-6
        )
        return support


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
        