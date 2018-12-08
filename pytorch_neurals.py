import torch
import numpy as np

def activation(x):
    """ Sigmoid activation function
        Arguments
        --------
        x: torch.Tensor
    """

    return 1/(1+ torch.exp(-x))

#generate some edata
torch.manual_seed(7)
features = torch.randn(1,5)
weights = torch.randn_like(features)
bias = torch.randn((1,1))

#calculate output using the weights and bias tensors

y= activation((features* weights).sum() + bias)
print(y)

#now use matrix multiplication:
#the first gives a size mismatch as our weights are not 1,5 size
#ym = activation(torch.mm(features, weights) +bias)

ym = activation(torch.mm(features, weights.view(5,1))+ bias)
print(ym)

#defining size of each layer