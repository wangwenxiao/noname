import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
def _fwwx(x, y):
    #x = torch.atleast_2d(x)
    #y = torch.atleast_2d(y)
    return torch.exp(-0.1 * torch.dist(x, y) ** 2)#

xnew=torch.Tensor([[1.0, 2.0], [2., 3.]])
xnew.requires_grad = True

loss = _fwwx(xnew[0],xnew[1])
loss.backward()
print (xnew.grad)
