import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from args import args
from cifar10 import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

checkpoint = torch.load('./param/' + args.unique_id + '.checkpoint')

debug_arg = args.debug

args = checkpoint['args']

if args.model == 'resnet18':
    from model.resnet import resnet18
    feature = resnet18()
else:
    raise Exception("invalid model", args.model)

feature.load_state_dict(checkpoint['model_state_dict'])
feature = feature.submodel(args.stage_id)
feature.to(device)
feature.eval()

checkpoint = torch.load('./extracted/' + args.unique_id + '.checkpoint')
args = checkpoint['args']

if args.classifier == 'linear':
    from classifier.linear import Linear
    model = Linear(args.feature_dim)
else:
    raise Exception("invalid classifier", classifier_arg)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

"""
grad on loss function(done)
grad on classifer's output(todo)
"""

args.debug = debug_arg

var_vec = torch.tensor([])

for i, batch in enumerate(testloader, 0):
    x, y = batch
    x, y = x.to(device), y.to(device)
    x.requires_grad = True
    
    fx = model(feature(x))
    loss = F.cross_entropy(fx, y)
    loss.backward()
    grad = x.grad.data
    
    #print (grad.shape)
    var_vec = torch.cat((var_vec, grad), 0)
    
    if args.debug and i>10:
        break

var_vec = var_vec.permute(1,2,3,0)

print (var_vec.shape)
torch.save(var_vec, './result/' + args.unique_id + '.vecs')
print ('vectors saved at ' + './result/' + args.unique_id +'_'+ '.vecs')

