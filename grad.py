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

checkpoint = torch.load('./extracted/' + args.unique_id+'.checkpoint')

uid_arg = args.unique_id
debug_arg = args.debug
args = checkpoint['args']
args.debug = debug_arg

num_class = 10 if args.target_label == -1 else 2

if args.classifier == 'linear':
    from classifier.linear import Linear
    model = Linear(args.feature_dim, num_class)
else:
    raise Exception("invalid classifier", classifier_arg)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

checkpoint = torch.load('./param/' + args.unique_id + '.checkpoint')

debug_arg = args.debug
stage_id_arg = args.stage_id
tlabel_arg = args.target_label

args = checkpoint['args']

args.stage_id = stage_id_arg
args.debug = debug_arg
args.target_label = tlabel_arg
args.unique_id = uid_arg

if args.model == 'resnet18':
    from model.resnet import resnet18
    feature = resnet18()
else:
    raise Exception("invalid model", args.model)

feature.load_state_dict(checkpoint['model_state_dict'])
feature = feature.submodel(args.stage_id)
feature.to(device)
feature.eval()


"""
grad on loss function(done)
grad on classifer's output(todo)
"""

args.debug = debug_arg

var_vec = torch.tensor([]).to(device)

for i, batch in enumerate(testloader, 0):
    x, y = batch
    x, y = x.to(device), y.to(device)
    x.requires_grad = True
    
    fx = model(feature(x))
    if args.target_label == -1:
        loss = F.cross_entropy(fx, y)
    else:
        loss = fx[:,0].sum()
    
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
