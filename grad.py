import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from args import args
from smo import *
from cifar10 import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

checkpoint = torch.load('./extracted/' + args.unique_id+'.checkpoint')

uid_arg = args.unique_id
debug_arg = args.debug
num_data_arg = args.num_data
gamma_arg = args.gamma
args = checkpoint['args']
args.debug = debug_arg


num_class = 10 if args.target_label != -1 else 2

if args.classifier == 'linear':
    from classifier.linear import Linear
    model = Linear(args.feature_dim, num_class)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
else:
    if args.classifier == 'svm':
        model = checkpoint['model']
    else: 
        raise Exception("invalid classifier", classifier_arg)



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

print ('stop')

feature.load_state_dict(checkpoint['model_state_dict'])
feature = feature.submodel(args.stage_id)
feature.to(device)
feature.eval()

print ('stop1')

"""
grad on loss function(done)
grad on classifer's output(todo)
"""

def _fwwx(x, y):
    #x = torch.atleast_2d(x)
    #y = torch.atleast_2d(y)
    return torch.exp(-gamma_arg * torch.dist(x, y) ** 2)


args.debug = debug_arg

if (args.classifier != 'svm'):
    exit(0)
else:
    var_vec = torch.tensor([]).to(device)
    
 #   dic = unpickle('./data/test_batch')
 #   x,y = dic['data'],dic['labels']
#    x = torch.Tensor(testset.test_data)
#    x= x.permute(0, 3, 1, 2)
#    #X = X.type('torch.FloatTensor')
#    #print (X.shape)
#    y = torch.IntTensor(testset.test_labels)
#    if (num_data_arg<x.shape[0]):
#        x = x[:num_data_arg]
#        y = y[:num_data_arg]
    testloader = torch.utils.data.DataLoader(testset, batch_size=num_data_arg,
                                         shuffle=True, num_workers=args.num_workers)
    for i, batch in enumerate(testloader, 0):
            x,y=batch
            break
    x,y=x.to(device),y.to(device)
    x.requires_grad = True
    
    if args.target_label != -1:
        bt = y == args.target_label
        bf = y != args.target_label
        y[bt] = 1
        y[bf] = -1
    loss=0
    
    print ('stop2')
    xnew = feature(x)
    xnew=xnew.view((xnew.shape[0],-1))
    mean=xnew.mean()
    std=xnew.max()
    xnew=(xnew-mean)/std
    #xnew = x
    
    print ('stop3')
    
    print (model.alpha)
    
 #   print (gamma_arg)
#    print (model.Y)
    print (xnew[0])
    print (model.X[0])
    
    for i in range(xnew.shape[0]):
        for j in range(model.m):
            loss += model.alpha[j]*model.Y[j]*_fwwx(xnew[i],model.X[j])
    
    print (loss)
    
    print ('stop4')
    loss.backward()
    grad = x.grad.data
    print ('stop5')
    
    #print (grad.shape)
    var_vec = torch.cat((var_vec, grad), 0)
    
    #if args.debug and i>10:
    #    break
    
    var_vec = var_vec.permute(1,2,3,0)

    print (var_vec.shape)
    torch.save(var_vec, './result/' + args.unique_id + '.vecs')
    print ('vectors saved at ' + './result/' + args.unique_id +'_'+ '.vecs')
    exit(0)