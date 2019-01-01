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
num_class = 10
#step 1: load original model and generate adversarial examples
#step 2: take derivative of extracted (composite) model on adversarial examples

checkpoint = torch.load('./param/' + args.unique_id + '.checkpoint')
##################
###    how to set args?    ###

num_class = 10 #if args.target_label == 10 else 2

from model.resnet import resnet18
original = resnet18()

original.load_state_dict(checkpoint['model_state_dict'])
original.to(device)
original.eval()

feature_dim_cal = int(8192/int(2**args.stage_id))

from classifier.linear import Linear
comp_linear = Linear(feature_dim_cal , num_class)

checkpoint = torch.load('./extracted/' + args.unique_id + '_stage_id_' + str(args.stage_id) +'_target_10.checkpoint')
comp_linear.load_state_dict(checkpoint['model_state_dict'])
comp_linear.to(device)
comp_linear.eval()

comp_resnet = resnet18()
comp_resnet = original.submodel(args.stage_id)
comp_resnet.to(device)
comp_resnet.eval()

"""
grad on loss function(done)
grad on classifer's output(todo)
"""
epsilon = 0.03

def attack(image,grad,epsilon=epsilon):
    #sign_data_grad=grad.sign()
    sign_data_grad = grad / ((grad * grad).view(grad.size(0), -1).mean(1).sqrt()).view(grad.size(0), 1, 1, 1)
    perturbed_image=image+epsilon*sign_data_grad
    return perturbed_image

#args.debug = debug_arg

var_vec = torch.tensor([]).to(device)
args.target_label=10
acc100=0
for i, batch in enumerate(testloader, 0):
    acc = 0
    x, y = batch
    x, y = x.to(device), y.to(device)
    x.requires_grad = True
    
    original_result = original(x)
    if args.target_label == 10:
        #loss = F.cross_entropy(original_result, y)
        #loss = F.l1_loss(original_result,y)
        T = original_result.clone()
        T[y]=-10000
        loss =- ((original_result * torch.eye(10).to(device)[y]).sum(1)-torch.mean(T,1)).sum()
    loss.backward()
    grad = x.grad.data
    with torch.no_grad():
        perturbed_image = attack(x,grad).to(device)
    adv_output = original(perturbed_image)
    acc += float((torch.max(adv_output.data, 1)[1] == y).sum().item()) / y.size(0)
    acc100 += acc
    print ("original acc= %.3f"%(acc))
    original.zero_grad

    perturbed_image.requires_grad=True
    gx = comp_linear(comp_resnet(perturbed_image))
    if args.target_label == 10:
        loss_comp = F.cross_entropy(gx, y)
    comp_linear.zero_grad()
    comp_resnet.zero_grad()

    loss_comp.backward()
    grad_comp = perturbed_image.grad.data

    var_vec = torch.cat((var_vec, grad_comp), 0)
    
    if args.debug and i>10:
        break
acc100 /= 100
var_vec = var_vec.permute(1,2,3,0)
print(acc100)
print (var_vec.shape)
torch.save(var_vec, './result/' + args.unique_id + '_stage_id_' + str(args.stage_id) + '_epsilon_' + str(epsilon) + '_comp.vecs')
torch.save(acc100, './result/' + args.unique_id + '_stage_id_' + str(args.stage_id) + '_epsilon_' + str(epsilon) + '_comp.acc100')
print ('vectors saved at ' + './result/' + args.unique_id +'_stage_id_' + str(args.stage_id) +'_epsilon_' + str(epsilon) + '_comp.vecs')