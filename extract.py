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

lr_arg = args.lr
momentum_arg = args.momentum
classifier_arg = args.classifier
feature_dim_arg = args.feature_dim
debug_arg = args.debug
stage_id_arg = args.stage_id
tlabel_arg = args.target_label

args = checkpoint['args']

args.debug = debug_arg
args.lr = lr_arg
args.momentum = momentum_arg
args.classifier = classifier_arg
args.feature_dim = feature_dim_arg
args.stage_id = stage_id_arg
args.target_label = tlabel_arg

if args.model == 'resnet18':
    from model.resnet import resnet18
    feature = resnet18()
else:
    raise Exception("invalid model", args.model)

feature.load_state_dict(checkpoint['model_state_dict'])
feature = feature.submodel(args.stage_id)
feature.to(device)
feature.eval()

num_class = 10 if args.target_label == -1 else 2
    

if args.classifier == 'linear':
    from classifier.linear import Linear
    model = Linear(args.feature_dim, num_class)
else:
    raise Exception("invalid classifier", classifier_arg)

model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

epoch = -1

if args.load_trial:
    checkpoint = torch.load('./extracted/' + args.unique_id + '.checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    args = checkpoint['args']


while True:
    epoch += 1

    cnt = 0
    train_loss = 0
    train_acc = 0
    t2t = 0
    
    model.train()
    for i, batch in enumerate(trainloader, 0):
        x, y = batch
        x, y = x.to(device), y.to(device)

        if args.target_label != -1:
            bt = y == args.target_label
            bf = y != args.target_label
            y[bt] = 0
            y[bf] = 1
        
        optimizer.zero_grad()
        fx = model(feature(x))
        print (fx.shape)
        loss = F.cross_entropy(fx, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += float((torch.max(fx.data, 1)[1] == y).sum().item()) / y.size(0)
        t2t += float(((torch.max(fx.data, 1)[1] == y)*(y==0)).sum().item())*10 / y.size(0)
        cnt += 1
        if (i+1) % args.period == 0:
            print ("[train] [epoch: %d] [batches: %5d] [loss: %.3f] [acc: %.3f] [t2f: %.3f]"%
	                (epoch, i+1, train_loss / cnt, train_acc / cnt, 1.0 - t2t / cnt))
            
            if args.debug:
                break
    
    train_loss /= cnt
    train_acc /= cnt

    cnt = 0
    valid_loss = 0
    valid_acc = 0
    t2t = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(testloader, 0):
            x, y = batch
            x, y = x.to(device), y.to(device)
            if args.target_label != -1:
                bt = y == args.target_label
                bf = y != args.target_label
                y[bt] = 0
                y[bf] = 1
            
            fx = model(feature(x))
            loss = F.cross_entropy(fx, y)

            valid_loss += loss.item()
            valid_acc += float((torch.max(fx.data, 1)[1] == y).sum().item()) / y.size(0)
            t2t += float(((torch.max(fx.data, 1)[1] == y)*(y==0)).sum().item())*10 / y.size(0)
            cnt += 1
            if (i+1) % args.period == 0:
                print ("[valid] [epoch: %d] [batches: %5d] [loss: %.3f] [acc: %.3f] [t2f: %.3f]"%
	                    (epoch, i+1, valid_loss / cnt, valid_acc / cnt, 1.0 - t2t / cnt))
                
                if args.debug:
                    break

    
    
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'train_loss': train_loss,
            'train_acc': train_acc
        }
    torch.save(checkpoint, './extracted/' + args.unique_id +'_stage_id_' + str(args.stage_id) + '_target_'+ str(args.target_label)+'_'+ str(epoch) + '.checkpoint')
    torch.save(checkpoint, './extracted/' + args.unique_id + '_stage_id_'+str(args.stage_id)+ '_target_'+ str(args.target_label)+'.checkpoint')
    print ('checkpoint saved at ' + './extracted/' + args.unique_id +'_stage_id_' + str(args.stage_id) + '_target_'+ str(args.target_label)+'_'+ str(epoch) + '.checkpoint')
    if args.debug:
        break
