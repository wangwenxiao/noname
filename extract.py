import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from args import args
from smo import *  #add
from cifar10 import *

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
gamma_arg = args.gamma
max_iter_arg = args.max_iter
C_arg = args.C

args = checkpoint['args']

args.debug = debug_arg
args.lr = lr_arg
args.momentum = momentum_arg
args.classifier = classifier_arg
args.feature_dim = feature_dim_arg
args.stage_id = stage_id_arg
args.target_label = tlabel_arg
args.gamma = gamma_arg
args.max_iter = max_iter_arg
args.C = C_arg

if args.model == 'resnet18':
    from model.resnet import resnet18
    feature = resnet18()
else:
    raise Exception("invalid model", args.model)

feature.load_state_dict(checkpoint['model_state_dict'])
feature = feature.submodel(args.stage_id)
feature.to(device)
feature.eval()

num_class = 10 if args.target_label != -1 else 2
    

if args.classifier == 'linear':
    from classifier.linear import Linear
    model = Linear(args.feature_dim, num_class)
else:
    if args.classifier == 'svm':     #add
        kernel = RBF(gamma=args.gamma)
        #dic = unpickle('./data/data_batch_1')
        #X,y = dic['data'],dic['labels']
        X = torch.Tensor(trainset.train_data)
        X = X.permute(0, 3, 1, 2)
        #X = X.type('torch.FloatTensor')
        print (X.shape)
        y = torch.IntTensor(trainset.train_labels)
        print (y.shape)
        #for i in range(2,6):
        #    dic = unpickle('./data/data_batch_'+str(i))
        #    X1,y1 = dic['data'],dic['labels']
        #    X = np.concatenate((X, X1), axis=0)
        #    y = np.concatenate((y, y1), axis=0)
        
        #X,y=trainset
        X,y=X.to(device),y.to(device)
        print (X.shape)
        X_train=feature(X)
        
        if args.target_label != -1:
            bt = y == args.target_label
            bf = y != args.target_label
            y[bt] = 1
            y[bf] = -1
        y_train=y
        
        print ('stop')
        model = SVM(X_train,y_train,max_iter=args.max_iter, kernel=kernel, C=args.C)
        
        print ('stop2')
        
        model.train()
        
        print ('stop3')
 #       dic = unpickle('./data/test_batch')
 #       X,y = dic['data'],dic['labels']
        #X,y = testset.test_data,testset.test_labels
        X = torch.Tensor(testset.test_data)
        X = X.permute(0, 3, 1, 2)
        #X = X.type('torch.FloatTensor')
        #print (X.shape)
        y = torch.IntTensor(testset.test_labels)
        X,y=X.to(device),y.to(device)
        #print (y.shape)
        X_test=feature(X)
        if args.target_label != -1:
            bt = y == args.target_label
            bf = y != args.target_label
            y[bt] = 1
            y[bf] = -1
        y_test=y
        predictions = model.predict(X_test)
        accuracyRate = accuracy(y_test, predictions)
        print('Classification accuracy (%s): %s'
          % (kernel, accuracyRate))
        checkpoint = {
          #  'epoch': epoch,
          #  'model_state_dict': model.state_dict(),
          #  'optimizer_state_dict': optimizer.state_dict(),
            'model': model,
            'args': args,
          #  'train_loss': train_loss,
          #  'train_acc': train_acc
        }
        torch.save(checkpoint, './extracted/' + args.unique_id +'_stage_id_' + str(args.stage_id) + '_target_'+ str(args.target_label)+'_'+ str(epoch) + '.checkpoint')
        torch.save(checkpoint, './extracted/' + args.unique_id + '_stage_id_'+str(args.stage_id)+ '_target_'+ str(args.target_label)+'.checkpoint')
        print ('checkpoint saved at ' + './extracted/' + args.unique_id +'_stage_id_' + str(args.stage_id) + '_target_'+ str(args.target_label)+'_'+ str(epoch) + '.checkpoint')
 #       if args.debug:
 #           break
        exit(0)
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
        loss = F.cross_entropy(fx, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += float((torch.max(fx.data, 1)[1] == y).sum().item()) / y.size(0)
        cnt += 1
        if (i+1) % args.period == 0:
            print ("[train] [epoch: %d] [batches: %5d] [loss: %.3f] [acc: %.3f]"%
                (epoch, i+1, train_loss / cnt, train_acc / cnt))
            
            if args.debug:
                break
    
    train_loss /= cnt
    train_acc /= cnt

    cnt = 0
    valid_loss = 0
    valid_acc = 0

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
            cnt += 1
            if (i+1) % args.period == 0:
                print ("[valid] [epoch: %d] [batches: %5d] [loss: %.3f] [acc: %.3f]"%
                    (epoch, i+1, valid_loss / cnt, valid_acc / cnt))
                
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