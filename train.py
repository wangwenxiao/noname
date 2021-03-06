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
print ("haha")

#label for cifar10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.model == 'resnet18':
    from model.resnet import resnet18
    model = resnet18()
else:
    raise Exception("invalid model", args.model)
print ("haha2")
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80], gamma=0.1)

epoch = -1
print(epoch)
if args.load_trial:
    checkpoint = torch.load('./param/' + args.unique_id + '.checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    args = checkpoint['args']
print("haha3")
def main():
    global epoch
    best_val_acc=0
    best_val_epoch=-1
    while True:
        epoch += 1
        print("loop")
        cnt = 0
        train_loss = 0
        train_acc = 0

        scheduler.step()
        
        model.train()
        print("train_complete")
        for i, batch in enumerate(trainloader, 0):
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            fx = model(x)
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
            
                fx = model(x)
                loss = F.cross_entropy(fx, y)

                valid_loss += loss.item()
                valid_acc += float((torch.max(fx.data, 1)[1] == y).sum().item()) / y.size(0)
                cnt += 1
                if (i+1) % args.period == 0:
                    print ("[valid] [epoch: %d] [batches: %5d] [loss: %.3f] [acc: %.3f]"%
                        (epoch, i+1, valid_loss / cnt, valid_acc / cnt))
                    
                    if args.debug:
                        break
        valid_acc/=cnt
        if valid_acc>best_val_acc:
            best_val_acc=valid_acc
            best_val_epoch=epoch

        
        
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'train_loss': train_loss,
                'train_acc': train_acc
            }
        torch.save(checkpoint, './param/' + args.unique_id +'_'+ str(epoch) + '.checkpoint')
        torch.save(checkpoint, './param/' + args.unique_id + '.checkpoint')
        print ('checkpoint saved at ' + './param/' + args.unique_id +'_'+ str(epoch) + '.checkpoint')
        print ('best val acc is %.3f in epoch %d'%(best_val_acc, best_val_epoch))
        if args.debug:
            break
if __name__=='__main__':
    main()
