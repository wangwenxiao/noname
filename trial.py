import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
import pickle
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='Dec20_01_stage_id_3_target_10')
parser.add_argument('--src2', type=str, default='Dec20_01_stage_id_2_target_10')
parser.add_argument('--src3', type=str, default='Dec20_01_stage_id_1_target_10')
parser.add_argument('--crop_len', type=int, default=0)
parser.add_argument('--gray', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--no_save', action='store_true')
parser.add_argument('--num_train', type=int, default=9000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--period', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--rescale', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--perturb', type=float, default=0)
parser.add_argument('--nn', type=str, default='SmallNet')


#args for evaluation on adversarial examples
parser.add_argument('--eval_adv', action='store_true')
parser.add_argument('--checkpoint', type=str, default='None')

args = parser.parse_args()
args.unique_id = args.src
if args.src2 != 'None':
    args.unique_id += '.' + args.src2
if args.src3 != 'None':
    args.unique_id += '.'+ args.src3
args.input_channel = 3 * (args.src!='None') + 3 * (args.src2!='None') + 3*(args.src3!='None')

def LoadVecs(src):
    vecs = torch.load('./result/' + src + '.vecs', map_location='cpu')
    vecs = vecs.permute(3, 0, 1, 2)
    if args.crop_len!=0:
        vecs = vecs[:, :, args.crop_len:-args.crop_len, args.crop_len:-args.crop_len].contiguous()
    if args.gray:
        vecs = vecs.mean(1)
    return vecs

def Normalize(r):
    r = r * (np.abs(r)**args.rescale)
    r = r / torch.abs(r).max()
    return r

def LoadData():
    fo = open('./data/cifar-10-batches-py/test_batch', 'rb')
    if sys.version_info[0] == 2:
        entry = pickle.load(fo)
    else:
        entry = pickle.load(fo, encoding='latin1')
    return entry


r1 = Normalize(LoadVecs(args.src))
if args.src2 != 'None':
    r2 = Normalize(LoadVecs(args.src2))
    r1 = torch.cat((r1, r2), 1)
if args.src3 != 'None':
    r3 = Normalize(LoadVecs(args.src3))
    r1 = torch.cat((r1, r3), 1)

testset = LoadData()

label = torch.tensor(testset['labels'])
print (r1.shape)

from model.small import *
if args.nn == 'SmallNet':
    model = SmallNet(args.input_channel)
elif args.nn =='SmallNet2':
    model = SmallNet2(args.input_channel)

model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

epoch = -1

##eval_adv
if args.eval_adv:
    checkpoint = torch.load('./trial/' + args.checkpoint + '.checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])

    cnt = 0
    valid_loss = 0
    valid_acc = 0
    model.eval()
    with torch.no_grad():
        for i in range(args.num_train, 10000, args.batch_size):
            x, y = r1[i: i + args.batch_size], label[i: i + args.batch_size]
            x, y = x.to(device), y.to(device)
            
            fx = model(x)
            loss = F.cross_entropy(fx, y)

            valid_loss += loss.item()
            valid_acc += float((torch.max(fx.data, 1)[1] == y).sum().item()) / y.size(0)
            cnt += 1
            if cnt % args.period == 0 or i + args.batch_size >= 10000:
                print ("[batches: %5d] [loss: %.3f] [acc: %.3f]"%
	                    (cnt, valid_loss / cnt, valid_acc / cnt))
                
                if args.debug:
                    break
    
    exit()

while True:
    epoch += 1

    cnt = 0
    train_loss = 0
    train_acc = 0
    
    model.train()
    for i in range(0, args.num_train, args.batch_size):
        x, y = r1[i: i + args.batch_size], label[i: i + args.batch_size]
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        fx = model(x + torch.randn(x.shape) * x * args.perturb)
        loss = F.cross_entropy(fx, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += float((torch.max(fx.data, 1)[1] == y).sum().item()) / y.size(0)
        cnt += 1
        if (cnt) % args.period == 0 or i + args.batch_size >= args.num_train:
            print ("[train] [epoch: %d] [batches: %5d] [loss: %.3f] [acc: %.3f]"%
	                (epoch, cnt, train_loss / cnt, train_acc / cnt))
            
            if args.debug:
                break
    
    train_loss /= cnt
    train_acc /= cnt

    cnt = 0
    valid_loss = 0
    valid_acc = 0

    model.eval()
    with torch.no_grad():
        for i in range(args.num_train, 10000, args.batch_size):
            x, y = r1[i: i + args.batch_size], label[i: i + args.batch_size]
            x, y = x.to(device), y.to(device)
            
            fx = model(x)
            loss = F.cross_entropy(fx, y)

            valid_loss += loss.item()
            valid_acc += float((torch.max(fx.data, 1)[1] == y).sum().item()) / y.size(0)
            cnt += 1
            if cnt % args.period == 0 or i + args.batch_size >= 10000:
                print ("[valid] [epoch: %d] [batches: %5d] [loss: %.3f] [acc: %.3f]"%
	                    (epoch, cnt, valid_loss / cnt, valid_acc / cnt))
                
                if args.debug:
                    break
    
    
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'src1': args.src,
            'src2': args.src2,
            'src3': args.src3
        }
    if not args.no_save:
        torch.save(checkpoint, './trial/' + args.nn + '_' + args.unique_id + '_'+ str(epoch) + '.checkpoint')
        torch.save(checkpoint, './trial/' + args.nn + '_' + args.unique_id +'.checkpoint')
        print ('checkpoint saved at ' + './trial/' + args.nn + '_' + args.unique_id + '_'+ str(epoch) + '.checkpoint')
    if args.debug:
        break
    
