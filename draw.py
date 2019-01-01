import torch
import argparse
import sys
from PIL import Image
import cv2
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--src1', type=str, default='trial_stage_id_3_target_1')
parser.add_argument('--src2', type=str, default='trial_stage_id_2_target_1')
parser.add_argument('--img_index', type=int, default=0)
parser.add_argument('--crop_len', type=int, default=0)
parser.add_argument('--mix', type=float, default=0.25)
parser.add_argument('--blurK', type=int, default=1)
parser.add_argument('--save', action='store_true')
parser.add_argument('--rescale', type=float, default=0)
output_size = (256, 256)
args = parser.parse_args()

def LoadVecs(src):
    vecs = torch.load('./result/' + src + '.vecs', map_location='cpu')
    vecs = vecs.permute(3, 0, 1, 2)
    if args.crop_len!=0:
        vecs = vecs[:, :, args.crop_len:-args.crop_len, args.crop_len:-args.crop_len].contiguous()
    return vecs

def LoadData():
    fo = open('./data/cifar-10-batches-py/test_batch', 'rb')
    if sys.version_info[0] == 2:
        entry = pickle.load(fo)
    else:
        entry = pickle.load(fo, encoding='latin1')
    return entry

def Map2Img(r):
    ret = np.zeros((32, 32, 3), dtype=np.uint8)
    ret += 255
    ret[:, :, 1][r > 0] = (1 - r[r > 0]) * 255
    ret[:, :, 2][r > 0] = (1 - r[r > 0]) * 255

    
    ret[:, :, 0][r < 0] = (1 + r[r < 0]) * 255
    ret[:, :, 1][r < 0] = (1 + r[r < 0]) * 255
    #ret = cv2.GaussianBlur(ret, (args.blurK, args.blurK), 0)
    ret = cv2.resize(ret, dsize = output_size, interpolation = cv2.INTER_NEAREST)
    return ret


def getMap(r): #h*w*c
    r = np.array(r)
    print (r.shape)
    #r = cv2.blur(r, (args.blurK, args.blurK))
    r = cv2.GaussianBlur(r, (args.blurK, args.blurK), 0)
    r = np.mean(r, axis=2) #to gray
    r = r / np.abs(r).max()
    return r

def Rescale(r):
    if args.rescale != 0:
        r = r * (np.abs(r)**args.rescale)
        return r / np.abs(r).max()
    return r

r1 = LoadVecs(args.src1)
r2 = LoadVecs(args.src2)
print (r1.shape)
print (np.abs(r1[0]).max())

map1 = getMap(r1[args.img_index].permute(1, 2, 0))
img1 = Map2Img(Rescale(map1))
map2 = getMap(Rescale(r2[args.img_index].permute(1, 2, 0)))
img2 = Map2Img(map2)

testset = LoadData()
img = testset['data'][args.img_index].reshape((3, 32, 32)).transpose((1, 2, 0))
label = testset['labels'][args.img_index]
print (img.shape)
print (label)
#label for cifar10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print (classes[label])

img = cv2.resize(img, dsize = output_size, interpolation = cv2.INTER_NEAREST)

####################################end of init############################################################

map3 = map1 * map2
map3 /= np.abs(map3).max()
img3 = Map2Img(map3)

#cv2.imshow('img3', img3)

img_row0 = np.concatenate((img, img1, img2, img3), axis=1)

img1 = np.array(args.mix * img + (1 - args.mix) * img1, dtype=np.uint8)
img2 = np.array(args.mix * img + (1 - args.mix) * img2, dtype=np.uint8)
img3 = np.array(args.mix * img + (1 - args.mix) * img3, dtype=np.uint8)


img_row1 = np.concatenate((img, img1, img2, img3), axis=1)

img = np.concatenate((img_row0, img_row1), axis=0)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
if args.save:
    cv2.imwrite('./map/' + args.src1 + ';' + args.src2 +';' + str(args.img_index) + '.jpg', img)