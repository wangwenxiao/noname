import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src1', type=str, default='Dec20_01_stage_id_2_target_2')
parser.add_argument('--src2', type=str, default='Dec20_03_stage_id_2_target_2')
parser.add_argument('--metric', type=str, default='L2')
parser.add_argument('--crop_len', type=int, default=0)
parser.add_argument('--metric_arg', type=str, default='0')
parser.add_argument('--gray', action='store_true')
args = parser.parse_args()

def LoadVecs(src):
    vecs = torch.load('./result/' + src + '.vecs', map_location='cpu')
    vecs = vecs.permute(3, 0, 1, 2)
    if args.crop_len!=0:
        vecs = vecs[:, :, args.crop_len:-args.crop_len, args.crop_len:-args.crop_len].contiguous()
    if args.gray:
        vecs = vecs.mean(1)
    return vecs

if args.metric=='L2':
    from metric.L2 import L2_normalized_distance as f
elif args.metric=='BinaryIOU':
    from metric.Binary import BinaryIOU as f
elif args.metric=='BinaryION':
    from metric.Binary import BinaryION as f
elif args.metric=='transferL2_mean':
    from metric.Transfer import L2_mean as f
elif args.metric=='transferL2_portion':
    from metric.Transfer import setBase
    setBase(float(args.metric_arg))
    from metric.Transfer import L2_portion as f
elif args.metric=='transferLinf_mean':
    from metric.Transfer import Linf_mean as f
elif args.metric=='transferLinf_portion':
    from metric.Transfer import setBase
    setBase(float(args.metric_arg))
    from metric.Transfer import Linf_portion as f
    

r1 = LoadVecs(args.src1)
r2 = LoadVecs(args.src2)
print (r1.shape)

print (f(r1,r2))
