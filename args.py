import argparse

parser = argparse.ArgumentParser()

#model-related arguments
parser.add_argument('--model', type=str, default='resnet18')

#training-related arguments
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--period', type=int, default=100)

#others
parser.add_argument('--unique_id', type=str, default='trial')
parser.add_argument('--load_trial', action='store_true')
parser.add_argument('--debug', action='store_true')

#test arguments
parser.add_argument('--classifier', type=str, default='linear')
parser.add_argument('--stage_id', type=int, default=0)
parser.add_argument('--feature_dim', type=int, default=4096)
parser.add_argument('--target_label', type=int, default=-1)


args = parser.parse_args()
