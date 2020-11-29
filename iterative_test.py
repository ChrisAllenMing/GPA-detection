import os, sys
sys.path.append('./lib/')
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('--dataset', dest='dataset',
                    help='target domain dataset',
                    default='pascal_voc', type=str)
parser.add_argument('--model_config', dest='model_config',
                    help='the config of model',
                    default='pascal_voc', type=str)
parser.add_argument('--net', dest='net',
                    help='vgg16, res50, res101, res152',
                    default='res50', type=str)
parser.add_argument('--load_dir', dest='load_dir',
                    help='directory to load models', default="models",
                    type=str)
parser.add_argument('--cuda', dest='cuda',
                    help='whether use CUDA',
                    action='store_true')
parser.add_argument('--checksession', dest='checksession',
                    help='checksession to load model',
                    default=1, type=int)
parser.add_argument('--checkepoch', dest='checkepoch',
                    help='checkepoch to load network',
                    default=1, type=int)
parser.add_argument('--checkpoint', dest='checkpoint',
                    help='checkpoint to load network',
                    default=10021, type=int)
parser.add_argument('--start_epoch', dest='start_epoch',
					help='the start epoch',
					default=1, type=int)
parser.add_argument('--end_epoch', dest='end_epoch',
					help='the end epoch',
					default=10, type=int)
parser.add_argument('--gpu_id', dest = 'gpu_id',
					help='the id of gpu running on',
					default=0, type=int)
parser.add_argument('--pos_r', dest='pos_r',
                    help='ration of positive example',
                    default=0.25, type=float)
parser.add_argument('--test_mode', dest='test_mode',
                    help='baseline or GPA',
                    default='GPA', type=str)
args = parser.parse_args()

for epoch in range(args.start_epoch, args.end_epoch + 1):
    if args.test_mode == 'GPA':
        template = 'CUDA_VISIBLE_DEVICES={} python test_GPA.py --dataset {} --net {} --load_dir {} --model_config {} --checksession {} --checkepoch {} --checkpoint {} --cuda --pos_r {}'
        cmd = template.format(args.gpu_id, args.dataset, args.net, args.load_dir, args.model_config, args.checksession,
                              epoch, args.checkpoint, args.pos_r)
    else:
        template = 'CUDA_VISIBLE_DEVICES={} python test_baseline.py --dataset {} --net {} --load_dir {} --model_config {} --checksession {} --checkepoch {} --checkpoint {} --cuda'
        cmd = template.format(args.gpu_id, args.dataset, args.net, args.load_dir, args.model_config, args.checksession,
                              epoch, args.checkpoint)
    os.system(cmd)
    print (cmd)

end_cmd = 'watch nvidia-smi'
os.system(end_cmd)