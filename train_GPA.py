# --------------------------------------------------------
# Pytorch GPA Cross-domain Detection
# Witten by Minghao Xu, Hang Wang
# Based on the Faster R-CNN code written by Jianwei Yang
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('./lib/')
import math
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, get_lr_at_iter
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv

from model.adaptive_faster_rcnn.vgg16 import vgg16
from model.adaptive_faster_rcnn.resnet import resnet

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# Cosine annealing learning rate
def cosine_da_weight(base_weight, curr_epoch, max_epoch):
    return base_weight * (1 + math.cos(math.pi * min(curr_epoch-1, max_epoch) / max_epoch)) / 2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='source training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--tgt_dataset', dest='tgt_dataset',
                        help='target training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--model_config', dest='model_config',
                        help='the config of model',
                        default='GPA-detection', type=str)
    parser.add_argument('--mode', dest='mode',
                        help='the mode of domain adaptation',
                        default='gcn_adapt', type=str)
    parser.add_argument('--rpn_mode', dest='rpn_mode',
                        help='the mode of domain adaptation for RPN',
                        default='adapt', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, etc.',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=1, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=3, type=int)
    parser.add_argument('--da_weight', dest='da_weight',
                        help='the weight of RCNN adaptation loss',
                        default=1.0, type=float)
    parser.add_argument('--rpn_da_weight', dest='rpn_da_weight',
                        help='the weight of RPN adaptation loss',
                        default=1.0, type=float)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--cosine_rpn_da_weight', dest='cosine_rpn_da_weight',
                        help='cosine_rpn_da_weight',
                        action='store_true')
    parser.add_argument('--pos_r', dest='pos_ratio',
                        help='ration of positive example',
                        default=0.25, type=float)
    parser.add_argument('--rpn_bs', dest='rpn_bs',
                        help='rpn batchsize',
                        default=128, type=int)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight_decay',
                        default=0.0005, type=float)
    parser.add_argument('--warm_up', dest='warm_up',
                        help='warm_up iters',
                        default=200, type=int)


    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default='5', type=str)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


class tgt_sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False

    def __iter__(self):
        self.rand_num_view = torch.randperm(self.num_data).view(-1)
        self.rand_num_view = self.rand_num_view[:(self.batch_size * self.num_per_batch)]

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    # for source domain
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "sim10k":
        args.imdb_name = "sim10k_train"
        args.imdbval_name = "sim10k_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "city":
        args.imdb_name = "city_train"
        args.imdbval_name = "city_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "city_multi":
        args.imdb_name = "city_multi_train"
        args.imdbval_name = "city_multi_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "fog_city":
        args.imdb_name = "fog_city_train"
        args.imdbval_name = "fog_city_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "kitti":
        args.imdb_name = "kitti_train"
        args.imdbval_name = "kitti_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    # for target domain
    if args.tgt_dataset == "sim10k":
        args.imdb_tgt_name = "sim10k_train"
        args.imdbval_tgt_name = "sim10k_test"
    elif args.tgt_dataset == "city":
        args.imdb_tgt_name = "city_train"
        args.imdbval_tgt_name = "city_val"
    elif args.tgt_dataset == "city_multi":
        args.imdb_tgt_name = "city_multi_train"
        args.imdbval_tgt_name = "city_multi_val"
    elif args.tgt_dataset == "fog_city":
        args.imdb_tgt_name = "fog_city_train"
        args.imdbval_tgt_name = "fog_city_val"
    elif args.tgt_dataset == "kitti":
        args.imdb_tgt_name = "kitti_train"
        args.imdbval_tgt_name = "kitti_val"

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.TRAIN.RPN_FG_FRACTION = args.pos_ratio
    cfg.TRAIN.RPN_BATCHSIZE = args.rpn_bs
    cfg.TRAIN.WEIGHT_DECAY = args.weight_decay
    print('RPN_FG_FRACTION:', cfg.TRAIN.RPN_FG_FRACTION)
    print('RPN_BATCHSIZE:', cfg.TRAIN.RPN_BATCHSIZE)
    print('WEIGHT_DECAY:', cfg.TRAIN.WEIGHT_DECAY)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # for source domain
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    imdb_val, roidb_val, ratio_list_val, ratio_index_val = combined_roidb(args.imdbval_name, False)
    val_size = len(roidb_val)
    # for target domain
    tgt_imdb, tgt_roidb, tgt_ratio_list, tgt_ratio_index = combined_roidb(args.imdb_tgt_name)
    tgt_train_size = len(tgt_roidb)
    tgt_imdb_val, tgt_roidb_val, tgt_ratio_list_val, tgt_ratio_index_val = combined_roidb(args.imdbval_tgt_name, False)
    tgt_val_size = len(tgt_roidb_val)

    print()
    print('{:d} roidb entries for source domain'.format(len(roidb)))
    print('{:d} roidb entries for target domain'.format(len(tgt_roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.model_config
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # define the dataloader for source domain
    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    dataset_val = roibatchLoader(roidb_val, ratio_list_val, ratio_index_val, 1, \
                             imdb_val.num_classes, training=False, normalize=False)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                             shuffle=False, num_workers=0, pin_memory=True)

    # define the dataloader for target domain
    tgt_sampler_batch = sampler(tgt_train_size, args.batch_size)

    tgt_dataset = roibatchLoader(tgt_roidb, tgt_ratio_list, tgt_ratio_index, args.batch_size, \
                                 tgt_imdb.num_classes, training=True)
    tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset, batch_size=args.batch_size,
                                                 sampler=tgt_sampler_batch, num_workers=args.num_workers)

    tgt_dataset_val = roibatchLoader(tgt_roidb_val, tgt_ratio_list_val, tgt_ratio_index_val, 1, \
                                 tgt_imdb_val.num_classes, training=False, normalize=False)
    tgt_dataloader_val = torch.utils.data.DataLoader(tgt_dataset_val, batch_size=1,
                                                 shuffle=False, num_workers=0, pin_memory=True)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    tgt_im_data = torch.FloatTensor(1)
    tgt_im_info = torch.FloatTensor(1)
    tgt_num_boxes = torch.FloatTensor(1)
    tgt_gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

        tgt_im_data = tgt_im_data.cuda()
        tgt_im_info = tgt_im_info.cuda()
        tgt_num_boxes = tgt_num_boxes.cuda()
        tgt_gt_boxes = tgt_gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    tgt_im_data = Variable(tgt_im_data)
    tgt_im_info = Variable(tgt_im_info)
    tgt_num_boxes = Variable(tgt_num_boxes)
    tgt_gt_boxes = Variable(tgt_gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, mode=args.mode,
                           rpn_mode=args.rpn_mode)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic, mode=args.mode,
                            rpn_mode=args.rpn_mode)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, mode=args.mode,
                            rpn_mode=args.rpn_mode)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic, mode=args.mode,
                            rpn_mode=args.rpn_mode)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch_size)
    tgt_iters_per_epoch = int(tgt_train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    lr_decay_step = sorted([int(decay_step) for decay_step in args.lr_decay_step.split(',') if decay_step.strip()])
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        loss_temp = 0
        start = time.time()

        while  lr_decay_step and epoch > lr_decay_step[0]:
            lr_decay_step.pop(0)
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        # training
        fasterRCNN.train()

        data_iter = iter(dataloader)
        tgt_data_iter = iter(tgt_dataloader)

        base_lr = lr

        for step in range(iters_per_epoch):

            if epoch == 1 and step <= args.warm_up:
                lr = base_lr * get_lr_at_iter(step / args.warm_up)
            else:
                lr = base_lr

            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            if (step % tgt_iters_per_epoch == 0):
                tgt_data_iter = iter(tgt_dataloader)
            tgt_data = next(tgt_data_iter)
            tgt_im_data.resize_(tgt_data[0].size()).copy_(tgt_data[0])
            tgt_im_info.resize_(tgt_data[1].size()).copy_(tgt_data[1])
            tgt_gt_boxes.resize_(tgt_data[2].size()).copy_(tgt_data[2])
            tgt_num_boxes.resize_(tgt_data[3].size()).copy_(tgt_data[3])

            fasterRCNN.zero_grad()
            rois, tgt_rois, cls_prob, tgt_cls_prob, bbox_pred, tgt_bbox_pred, \
            rpn_loss_cls, _, rpn_loss_box, _, \
            RCNN_loss_cls, _, RCNN_loss_bbox, _, \
            RCNN_loss_intra, RCNN_loss_inter, rois_label, tgt_rois_label, \
            RPN_loss_intra, RPN_loss_inter = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                        tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)

            # adjust RPN's domain adaptation weight / fix it as constant
            if args.cosine_rpn_da_weight:
                rpn_da_weight = cosine_da_weight(args.rpn_da_weight, epoch, args.max_epochs)
            else:
                rpn_da_weight = args.rpn_da_weight

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
                   + args.da_weight * (RCNN_loss_intra + RCNN_loss_inter) \
                   + rpn_da_weight * (RPN_loss_intra + RPN_loss_inter)

            if args.mGPUs:
                loss_temp = loss.mean().item()
            else:
                loss_temp = loss.item()

            # backward
            optimizer.zero_grad()
            if args.mGPUs:
                loss = loss.mean()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()

                    intra_loss = RCNN_loss_intra.mean().item()
                    inter_loss = RCNN_loss_inter.mean().item()

                    rpn_intra_loss = RPN_loss_intra.mean().item()
                    rpn_inter_loss = RPN_loss_inter.mean().item()

                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                    tgt_fg_cnt = torch.sum(tgt_rois_label.data.ne(0))
                    tgt_bg_cnt = tgt_rois_label.data.numel() - tgt_fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()

                    intra_loss = RCNN_loss_intra.item()
                    inter_loss = RCNN_loss_inter.item()

                    rpn_intra_loss = RPN_loss_intra.item()
                    rpn_inter_loss = RPN_loss_inter.item()

                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                    tgt_fg_cnt = torch.sum(tgt_rois_label.data.ne(0))
                    tgt_bg_cnt = tgt_rois_label.data.numel() - tgt_fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), tgt_fg/tgt_bg=(%d/%d), time cost: %f"
                      % (fg_cnt, bg_cnt, tgt_fg_cnt, tgt_bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                print("\t\t\tintra_loss: %.4f, inter_loss: %.4f" \
                      % (intra_loss, inter_loss))
                print("\t\t\trpn_intra_loss: %.4f, rpn_inter_loss: %.4f" \
                      % (rpn_intra_loss, rpn_inter_loss))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
    os.system("watch nvidia-smi")
