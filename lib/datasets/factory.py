# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.sim10k import sim10k
from datasets.city import city
from datasets.city_multi import city_multi
from datasets.fog_city import fog_city
from datasets.kitti import kitti
from datasets.vg import vg

import numpy as np
num_shot = 10

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

# set up sim10k <split>
for split in ['train', 'test', 'transfer_train', 'transfer_test']:
    name = 'sim10k_{}'.format(split)
    __sets[name] = (lambda split=split: sim10k(split))
    tgt_name = 'sim10k_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: sim10k(split, num_shot))

# set up cityscapes <split>
for split in ['train', 'val']:
    name = 'city_{}'.format(split)
    __sets[name] = (lambda split=split: city(split))
    tgt_name = 'city_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: city(split, num_shot))

# set up multi-class cityscapes <split>
for split in ['train', 'val']:
    name = 'city_multi_{}'.format(split)
    __sets[name] = (lambda split=split: city_multi(split))
    tgt_name = 'city_multi_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: city_multi(split, num_shot))

# set up foggy cityscapes <split>
for split in ['train', 'val']:
    name = 'fog_city_{}'.format(split)
    __sets[name] = (lambda split=split: fog_city(split))
    tgt_name = 'fog_city_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: fog_city(split, num_shot))

# set up kitti <split>
for split in ['train', 'val']:
    name = 'kitti_{}'.format(split)
    __sets[name] = (lambda split=split: kitti(split))
    tgt_name = 'kitti_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: kitti(split, num_shot))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))

  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
