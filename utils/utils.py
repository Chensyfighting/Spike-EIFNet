import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from utils.colorize_mask import cityscapes_colorize_mask, camvid_colorize_mask

from spikingjelly.activation_based import layer, neuron, functional


def __init_weight(feature, conv_init, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, layer.Conv2d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d)):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # 增加的
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=True, gt_color=True):
    if output_grey:
        output_grey = Image.fromarray(output)
        output_grey.save(os.path.join(save_path, img_name + '.png'))

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'camvid':
            output_color = camvid_colorize_mask(output)
        elif dataset == 'DDD17_events':
            output_color = ddd17_colorize_mask(output)

        output_color.save(os.path.join(save_path, img_name + '_color.png'))

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'camvid':
            gt_color = camvid_colorize_mask(gt)
        elif dataset == 'DDD17_events':
            gt_color = ddd17_colorize_mask(gt)

        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))


def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters
