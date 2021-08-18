import numpy as np
import re
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..effdet import EfficientDet, DetBenchTrain, DetBenchPredict, get_efficientdet_config
from omegaconf import OmegaConf
from .. import builder
from . import backbones
from .pooling import GeM, AdaptiveConcatPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from .sequence import *

POOL2D_LAYERS = {
    'gem': GeM(p=3.0, dim=2),
    'concat': AdaptiveConcatPool2d(),
    'avg': AdaptiveAvgPool2d(1),
    'max': AdaptiveMaxPool2d(1),
}


def change_num_input_channels(model, in_channels=1):
    for i, m in enumerate(model.modules()):
      if isinstance(m, (nn.Conv2d,nn.Conv3d)) and m.in_channels == 3:
        m.in_channels = in_channels
        # First, sum across channels
        W = m.weight.sum(1, keepdim=True)
        # Then, divide by number of channels
        W = W / in_channels
        # Then, repeat by number of channels
        size = [1] * W.ndim
        size[1] = in_channels
        W = W.repeat(size)
        m.weight = nn.Parameter(W)
        break
    return model


def swap_pooling_layer(backbone, pool_layer_name, pool_layer):
    if hasattr(backbone, pool_layer_name):
        setattr(backbone, pool_layer_name, pool_layer)
    else:
        assert hasattr(backbone.head, pool_layer_name)
        setattr(backbone.head, pool_layer_name, pool_layer)
    return backbone


def replace_classifier_with_identity(backbone): 
    if hasattr(backbone, 'classifier'):
        backbone.classifier = nn.Identity()
    elif hasattr(backbone, 'fc'):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, 'head'):
        assert hasattr(backbone.head, 'fc')
        backbone.head.fc = nn.Identity()
    else:
        print(backbone)
        raise Exception('Unable to find classifier layer to replace')
    return backbone


def remove_strides(model, backbone_name, strides_to_remove):
    if 'efficientnet' in backbone_name:
        return _remove_efficientnet_strides(model, strides_to_remove)


def remove_classifier_weights(_state_dict):
    state_dict = _state_dict.copy()
    for k in _state_dict:
        if 'class_net' in k:
            del state_dict[k]
    return state_dict


def EffDet(base, backbone, num_classes, image_size, pretrained, 
           add_classification_head=False, 
           use_fpn_features=False, 
           inference=False, **kwargs):
    config = get_efficientdet_config(base)
    config.backbone_name = backbone
    config.num_classes = num_classes  # do not include background class
    config.image_size = image_size
    for k,v in kwargs.items():
        if k in config:
            config[k] = v 
        else: 
            f'`{k}` does not exist in the current configuration'
    model = EfficientDet(config, 
        pretrained_backbone=pretrained,
        add_classification_head=add_classification_head, 
        use_fpn_features=use_fpn_features)
    if pretrained:
        model_weights = torch.hub.load_state_dict_from_url(config['url'])
        model_weights = remove_classifier_weights(model_weights)
        _ = model.load_state_dict(model_weights, strict=False)
    model = DetBenchTrain(model) if not inference else DetBenchPredict(model)
    return model
