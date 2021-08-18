import numpy as np
import re
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from .. import builder
from . import backbones
from . import smp
from .pooling import GeM, AdaptiveConcatPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from .sequence import *
from .swin_dlv3p import SwinDeepLabV3Plus
from .swin_fpn import SwinFPN


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
    if 'swintransformer' in str(backbone):
        return backbone

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
        if 'swintransformer' in str(backbone):
            backbone.head = nn.Identity()
        else:
            assert hasattr(backbone.head, 'fc')
            backbone.head.fc = nn.Identity()
    else:
        print(backbone)
        raise Exception('Unable to find classifier layer to replace')
    return backbone


def remove_strides(model, backbone_name, strides_to_remove):
    if 'efficientnet' in backbone_name:
        return _remove_efficientnet_strides(model, strides_to_remove)


class Net2D(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 remove_stride=0,
                 remove_maxpool=False,
                 pool='avg',
                 multisample_dropout=False,
                 in_channels=3,
                 load_pretrained=None):
        super().__init__()
        if remove_stride > 0:
            self.backbone = getattr(backbones, backbone)(pretrained, remove_stride)
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.backbone = replace_classifier_with_identity(self.backbone)
        if remove_maxpool:
            assert hasattr(self.backbone, 'maxpool'), f'`{backbone}` does not have `maxpool` layer'
            self.backbone.maxpool = nn.Identity()
        # Determine feature dimension
        if hasattr(self.backbone, 'patch_embed'):
            h,w = self.backbone.patch_embed.img_size
            dummy = torch.from_numpy(np.ones((2,3,h,w))).float()
        else:
            dummy = torch.from_numpy(np.ones((2,3,128,128))).float()
        feats = self.backbone(dummy)
        dim_feats = feats.size(1)
        self.backbone = swap_pooling_layer(self.backbone, 'global_pool', POOL2D_LAYERS[pool])
        if in_channels != 3: self.backbone = change_num_input_channels(self.backbone, in_channels)
        self.msdo = multisample_dropout
        self.dropout = nn.Dropout(p=dropout) if isinstance(dropout, float) else nn.Identity()
        self.linear = nn.Linear(dim_feats, num_classes)
        if load_pretrained:
            print(f'Loading pretrained weights from {load_pretrained} ...')
            weights = torch.load(load_pretrained, map_location=lambda storage, loc: storage)['state_dict']
            weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items()}
            # Get backbone only
            weights = {re.sub(r'^backbone.', '', k) : v for k,v in weights.items() if 'backbone' in k}
            self.backbone.load_state_dict(weights)

    def extract_features(self, x):
        return F.normalize(self.forward_features(x))

    def forward_features(self, x):
        return self.backbone(x).view(x.size(0), -1) 

    def forward(self, x):
        features = self.forward_features(x)
        if self.msdo:
            out = torch.mean(torch.stack([self.linear(self.dropout(features)) for _ in range(self.msdo)]), dim=0)
        else:
            out = self.linear(self.dropout(features))
        return out[:,0] if self.linear.out_features == 1 else out


class MultiNet2D(Net2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(self.linear.in_features*2, self.linear.out_features)

    def forward_features(self, x):
        # x.shape = (B, C, H, W)
        features = torch.stack([self.backbone(x[:,c].unsqueeze(1)).view(x.size(0), -1) for c in range(x.size(1))])
        # features.shape = (C, B, D)
        return torch.cat([features.mean(0), features.max(0)[0]], dim=1)


def NetSMPHybrid(segmentation_model, load_pretrained=None, load_pretrained_decoder=False, **kwargs):

    model = getattr(smp, segmentation_model)(**kwargs)
    if load_pretrained:
        print(f'Loading pretrained weights from {load_pretrained} ...')
        weights = torch.load(load_pretrained, map_location=lambda storage, loc: storage)['state_dict']
        weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items()}
        # Get backbone only
        backbone_weights = {re.sub(r'^backbone.', '', k) : v for k,v in weights.items() if 'backbone' in k}
        if load_pretrained_decoder:
            print('>> Loading decoder weights ...')
            decoder_weights = {re.sub(r'^decoder.', '', k) : v for k,v in weights.items() if 'decoder' in k}
            model.decoder.load_state_dict(decoder_weights)
        model.encoder.load_state_dict(backbone_weights)
    return model 



