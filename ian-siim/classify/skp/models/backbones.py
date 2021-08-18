import timm


def efficientnet_b0(pretrained, strides_to_remove):
    assert 0 < strides_to_remove < 3
    model = timm.create_model('efficientnet_b0', pretrained=pretrained)
    model.conv_stem.stride = (1, 1)
    if strides_to_remove == 2: 
        model.blocks[1][0].conv_dw.stride = (1, 1)
    return model


def mixnet_s(pretrained, strides_to_remove):
    assert 0 < strides_to_remove < 3
    model = timm.create_model('mixnet_s', pretrained=pretrained)
    model.conv_stem.stride = (1, 1)
    if strides_to_remove == 2:
        model.blocks[1][0].conv_dw.stride = (1, 1)
    return model 


def resnet18d(pretrained, strides_to_remove):
    assert 0 < strides_to_remove < 2
    model = timm.create_model('resnet18d', pretrained=pretrained)
    model.conv1[0].stride = (1, 1)
    return model 


def resnest14d(pretrained, strides_to_remove):
    assert 0 < strides_to_remove < 2
    model = timm.create_model('resnest14d', pretrained=pretrained)
    model.conv1[0].stride = (1, 1)
    return model 


def nfnet_l0(pretrained, strides_to_remove):
    assert 0 < strides_to_remove < 3
    model = timm.create_model('nfnet_l0', pretrained=pretrained)
    model.stem.conv1.stride = (1, 1)
    if strides_to_remove == 2: 
        model.stem.conv4.stride = (1, 1)
    return model
