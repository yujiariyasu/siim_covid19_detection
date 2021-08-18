# 4channel
in_channel = 4
model = resnet_mod(resnet34, in_channel=3, num_classes=2, pretrained=True)
if in_channel != 3:
    trained_weight = copy.deepcopy(model.conv1.weight)
    model.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Iafoss way
    model.conv1.weight = nn.Parameter(torch.cat((trained_weight, 0.5*(trained_weight[:,:1,:,:]+trained_weight[:,2:,:,:])),dim=1))
    # model.conv1.weight[:, :] = torch.stack([torch.mean(trained_weight, 1)] * in_channel, dim=1)
