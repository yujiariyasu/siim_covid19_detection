import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from scipy.special import softmax


from .backbones import *
from .senet import *
from .activation import *
from .layers import *
from .self_attention import SelfAttention

def drop_fc(model):
    if model.__class__.__name__ == 'FeatureEfficientNet':
        new_model = model
        nc = model._fc.in_features
    elif model.__class__.__name__ == 'RegNetX':
        new_model = nn.Sequential(*list(model.children())[0])[:-1]
        nc = list(model.children())[0][-1].fc.in_features
    elif model.__class__.__name__ == 'DenseNet':
        new_model = nn.Sequential(*list(model.children())[:-1])
        nc = list(model.children())[-1].in_features
    # elif model.__class__.__name__ == 'EfficientNet':
    #     new_model = nn.Sequential(*list(model.children())[:-2])
    #     import pdb;pdb.set_trace()
    #     nc = 1280
    else:
        new_model = nn.Sequential(*list(model.children())[:-2])
        nc = list(model.children())[-1].in_features
    return new_model, nc


'''
Models
'''

class MultiInstanceModel(nn.Module):
    def __init__(self, base_model, num_classes=2, in_channels=3, effnet=False):
        super(MultiInstanceModel, self).__init__()

        self.model_name = base_model.__class__.__name__
        self.effnet = effnet
        self.encoder, nc = drop_fc(base_model)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(),
            nn.Linear(2*nc, 512), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(512, num_classes)
        if in_channels != 3:
            self.encoder[0].conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, n, ch, w, h = x.shape
        x = x.view(bs*n, ch, w, h) # x: N bs x C x W x W
        x = self.encoder(x) # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x.view(-1, n, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
            .contiguous().view(bs, ch2, n*w2, h2) # x: bs x C' x N W'' x W''
        feature_output = self.head(x)

        x = self.fc(feature_output)
        return feature_output, x

    def __repr__(self):
        return f'MIL({self.model_name})'

class MultiInstanceModelWithWataruAttention(nn.Module):
    def __init__(self, base_model, num_classes=2, in_channels=3, effnet=False):
        super(MultiInstanceModelWithWataruAttention, self).__init__()

        self.model_name = base_model.__class__.__name__
        self.effnet = effnet
        self.encoder, nc = drop_fc(base_model)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.self_attention = SelfAttention(channels=nc, downsampling=1, initial_gamma=-8)

        self.head = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(),
            nn.Linear(2*nc, 512), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(512, num_classes)
        if in_channels != 3:
            self.encoder[0].conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, n, ch, w, h = x.shape
        x = x.view(bs*n, ch, w, h) # x: N bs x C x W x W
        x = self.encoder(x) # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x.view(-1, n, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
            .contiguous().view(bs, ch2, n*w2, h2) # x: bs x C' x N W'' x W''

        x, _ = self.self_attention(x)  # x: bs x C' x N W' x W'   bag-wise

        feature_output = self.head(x)

        x = self.fc(feature_output)
        return feature_output, x

    def __repr__(self):
        return f'MIL({self.model_name})'

sigmoid = nn.Sigmoid()
class Swish(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return Swish.apply(x)

class MetaNN(nn.Module):
    def __init__(self, base_model, num_classes=2, in_channels=3, effnet=False, n_meta_features=None):
        super(MetaNN, self).__init__()

        self.model_name = base_model.__class__.__name__
        self.effnet = effnet
        self.encoder, nc = drop_fc(base_model)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(),
            nn.Linear(2*nc, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 2) # added
        )

        self.meta_nn = nn.Sequential(
            # nn.Linear(n_meta_features, 1024),
            # nn.BatchNorm1d(1024),
            # Swish_Module(),
            # nn.Dropout(p=0.7),
            nn.Linear(n_meta_features+2, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(p=0.7),
            nn.Linear(512, 128),  # FC layer output will have 250 features
            nn.BatchNorm1d(128),
            Swish_Module(),
            # nn.Linear(128, 2),  # added
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(2 + 128, num_classes)
            # nn.Dropout(p=0.5), nn.Linear(512 + 128, num_classes)
        )

        # self.fc = nn.Linear(512 + 128, num_classes)

    def forward(self, x, meta):
        # x: bs x N x C x W x W
        bs, n, ch, w, h = x.shape
        x = x.view(bs*n, ch, w, h) # x: N bs x C x W x W
        x = self.encoder(x) # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x.view(-1, n, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
            .contiguous().view(bs, ch2, n*w2, h2) # x: bs x C' x N W'' x W''
        feature_output = self.head(x)

        # return feature_output, self.meta_nn(torch.cat((feature_output, meta), dim=1))

        meta = self.meta_nn(meta)
        x = torch.cat((feature_output, meta), dim=1)

        x = self.fc(x)
        return feature_output, x

    def __repr__(self):
        return f'MetaNN({self.model_name})'

class AttentionMILModel(nn.Module):

    def __init__(self, base_model, num_instances=3,
                 num_classes=2, gated_attention=True):

        super(AttentionMILModel, self).__init__()

        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        self.squeeze_flatten = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten())
        self.attention = MultiInstanceAttention(
            2*nc, num_instances, 1, gated_attention=gated_attention)
        self.classifier = nn.Sequential(
            Flatten(), nn.Linear(2*nc, num_classes))

    def forward(self, x):
        bs, n, ch, w, h = x.shape
        x =  x.view(bs*n, ch, w, h) # x: bs N x C x W x W
        x = self.encoder(x)  # x: bs N x C' x W' x W'
        x = self.squeeze_flatten(x)  # x: bs N x C'
        x = x.view(bs, n, -1)  # x: bs x N x C'
        a = self.attention(x)  # a: bs x 1 x N
        # x = torch.matmul(a, x)  # x: bs x 1 x C'
        x = torch.matmul((1+a), x)
        x = self.classifier(x)
        return x, x

    def __repr__(self):
        return f'AMIL({self.model_name})'

class MultiInstanceModelLightning(LightningModule):
    def __init__(self, train_loader, valid_loader, num_classes=2,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True),
        metric0=None, metric1=None, lr=1e-4
    ):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = nn.CrossEntropyLoss()
        self.encoder, nc = drop_fc(base_model)
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(2 * nc, 512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(512, num_classes)
        self.train_outputs = []
        self.metric0 = metric0
        self.metric1 = metric1
        self.lr = lr

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, n, ch, w, h = x.shape
        x = x.view(bs * n, ch, w, h)  # x: N bs x C x W x W
        x = self.encoder(x)  # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape

        x = (
            x.view(-1, n, ch2, w2, h2)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, ch2, n * w2, h2)
        )  # x: bs x C' x N W'' x W''
        feature_output = self.head(x)
        x = self.fc(feature_output)
        return x, feature_output

    # dataloaders
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    # optimizer
    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=self.lr)
        optimizer = optim.RAdam(self.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=5, cooldown=1, verbose=True, min_lr=5e-7,
        )
        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=5e-7)
        scheduler = {
            "scheduler": lr_scheduler,
            "reduce_on_plateau": True,
            # "reduce_on_plateau": False,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    # training
    def training_step(self, batch, batch_idx):
        loss, outputs, y = self.__evaluate_batch(batch)

        # self.train_outputs.append({"outputs": outputs.detach().cpu(), "labels": y.detach().cpu()})
        self.train_outputs.append({"outputs": outputs, "labels": y})

        self.log("train_loss", loss.item(), on_step=True, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        train_auc = self.__calculate_metrics(outputs=self.train_outputs, mode="train")
        self.log("train_auc", train_auc, on_epoch=True)
        self.train_outputs = []

    # validation
    def validation_step(self, batch, batch_idx):
        loss, outputs, y = self.__evaluate_batch(batch)
        self.log("val_loss", loss.item(), on_epoch=True)
        return {"val_loss": loss, "outputs": outputs, "labels": y}

    def validation_epoch_end(self, outputs):
        val_auc = self.__calculate_metrics(outputs, mode="valid")
        self.log("val_auc", val_auc)
        self.log("step", self.current_epoch)

    # common step in train and validation
    def __evaluate_batch(self, batch):
        x, y = batch
        outputs, features = self(x)
        loss = self.criterion(outputs, y)
        return loss, outputs, y

    # utils
    def __calculate_metrics(self, outputs, mode="train"):
        total_outputs = torch.cat([x["outputs"] for x in outputs])
        # total_outputs = torch.softmax(total_outputs.cuda(), 1)
        total_outputs = torch.softmax(total_outputs, 1)[:, 1]
        total_labels = torch.cat([x["labels"] for x in outputs])

        auc_value = self.metric0(total_labels.detach().cpu(), total_outputs.detach().cpu())
        return auc_value
