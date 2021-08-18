import torch
import torch.nn as nn
import torch.nn.functional as F
from .lovasz_losses import *
import copy

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, approx, targets):
#         ce_loss = self.bce(approx, targets)
#         pt = torch.exp(-ce_loss)
#         focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
#         return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma=gamma

    def forward(self, input, target):
        assert target.size() == input.size()

        max_val = (-input).clamp(min=0)

        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class FocalLossV2(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLossV2, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * target * torch.log(logit)
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()

class EloLoss(nn.Module):
    def __init__(self):
        super(EloLoss, self).__init__()

    def forward(self, logits, y):
        # "https://towardsdatascience.com/explicit-auc-maximization-70beef6db14e"
        losses = [] 
        y_ = copy.deepcopy(y)
        y_[:, 0] = 0
        y_[:, 2] = y_[:, 2]*2
        y_[:, 3] = y_[:, 3]*3
        opa_y = y_[:, 4]
        y_ = y_[:, :4]
        y_ = y_.sum(1)
        class_ids = y_.unique()
        for i in class_ids:
            class_logits = logits[:,int(i)]
            class_targs = (y_ == i).float()

            mask = (class_targs.unsqueeze(1)*(1-class_targs.unsqueeze(0))).bool()
            class_loss = -torch.sigmoid(class_logits.unsqueeze(1) - class_logits.unsqueeze(0))[mask].mean()
            losses.append(class_loss)

        class_logits = logits[:, 4]
        class_targs = (opa_y == 1).float()

        mask = (class_targs.unsqueeze(1)*(1-class_targs.unsqueeze(0))).bool()
        class_loss = -torch.sigmoid(class_logits.unsqueeze(1) - class_logits.unsqueeze(0))[mask].mean()
        losses.append(class_loss)


        loss = torch.stack(losses)
        return torch.mean(loss)


class F1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, approx, target):
        tp = (approx * target).sum(0)
        tn = (1-approx * 1-target).sum(0)
        fp = (approx > target).sum(0)
        fn = (approx < target).sum(0)

        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)

        f1 = 2*p*r / (p+r+1e-6)
        return 1 - f1.mean()

class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        # import pdb;pdb.set_trace()
        inputs = F.sigmoid(inputs)
        Lovasz = lovasz_hinge(inputs, targets, per_image=False)
        return Lovasz

class LovaszMultiClass(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszMultiClass, self).__init__()

    def forward(self, inputs, targets):
        Lovasz = lovasz_softmax(inputs, targets, per_image=False)
        return Lovasz

