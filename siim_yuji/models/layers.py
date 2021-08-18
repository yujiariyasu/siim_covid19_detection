import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None, flatten=False):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
        self.flatten_layer = Flatten()
        self.flatten = flatten

    def forward(self, x):
        x = torch.cat([self.mp(x), self.ap(x)], 1)
        if self.flatten:
            x = self.flatten_layer(x)
        return x


class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        input_shape = x.shape
        output_shape = [input_shape[i] for i in range(self.dim)] + [-1]
        return x.view(*output_shape)


class ChannelPool(nn.Module):

    def __init__(self, dim=1, concat=True):
        super().__init__()
        self.dim = dim
        self.concat = concat
    
    def forward(self, x):
        max_out = torch.max(x, self.dim)[0].unsqueeze(1)
        avg_out = torch.mean(x, self.dim).unsqueeze(1)
        if self.concat:
            return torch.cat((max_out, avg_out), dim=self.dim)
        else:
            return max_out, avg_out


class AdaptiveConcatPool3d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1, 1)
        self.ap = nn.AdaptiveAvgPool3d(sz)
        self.mp = nn.AdaptiveMaxPool3d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class MultiInstanceAttention(nn.Module):
    '''
    Attention-based Multiple Instance Learning
    '''

    def __init__(self, feature_size, instance_size,
                 num_classes=1, hidden_size=512, gated_attention=False):
        super().__init__()

        self.gated = gated_attention

        self.attn_U = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.Tanh()
        )
        if self.gated:
            self.attn_V = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.Sigmoid()
            )
        self.attn_W = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: bs x k x f
        # k: num of instance
        # f: feature dimension
        bs, k, f = x.shape
        x = x.view(bs*k, f)
        if self.gated:
            x = self.attn_W(self.attn_U(x) * self.attn_V(x))
        else:
            x = self.attn_W(self.attn_U(x))
        x = x.view(bs, k, self.attn_W.out_features)
        x = F.softmax(x.transpose(1, 2), dim=2)  # Softmax over k
        return x  # : bs x 1 x k
