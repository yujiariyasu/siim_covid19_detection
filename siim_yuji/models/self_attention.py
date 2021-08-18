import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from models.layers import AdaptiveConcatPool2d, Flatten


class SelfAttention(nn.Module):
    def __init__(self, channels, ch_reduction1=8, ch_reduction2=2, downsampling=2, initial_gamma=0, initial_beta=1, kernel='dot'):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.ch_reduction1 = ch_reduction1
        self.ch_reduction2 = ch_reduction2
        self.downsampling = downsampling
        self.initial_gamma = initial_gamma
        self.initial_beta = initial_beta
        self.kernel = kernel

        self.conv1x1_theta = nn.Sequential(
            # nn.Conv2d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction1, kernel_size=1),
            # nn.InstanceNorm2d(self.channels // self.ch_reduction1),
            spectral_norm(nn.Conv2d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction1, kernel_size=1)),
            nn.Tanh()
        )
        self.conv1x1_phi = nn.Sequential(
            # nn.Conv2d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction1, kernel_size=1),
            # nn.InstanceNorm2d(self.channels // self.ch_reduction1),
            spectral_norm(nn.Conv2d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction1, kernel_size=1)),
            nn.Tanh()
        )
        self.conv1x1_g = nn.Sequential(
            # nn.Conv2d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction2, kernel_size=1),
            # nn.InstanceNorm2d(self.channels // self.ch_reduction2),
            spectral_norm(nn.Conv2d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction2, kernel_size=1)),
        )
        self.conv1x1_attn_g = nn.Sequential(
            # nn.Conv2d(in_channels=self.channels // self.ch_reduction2, out_channels=self.channels, kernel_size=1),
            # nn.InstanceNorm2d(self.channels),
            spectral_norm(nn.Conv2d(in_channels=self.channels // self.ch_reduction2, out_channels=self.channels, kernel_size=1)),
        )
        self.gamma = nn.Parameter(torch.full((1,), fill_value=self.initial_gamma, dtype=torch.float), requires_grad=True)
        self.alpha = torch.sigmoid(self.gamma)

        if kernel == 'rbf':
            self.beta = nn.Parameter(torch.full((1,), fill_value=self.initial_beta, dtype=torch.float), requires_grad=True)
        else:
            self.beta = None

        self.maxpool = nn.MaxPool2d(kernel_size=[self.downsampling, self.downsampling], stride=self.downsampling)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: N bs x C x W x W
        nbs, ch, w, h = x.shape
        location_num = h * w
        downsampled_num = location_num // self.downsampling

        # theta path
        theta = self.conv1x1_theta(x)
        theta = (
            theta.permute(0, 2, 3, 1)
            .contiguous()
            .view(nbs, location_num, ch // self.ch_reduction1)
        )  # theta: N bs x W W x C'

        # phi path
        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = (
            phi.permute(0, 2, 3, 1)
            .contiguous()
            .view(nbs, downsampled_num, ch // self.ch_reduction1)
        )  # phi: N bs x W' W' x C'

        if self.kernel == 'rbf':
            sqr = torch.bmm(theta - phi, (theta - phi).transpose(1, 2))  # sqr: N bs x W W x W' W'
            attn = torch.exp(-F.relu(self.beta) * sqr)
        else:   # 'dot'
            attn = torch.bmm(theta, phi.transpose(1, 2))  # attn: N bs x W W x W' W'

        attn = self.softmax(attn)

        # g path
        g = self.conv1x1_g(x)
        g = self.maxpool(g)
        g = (
            g.permute(0, 2, 3, 1)
            .contiguous()
            .view(nbs, downsampled_num, ch // self.ch_reduction2)
        )  # g: N bs x W' W' x C''

        attn_g = torch.bmm(attn, g)
        attn_g = (
            attn_g.permute(0, 2, 1)
            .contiguous()
            .view(nbs, ch // self.ch_reduction2, w, h)
        )  # attn_g: N bs x C'' x W x W
        attn_g = self.conv1x1_attn_g(attn_g)  # attn_g: N bs x C x W x W

        self.alpha = torch.sigmoid(self.gamma)

        return (1 - self.alpha) * x + self.alpha * attn_g, attn_g


class SelfAttention3D(nn.Module):
    def __init__(self, channels, ch_reduction1=8, ch_reduction2=2, downsampling=2, initial_gamma=0, initial_beta=1, kernel='dot'):
        super(SelfAttention3D, self).__init__()
        self.channels = channels
        self.ch_reduction1 = ch_reduction1
        self.ch_reduction2 = ch_reduction2
        self.downsampling = downsampling
        self.initial_gamma = initial_gamma
        self.initial_beta = initial_beta
        self.kernel = kernel

        self.conv_nx1x1_theta = nn.Sequential(
            # nn.Conv3d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction1, kernel_size=(self.n_instance, 1, 1)),
            # nn.InstanceNorm3d(self.channels // self.ch_reduction1),
            spectral_norm(nn.Conv3d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction1, kernel_size=1)),
            nn.Tanh()
        )
        self.conv_nx1x1_phi = nn.Sequential(
            # nn.Conv3d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction1, kernel_size=(self.n_instance, 1, 1)),
            # nn.InstanceNorm3d(self.channels // self.ch_reduction1),
            spectral_norm(nn.Conv3d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction1, kernel_size=1)),
            nn.Tanh()
        )
        self.conv_nx1x1_g = nn.Sequential(
            # nn.Conv3d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction2, kernel_size=(self.n_instance, 1, 1)),
            # nn.InstanceNorm3d(self.channels // self.ch_reduction2),
            spectral_norm(nn.Conv3d(in_channels=self.channels, out_channels=self.channels // self.ch_reduction2, kernel_size=1)),
        )
        self.conv_nx1x1_attn_g = nn.Sequential(
            # nn.Conv3d(in_channels=self.channels // self.ch_reduction2, out_channels=self.channels, kernel_size=(self.n_instance, 1, 1)),
            # nn.InstanceNorm3d(self.channels),
            spectral_norm(nn.Conv3d(in_channels=self.channels // self.ch_reduction2, out_channels=self.channels, kernel_size=1)),
        )
        self.gamma = nn.Parameter(torch.full((1,), fill_value=self.initial_gamma, dtype=torch.float), requires_grad=True)
        self.alpha = torch.sigmoid(self.gamma)

        self.beta = nn.Parameter(torch.full((1,), fill_value=self.initial_beta, dtype=torch.float), requires_grad=True)

        self.maxpool = nn.MaxPool2d(kernel_size=[self.downsampling, self.downsampling], stride=self.downsampling)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: bs x C x N x W x W
        bs, ch, n, w, h = x.shape
        location_num = n * h * w
        downsampled_num = location_num // self.downsampling

        # theta path
        theta = self.conv_nx1x1_theta(x)
        theta = (
            theta.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(bs, location_num, ch // self.ch_reduction1)
        )  # theta: bs x N W W x C'

        # phi path
        phi = self.conv_nx1x1_phi(x)
        phi = (
            phi.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs * n, ch // self.ch_reduction1, w, h)
        )  # phi: N bs x C' x W x W
        phi = self.maxpool(phi)
        phi = (
            phi.view(bs, n, ch // self.ch_reduction1, w, h)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(bs, downsampled_num, ch // self.ch_reduction1)
        )  # phi: bs x N W' W' x C'

        if self.kernel == 'rbf':
            sqr = torch.bmm(theta - phi, (theta - phi).transpose(1, 2))  # sqr: bs x N W W x N W' W'
            attn = torch.exp(-F.relu(self.beta) * sqr)
        else:   # 'dot'
            attn = torch.bmm(theta, phi.transpose(1, 2))  # attn: bs x N W W x N W' W'

        attn = self.softmax(attn)

        # g path
        g = self.conv_nx1x1_g(x)
        g = (
            g.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs * n, ch // self.ch_reduction2, w, h)
        )  # g: N bs x C' x W' x  W'
        g = self.maxpool(g)
        g = (
            g.view(bs, n, ch // self.ch_reduction2, w, h)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(bs, downsampled_num, ch // self.ch_reduction2)
        )  # g: bs x N W' W' x C''

        attn_g = torch.bmm(attn, g)
        attn_g = (
            attn_g.permute(0, 2, 1)
            .contiguous()
            .view(bs, ch // self.ch_reduction2, n, w, h)
        )  # attn_g: bs x C'' x N x W x W
        attn_g = self.conv_nx1x1_attn_g(attn_g)  # attn_g: bs x C x N x W x W

        self.alpha = torch.sigmoid(self.gamma)

        return (1 - self.alpha) * x + self.alpha * attn_g, attn_g


class GatedAttention(nn.Module):
    def __init__(self, channels, ch_reduction=4, initial_gamma=0):
        super(GatedAttention, self).__init__()

        self.L = channels
        self.D = self.L // ch_reduction
        self.K = 1
        self.initial_gamma = initial_gamma

        self.attention_V = nn.Sequential(
            # nn.Linear(self.L, self.D),
            # nn.BatchNorm1d(self.D),
            spectral_norm(nn.Linear(self.L, self.D)),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            # nn.Linear(self.L, self.D),
            # nn.BatchNorm1d(self.D),
            spectral_norm(nn.Linear(self.L, self.D)),
            nn.Sigmoid()
        )
        # self.attention = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.Tanh(),
        #     nn.Linear(self.D, self.K)
        # )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.gamma = nn.Parameter(torch.full((1,), fill_value=self.initial_gamma, dtype=torch.float), requires_grad=True)
        self.alpha = torch.sigmoid(self.gamma)

        self.spatial_pool = AdaptiveConcatPool2d()

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, n, ch, w, h = x.shape
        x_pool = (
            x.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, ch, n * w, h)
        )  # x_pool: bs x C x N W x W
        x_pool = self.spatial_pool(x_pool)  # x_pool: bs x 2C x 1 x 1

        H = x.view(bs * n, ch, w, h)    # H: N bs x C x W x W
        H = self.spatial_pool(H)  # H: N bs x 2C x 1 x 1
        nbs, ch2, w2, h2 = H.shape
        H = H.view(nbs, ch2 * w2 * h2)  # H: bs N x 2C

        A_V = self.attention_V(H)  # bs N x D
        A_U = self.attention_U(H)  # bs N x D
        A = self.attention_weights(A_V * A_U)   # element wise multiplication # bs N x K
        # A = self.attention(H)  # bs N x K
        A = (
            A.view(bs, n, self.K)
            .transpose(2, 1)
            .contiguous()
        )   # bs x K x N
        A = F.softmax(A, dim=2)  # softmax over N

        M = torch.bmm(A, H.view(bs, n, ch2 * w2 * h2))  # bs x K x 2C
        M = M.view(bs, ch2, w2, h2)    # bs x 2C x 1 x 1

        self.alpha = torch.sigmoid(self.gamma)

        return (1 - self.alpha) * x_pool + self.alpha * M, A
