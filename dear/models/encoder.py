from torch import nn
import torch
import utils


class DrQV2Encoder(nn.Module):

    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class PoolEncoder(nn.Module):

    def __init__(self, obs_shape, repr_dim=None):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),  # 41 x 41
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 39 x 39
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 37 x 37
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 35 x 35
            nn.ReLU(),
            nn.AvgPool2d(4, stride=4)  # 32 * 8 * 8
        )

        if repr_dim is None:
            self.repr_dim = 32 * 8 * 8
            self.projection = nn.Identity()
        else:
            self.repr_dim = repr_dim
            self.projection = nn.Sequential(nn.Linear(32 * 8 * 8, repr_dim),)
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.projection(h)
        return h

class TEDClassifier(nn.Module):
    """TED classifer to predict if the input pair is temporal or non-temporal."""
    def __init__(self, feature_dim):
        super().__init__()

        self.W = nn.Parameter(torch.empty(2, feature_dim))
        self.b = nn.Parameter(torch.empty((1, feature_dim)))
        self.W_bar = nn.Parameter(torch.empty((1, feature_dim)))
        self.b_bar = nn.Parameter(torch.empty((1, feature_dim)))
        self.c = nn.Parameter(torch.empty((1, 1)))

        self.W.requires_grad = True
        self.b.requires_grad = True
        self.W_bar.requires_grad = True
        self.b_bar.requires_grad = True
        self.c.requires_grad = True

        nn.init.orthogonal_(self.W)
        nn.init.orthogonal_(self.b)
        nn.init.orthogonal_(self.W_bar)
        nn.init.orthogonal_(self.b_bar)
        nn.init.orthogonal_(self.c)

    def forward(self, inputs):

        x = self.W * inputs
        x = torch.sum(x, dim=1)
        x = x + self.b
        x = torch.abs(x)

        y = torch.square((self.W_bar * torch.transpose(inputs, 1, 0)[0]) + self.b_bar)

        output = (torch.sum((x-y), dim=1) + self.c).squeeze()

        return output