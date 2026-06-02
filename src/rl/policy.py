import math
import torch
import torch.nn as nn
from torch.distributions import Normal


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        nn.init.constant_(m.bias, 0.0)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim=32, hidden_dim=256, init_log_std=-1.0):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),
        )
        self.net.apply(init_weights)
        with torch.no_grad():
            self.net[-1].weight *= 0.01
            self.net[-1].bias[:action_dim] = 0.0
            self.net[-1].bias[action_dim:] = init_log_std

    def forward(self, obs):
        if torch.isnan(obs).any():
            obs = torch.nan_to_num(obs, 0.0)
        out = self.net(obs)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()
        mu = torch.nan_to_num(mu, 0.0)
        std = torch.nan_to_num(std, 1.0)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        z = dist.rsample()
        log_prob = dist.log_prob(z).sum(dim=-1)
        return z, log_prob, mu, std

    def log_prob(self, obs, z):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        return dist.log_prob(z).sum(dim=-1)

    def entropy(self, obs):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        return dist.entropy().sum(dim=-1)
