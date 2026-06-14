"""Conditional Gaussian Policy for Goal-Conditioned SAC + HER.

输入: obs (obs_dim) + goal_onehot (goal_dim=16) -> 输出 z (z_dim) 高斯分布
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ConditionalPolicy(nn.Module):
    """obs + goal -> N(mu, sigma) over z.

    用一个 obs encoder (来自现有 build_obs 思路) + goal embedding + MLP.
    """

    def __init__(self, obs_dim, goal_dim=16, z_dim=32, hidden=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.z_dim = z_dim

        self.goal_embed = nn.Linear(goal_dim, 32)
        in_dim = obs_dim + 32

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, z_dim)
        self.logsig_head = nn.Linear(hidden, z_dim)

    def forward(self, obs, goal_onehot):
        """Return (mu, sigma) Gaussian distribution parameters."""
        g_emb = F.relu(self.goal_embed(goal_onehot))
        x = torch.cat([obs, g_emb], dim=-1)
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_sig = torch.clamp(self.logsig_head(h), -5, 2)
        sigma = log_sig.exp()
        return mu, sigma

    def sample(self, obs, goal_onehot, deterministic=False):
        """Sample z from N(mu, sigma) and return (z, log_prob)."""
        mu, sigma = self.forward(obs, goal_onehot)
        if deterministic:
            z = mu
        else:
            dist = Normal(mu, sigma)
            z = dist.rsample()
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(z).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return z, log_prob, entropy

    def get_log_prob_entropy(self, obs, goal_onehot, z):
        """For SAC update: 重新计算给定 z 的 log_prob + entropy."""
        mu, sigma = self.forward(obs, goal_onehot)
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(z).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class QNetwork(nn.Module):
    """obs + z + goal -> scalar Q-value. 双 critic 之一."""

    def __init__(self, obs_dim, goal_dim=16, z_dim=32, hidden=256):
        super().__init__()
        self.goal_embed = nn.Linear(goal_dim, 32)
        in_dim = obs_dim + z_dim + 32

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, z, goal_onehot):
        g_emb = F.relu(self.goal_embed(goal_onehot))
        x = torch.cat([obs, z, g_emb], dim=-1)
        return self.net(x).squeeze(-1)
