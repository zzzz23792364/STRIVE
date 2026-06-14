"""Mixture-of-Gaussians Policy for Multi-Solution RL.

核心思想: 每个 goal 用 K 个高斯 mode 表示"多种撞法", 解决单 policy 无法多解问题.

设计:
- K=4 modes per goal (sub60 实际可达 cell 数)
- Mode k 专门负责撞到 BD cell k (or k-1, k-4, etc.)
- Sample: (mode_k, z) jointly from mixture
- REINFORCE update with per-(goal, mode) baseline

接口兼容 ConditionalGaussianPolicy, 但 sample 返回 (k, z, log_prob, entropy).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class MixtureGaussianPolicy(nn.Module):
    """obs + goal -> mixture of K Gaussians over z.

    Returns:
        mu: (B, K, z_dim)
        sigma: (B, K, z_dim)
        pi_logits: (B, K) - mixture weights (logits)
    """

    def __init__(self, obs_dim, goal_dim=16, z_dim=32, n_modes=4, hidden=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.z_dim = z_dim
        self.n_modes = n_modes

        self.goal_embed = nn.Linear(goal_dim, 32)
        in_dim = obs_dim + 32

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # K 个 mu
        self.mu_head = nn.Linear(hidden, z_dim * n_modes)
        # K 个 log_sigma (clamp to [-5, 2])
        self.logsig_head = nn.Linear(hidden, z_dim * n_modes)
        # K 个 mixture logits
        self.pi_head = nn.Linear(hidden, n_modes)

    def forward(self, obs, goal_onehot):
        """Return (mu (B,K,z), sigma (B,K,z), pi_logits (B,K))."""
        B = obs.size(0)
        g_emb = F.relu(self.goal_embed(goal_onehot))
        x = torch.cat([obs, g_emb], dim=-1)
        h = self.backbone(x)
        mu = self.mu_head(h).view(B, self.n_modes, self.z_dim)
        log_sig = torch.clamp(self.logsig_head(h), -5, 2).view(B, self.n_modes, self.z_dim)
        sigma = log_sig.exp()
        pi_logits = self.pi_head(h)
        return mu, sigma, pi_logits

    def sample(self, obs, goal_onehot, deterministic=False):
        """Sample (mode_k, z) from mixture.

        Returns:
            mode_k: (B,) int
            z: (B, z_dim)
            log_prob: (B,)
            entropy: (B,)
        """
        mu, sigma, pi_logits = self.forward(obs, goal_onehot)  # (B,K,z), (B,K,z), (B,K)
        pi_dist = Categorical(logits=pi_logits)
        if deterministic:
            mode_k = pi_logits.argmax(dim=-1)  # (B,)
        else:
            mode_k = pi_dist.sample()  # (B,)

        # Sample z from selected mode
        # mu[bs, mode_k, :]
        mu_sel = mu.gather(1, mode_k.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.z_dim)).squeeze(1)  # (B, z)
        sigma_sel = sigma.gather(1, mode_k.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.z_dim)).squeeze(1)  # (B, z)
        normal_dist = Normal(mu_sel, sigma_sel)
        z = normal_dist.rsample()  # (B, z)

        # log_prob = log P(mode_k) + log P(z | mode_k)
        log_prob_mode = pi_dist.log_prob(mode_k)  # (B,)
        log_prob_z = normal_dist.log_prob(z).sum(dim=-1)  # (B,)
        log_prob = log_prob_mode + log_prob_z

        # Entropy (近似)
        entropy_z = normal_dist.entropy().sum(dim=-1)  # (B,)
        entropy_mode = pi_dist.entropy()  # (B,)
        entropy = entropy_z + entropy_mode

        return mode_k, z, log_prob, entropy

    def get_log_prob_entropy(self, obs, goal_onehot, mode_k, z):
        """重新算给定 (mode_k, z) 的 log_prob + entropy (用于 SAC update)."""
        mu, sigma, pi_logits = self.forward(obs, goal_onehot)
        pi_dist = Categorical(logits=pi_logits)
        mu_sel = mu.gather(1, mode_k.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.z_dim)).squeeze(1)
        sigma_sel = sigma.gather(1, mode_k.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.z_dim)).squeeze(1)
        normal_dist = Normal(mu_sel, sigma_sel)
        log_prob_mode = pi_dist.log_prob(mode_k)
        log_prob_z = normal_dist.log_prob(z).sum(dim=-1)
        log_prob = log_prob_mode + log_prob_z
        entropy = normal_dist.entropy().sum(dim=-1) + pi_dist.entropy()
        return log_prob, entropy
