"""Goal-Conditioned Soft Actor-Critic (single-step variant).

- 双 critic + target networks (soft update tau=0.005)
- Auto entropy tuning (target_entropy = -z_dim)
- 适用于单步决策 (sub60 一次性输出 z)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .conditional_policy import QNetwork


class GoalConditionedSAC:
    def __init__(self, obs_dim, goal_dim=16, z_dim=32, hidden=256,
                 lr=3e-4, tau=0.005, gamma=0.99, target_entropy=None,
                 device='cuda'):
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.z_dim = z_dim
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.target_entropy = target_entropy if target_entropy is not None else -float(z_dim)

        # 双 critic + target networks
        self.q1 = QNetwork(obs_dim, goal_dim, z_dim, hidden).to(device)
        self.q2 = QNetwork(obs_dim, goal_dim, z_dim, hidden).to(device)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # Policy 由外部传入 (避免循环 import)
        # Optimizers 在主训练循环中定义

    def critic_loss(self, policy, batch):
        """SAC critic update: minimize (Q - target)^2."""
        obs = batch['obs'].to(self.device)
        z = batch['z'].to(self.device)
        goal = batch['goal'].to(self.device)
        reward = batch['reward'].to(self.device)
        done = batch['done'].to(self.device)

        goal_onehot = F.one_hot(goal, num_classes=self.goal_dim).float()

        with torch.no_grad():
            # target = r + gamma * (1-done) * (min Q_target - alpha * log_prob)
            # single-step: done=True 总是, 所以 target = r
            target = reward.unsqueeze(-1)

        q1_pred = self.q1(obs, z, goal_onehot).unsqueeze(-1)
        q2_pred = self.q2(obs, z, goal_onehot).unsqueeze(-1)
        loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
        return loss, q1_pred.mean().item(), q2_pred.mean().item()

    def soft_update_targets(self):
        """Soft update of target networks."""
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_t.data.mul_(1 - self.tau)
                p_t.data.add_(self.tau * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_t.data.mul_(1 - self.tau)
                p_t.data.add_(self.tau * p.data)
