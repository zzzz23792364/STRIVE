import torch
from torch import nn
from torch.distributions import Normal


class PPO:
    def __init__(self, policy, optimizer, gamma=0.99, lam=0.95, clip_eps=0.2,
                 target_kl=0.01, entropy_coef=0.0, max_grad_norm=0.5,
                 device="cuda"):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def _compute_gae(self, rewards, values):
        advantages = []
        gae = 0.0
        values = values + [0.0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values[:-1])]
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, obs_list, z_list, rewards, log_probs, values):
        advantages, returns = self._compute_gae(rewards, values)
        obs_batch = torch.stack(obs_list)
        z_batch = torch.stack(z_list)

        mu, std = self.policy(obs_batch)
        log_std = torch.log(std + 1e-8)
        std = log_std.exp()
        dist = Normal(mu, std)
        new_log_probs = dist.log_prob(z_batch).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        old_log_probs = torch.stack(log_probs)
        log_ratio = new_log_probs - old_log_probs
        ratio = log_ratio.exp()
        ratio = torch.clamp(ratio, 0.0, 10.0)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        if torch.isnan(policy_loss) or torch.isinf(policy_loss):
            return {"policy_loss": float("nan"), "entropy": entropy.item(),
                    "approx_kl": 0.0, "clip_frac": 0.0, "skipped": True}

        approx_kl = (-log_ratio).mean().item()

        loss = policy_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            total_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm,
                error_if_nonfinite=False,
            )
        self.optimizer.step()

        with torch.no_grad():
            clip_frac = ((ratio < 1 - self.clip_eps).float().mean()
                         + (ratio > 1 + self.clip_eps).float().mean()).item()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "skipped": False,
        }
