import torch
from torch import nn


class REINFORCE:
    def __init__(self, policy, optimizer, gamma=0.99, device="cuda"):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device

    def update(self, rewards, log_probs, baselines=None):
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=self.device)

        if baselines is not None:
            returns = returns - torch.tensor(baselines, device=self.device)

        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)
        policy_loss = -(log_probs_t * returns.unsqueeze(-1)).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {"policy_loss": policy_loss.item()}
