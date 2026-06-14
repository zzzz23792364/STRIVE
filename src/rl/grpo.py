import torch
from torch import nn
from torch.distributions import Normal


class GRPO:
    def __init__(self, policy, optimizer, clip_eps=0.2, group_size=8,
                 max_grad_norm=1.0, device="cuda", reward_baseline_decay=0.99):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.group_size = group_size
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.baseline = None
        self.baseline_decay = reward_baseline_decay

    @torch.no_grad()
    def sample_and_score(self, obs, prior_mu_atk, z_base, attack_idx, model,
                         embed_info, scene_graph, map_idx, map_env):
        """
        Sample K actions using reparameterization:
          delta = delta_mu + sigma * eps,  eps ~ N(0, I)
          z_attack = prior_mu_atk + delta
        """
        K = self.group_size
        NA = z_base.size(0)

        # Policy outputs deterministic shift + exploration std
        delta_mu, sigma = self.policy(obs)          # (1, D), (1, D)
        delta_mu = delta_mu.squeeze(0)             # (D,)
        sigma = sigma.squeeze(0) + 1e-5            # (D,)

        # Reparameterize
        eps = torch.randn(K, delta_mu.size(-1), device=self.device)
        delta = delta_mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (K, D)

        # z = prior_mu + delta
        z_attack = prior_mu_atk.unsqueeze(0) + delta               # (K, D)

        # log_prob of delta under N(delta_mu, sigma^2)
        dist = Normal(delta_mu.unsqueeze(0), sigma.unsqueeze(0))
        log_probs = dist.log_prob(delta).sum(dim=-1)

        # Assemble full z: [ego_prior, other_posterior, attack_z]
        z_base_k = z_base.unsqueeze(0).expand(K, NA, 32).clone()
        z_base_k[:, attack_idx, :] = z_attack
        z_full = z_base_k.transpose(0, 1)

        # Decode all K at once
        dec_out = model.decode_embedding(z_full, embed_info,
                                          scene_graph, map_idx, map_env)
        future_pred = dec_out["future_pred"]

        # Compute rewards: -min distance between attacker and ego
        rewards = []
        norm = model.get_normalizer()
        for k in range(K):
            fp_k = norm.unnormalize(future_pred[:, k])
            atk_pos = fp_k[attack_idx, :, :2]
            ego_pos = fp_k[0, :, :2]
            min_d = torch.norm(atk_pos - ego_pos, dim=-1).min()
            rewards.append(-min_d.item())

        rewards_t = torch.tensor(rewards, device=self.device)
        r_mean = rewards_t.mean().item()
        if self.baseline is None:
            self.baseline = r_mean
        else:
            self.baseline = self.baseline_decay * self.baseline + \
                            (1 - self.baseline_decay) * r_mean

        adv = rewards_t - self.baseline
        if adv.std() > 1e-6:
            adv = adv / (adv.std() + 1e-8)

        best_k = rewards_t.argmax()
        best_delta = delta[best_k].clone()

        return delta, log_probs, adv, rewards_t, best_delta

    def update(self, obs, delta, log_probs_old, advantages):
        """PPO-clip update on the deltas"""
        delta_mu, sigma = self.policy(obs)
        sigma = sigma + 1e-5
        dist = Normal(delta_mu, sigma)
        log_probs_new = dist.log_prob(delta).sum(dim=-1)

        ratio = (log_probs_new - log_probs_old).exp()
        ratio = torch.clamp(ratio, 0.0, 10.0)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        if torch.isnan(policy_loss) or torch.isinf(policy_loss):
            return {"policy_loss": float("nan"), "skipped": True}

        self.optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm,
                                     error_if_nonfinite=False)
        self.optimizer.step()

        return {"policy_loss": policy_loss.item(), "skipped": False}
