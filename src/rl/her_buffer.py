"""HER Replay Buffer for Goal-Conditioned RL.

- 存 (obs, z, goal, reward, done, info) transitions
- 每个 successful episode 自动 HER 重标记 (achieved_bd -> new_goal)
- 采样时按 her_ratio 混合原始/HER 样本
"""
import numpy as np
import torch


class HERReplayBuffer:
    def __init__(self, capacity=10000, her_k=4, her_ratio=0.8):
        self.capacity = capacity
        self.her_k = her_k  # 多少个 future 状态作为 achieved goal
        self.her_ratio = her_ratio
        self.buffer = []
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add_episode(self, obs, z, goal, reward, done, info):
        """Add a single transition (single-step episode).

        Args:
            obs: (obs_dim,) array
            z: (z_dim,) array
            goal: scalar int (target BD cell)
            reward: float
            done: bool
            info: dict with keys: collides, min_dist, bd_actual
        """
        transition = {
            'obs': obs,
            'z': z,
            'goal': goal,
            'reward': reward,
            'done': done,
            'info': info,
        }
        self._add(transition)

        # HER: 如果撞到, 用 achieved_bd 作新 goal 重标记
        if info.get('collides', False) and info.get('bd_actual', -1) >= 0:
            achieved_bd = info['bd_actual']
            # 单步 episode: 只有 1 个 transition, HER 重标记它
            her_transition = {
                'obs': obs.copy() if isinstance(obs, np.ndarray) else obs,
                'z': z.copy() if isinstance(z, np.ndarray) else z,
                'goal': achieved_bd,
                'reward': 1.0 + max(0.0, 1.0 - info.get('min_dist', 3.0) / 3.0),
                'done': True,  # single-step, done
                'info': {**info, 'her': True},
            }
            self._add(her_transition)

    def _add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample batch_size transitions.

        按 her_ratio 决定 HER 样本比例 (但因为 add 时已 mix, 实际就是均匀采样).
        """
        if self.size == 0:
            return None
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = [self.buffer[i] for i in indices]
        return self._collate(batch)

    def _collate(self, batch):
        """Convert list of dicts to batched dict of tensors."""
        out = {
            'obs': torch.tensor(np.stack([t['obs'] for t in batch]), dtype=torch.float32),
            'z': torch.tensor(np.stack([t['z'] for t in batch]), dtype=torch.float32),
            'goal': torch.tensor([t['goal'] for t in batch], dtype=torch.long),
            'reward': torch.tensor([t['reward'] for t in batch], dtype=torch.float32),
            'done': torch.tensor([t['done'] for t in batch], dtype=torch.float32),
        }
        return out

    def her_stats(self):
        """Return fraction of buffer that's HER."""
        if self.size == 0:
            return 0.0
        her_count = sum(1 for t in self.buffer[:self.size] if t['info'].get('her', False))
        return her_count / self.size
