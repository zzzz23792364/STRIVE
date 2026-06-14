"""Multi-step rollout for GC-SAC+HER.

sub60 单场景的多步 episode 设计:
- 1 episode = 5 步
- 每步 policy 输出 1 个 z (chunk), decode 出 12 步未来
- 拼接 5 个 chunk (首尾相接) → 60 步完整 trajectory
- 但实际只用最后 12 步 (覆盖 ego logger replay 6s)
- 每步有 next state, 可 bootstrap
- HER 跨步: episode 内任何 t 撞到, 把 achieved_bd 作 new goal 重标记

简洁版: 不拼接 5 段, 而是每步 1 z, decode 出 12 步, 取前 1/5 步作为子 trajectory
- step 0: 用 z_0 1-3 步
- step 1: 用 z_1 4-6 步
- step 2: 用 z_2 7-9 步
- step 3: 用 z_3 10-12 步
- step 4: 用 z_4 1-12 步 (完整)

简单版: 每步 1 z, 但只对 1 个完整 z 给 reward
- 减少计算量
- episode 末才给 reward (整 12 步)
"""
import os, sys, json, time
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from datasets.utils import NUSC_BIKE_PARAMS
from rl.prior_perturbation import compute_bd_from_collision, fast_collision_check_vectorized
from rl.mixture_policy import MixtureGaussianPolicy

device = get_device()
print(f"Device: {device}")

OUT = "./out/gc_sac_her"
os.makedirs(OUT, exist_ok=True)

# ===== 加载 =====
print("Loading val subseq 60...")
data_path = "./data/nuscenes/trainval"
map_env = NuScenesMapEnv(data_path, bounds=[-17.0, -38.5, 60.0, 38.5],
                         L=256, W=256,
                         layers=["drivable_area", "carpark_area",
                                 "road_divider", "lane_divider"],
                         device=device)
dataset = NuScenesDataset(data_path, map_env, version="trainval", split="val",
                           categories=["car", "truck"], npast=4, nfuture=12,
                           reduce_cats=False, seq_interval=10,
                           randomize_val=True, val_size=400)
loader = GraphDataLoader(dataset, batch_size=1, shuffle=False,
                          num_workers=0, pin_memory=False)
for i, data in enumerate(loader):
    if i == 60:
        sg, mi = data; sg, mi = sg.to(device), mi.to(device)
        break

NA = sg.future_gt.size(0)
ptr = sg.ptr
ego_mask = torch.zeros(NA, dtype=torch.bool, device=device)
ego_mask[ptr[:-1]] = True
atk_idx = 1

print("Loading model...")
model = TrafficModel(4, 12, 256, len(dataset.categories), output_bicycle=True,
                     conv_kernel_list=[7,5,5,3,3,3],
                     conv_stride_list=[2,2,2,2,2,2],
                     conv_filter_list=[16,32,64,64,128,128]).to(device)
load_state("./model_ckpt/traffic_model.pth", model)
model.set_normalizer(dataset.get_state_normalizer())
model.set_att_normalizer(dataset.get_att_normalizer())
model.set_bicycle_params(NUSC_BIKE_PARAMS)
model.eval()

with torch.no_grad():
    ei = model.embed(sg, mi, map_env)
embed_info = detach_embed_info(ei)
prior_mu, prior_var = ei['prior_out']
sigma_prior = torch.sqrt(prior_var)
norm = model.get_normalizer()

ego_replay = norm.unnormalize(sg.future_gt[ego_mask][:, :, :4])[0]
ego_lw = model.get_att_normalizer().unnormalize(sg.lw[ego_mask])[0]
atk_lw = model.get_att_normalizer().unnormalize(sg.lw[~ego_mask])[0]


def build_obs(embed_info, sg):
    map_feat = embed_info['map_feat']
    past_feat = embed_info['past_feat']
    obs = torch.cat([map_feat, past_feat, embed_info['prior_out'][0],
                     embed_info['prior_out'][1], sg.lw, sg.sem], dim=-1)
    return obs

obs = build_obs(embed_info, sg)
obs_atk = obs[atk_idx:atk_idx+1].cpu().numpy()[0]
OBS_DIM = obs_atk.shape[0]

POS_DIRS = {
    0: np.array([1.0, 0.0]),    # 前
    1: np.array([0.0, 1.0]),    # 右
    2: np.array([-1.0, 0.0]),   # 后
    3: np.array([0.0, -1.0]),   # 左
}
R_TARGET = 6.0
REACHABLE_CELLS = [2, 6, 10, 14]
N_MODES = 4


@torch.no_grad()
def evaluate_z(z_atk):
    """Decode 1 个 z, 评估碰撞. 返回 (collides, min_dist, bd_actual, fut_normalized)."""
    z_atk_t = torch.tensor(z_atk, device=device, dtype=torch.float32).unsqueeze(0)
    ego_z = prior_mu[0].unsqueeze(0)
    z_full = torch.stack([ego_z, z_atk_t], dim=0)
    dec = model.decode_embedding(z_full, embed_info, sg, mi, map_env)
    fut_raw = dec['future_pred']  # NORMALIZED
    fut_world = norm.unnormalize(fut_raw)
    decoded_atk_world = fut_world[atk_idx, 0, :, :4]

    collides, coll_times, min_dists = fast_collision_check_vectorized(
        decoded_atk_world.unsqueeze(0), ego_replay, atk_lw, ego_lw
    )
    collides = bool(collides[0].item())
    ct = int(coll_times[0].item())
    min_dist = float(min_dists[0].item())

    if collides and 0 <= ct < decoded_atk_world.size(0):
        bd_idx, _, _ = compute_bd_from_collision(ego_replay, decoded_atk_world, ct)
    else:
        bd_idx = -1
    return collides, min_dist, bd_idx, fut_raw  # NORMALIZED fut


def shaping_reward(g, bd_actual, min_dist, collides, t):
    """Per-step reward. 接近 target + bonus + 时间奖励."""
    target = POS_DIRS[g // 4] * R_TARGET
    # 没撞到: 用 min_dist 算 proximity
    if not collides or bd_actual < 0:
        proximity = max(0, 1.0 - min_dist / R_TARGET)
        return proximity
    # 撞到且 bd 是 g: 完美
    if bd_actual == g:
        bonus = 1.0
    else:
        bonus = 0.0  # 撞到其他 cell 也算中性 (HER 会重标记)
    return 1.0 - min(min_dist, 3.0) / 3.0 + bonus


# ===== Replay Buffer (支持 HER 跨步) =====
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = []
        self.ptr = 0
        self.size = 0

    def add(self, obs, z, goal, mode_k, reward, done, her=False):
        item = {
            'obs': obs, 'z': z, 'goal': goal, 'mode_k': mode_k,
            'reward': reward, 'done': done, 'her': her,
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def her_frac(self):
        if self.size == 0:
            return 0.0
        return sum(1 for t in self.buffer[:self.size] if t['her']) / self.size

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        b = [self.buffer[i] for i in idx]
        return {
            'obs': torch.tensor(np.stack([t['obs'] for t in b]), dtype=torch.float32),
            'z': torch.tensor(np.stack([t['z'] for t in b]), dtype=torch.float32),
            'goal': torch.tensor([t['goal'] for t in b], dtype=torch.long),
            'mode_k': torch.tensor([t['mode_k'] for t in b], dtype=torch.long),
            'reward': torch.tensor([t['reward'] for t in b], dtype=torch.float32),
            'done': torch.tensor([t['done'] for t in b], dtype=torch.float32),
        }


# ===== Q Network (per-(goal, mode) shared) =====
class QNetwork(nn.Module):
    def __init__(self, obs_dim, goal_dim=16, z_dim=32, hidden=256, n_modes=4):
        super().__init__()
        self.goal_embed = nn.Linear(goal_dim, 32)
        self.mode_embed = nn.Linear(n_modes, 32)  # one-hot
        in_dim = obs_dim + z_dim + 32 + 32
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, z, goal_onehot, mode_onehot):
        g_emb = F.relu(self.goal_embed(goal_onehot))
        m_emb = F.relu(self.mode_embed(mode_onehot))
        x = torch.cat([obs, z, g_emb, m_emb], dim=-1)
        return self.net(x).squeeze(-1)


# ===== Init networks =====
print("Initializing policy + 2 critics + 2 target critics...")
policy = MixtureGaussianPolicy(obs_dim=OBS_DIM, goal_dim=16, z_dim=32, n_modes=N_MODES, hidden=256).to(device)
q1 = QNetwork(obs_dim=OBS_DIM, goal_dim=16, z_dim=32, n_modes=N_MODES, hidden=256).to(device)
q2 = QNetwork(obs_dim=OBS_DIM, goal_dim=16, z_dim=32, n_modes=N_MODES, hidden=256).to(device)
import copy
q1_target = copy.deepcopy(q1)
q2_target = copy.deepcopy(q2)
for p in q1_target.parameters():
    p.requires_grad = False
for p in q2_target.parameters():
    p.requires_grad = False

# ===== Optimizers =====
LR = 5e-5  # 比 v8 更小, 防 Q 爆炸
policy_optim = torch.optim.Adam(policy.parameters(), lr=LR)
q1_optim = torch.optim.Adam(q1.parameters(), lr=LR)
q2_optim = torch.optim.Adam(q2.parameters(), lr=LR)

# Auto-entropy
log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
alpha_optim = torch.optim.Adam([log_alpha], lr=LR)
target_entropy = -3.0  # 更保守: 鼓励小 σ, 减少 chaos

# Hyperparams
gamma = 0.9  # 折扣更狠, 防止跨步污染
tau = 0.005
grad_clip = 0.1  # 极狠 grad clip

# Replay buffer
buffer = ReplayBuffer(capacity=20000)

# ===== 训练循环 =====
N_EPISODES = 1000
N_STEPS = 5  # 多步 episode
BATCH_SIZE = 64
LOG_INTERVAL = 20
UPDATES_PER_EPISODE = 1  # 1 个 update per episode (防爆炸)
CURRICULUM_START_EP = 100

print(f"\nStarting v9: GC-SAC+HER, multi-step (n={N_STEPS}), Mixture K={N_MODES}, {N_EPISODES} ep")

losses_q, losses_pi, losses_alpha, losses_pi_clip = [], [], [], []
episode_rewards = []
hit_per_cell = {b: 0 for b in range(16)}
best_min_dist_per_cell = {b: float('inf') for b in range(16)}
mode_hits = {}  # (goal, mode_k) -> count

start_t = time.time()
rng = np.random.default_rng(42)

for ep in range(N_EPISODES):
    # 1. 选 goal (curriculum)
    if ep < CURRICULUM_START_EP:
        goal = int(rng.integers(0, 16))
    else:
        if rng.random() < 0.7:
            goal = REACHABLE_CELLS[rng.integers(0, 4)]
        else:
            goal = int(rng.integers(0, 16))

    goal_onehot = torch.zeros(1, 16, device=device); goal_onehot[0, goal] = 1.0
    obs_t = torch.tensor(obs_atk, device=device, dtype=torch.float32).unsqueeze(0)

    # 2. Multi-step episode
    ep_rewards = []
    ep_transitions = []  # 暂存, episode 末做 HER
    ep_achieved_bds = []  # 记录每步 achieved_bd
    ep_collides_list = []

    for t in range(N_STEPS):
        with torch.no_grad():
            mode_k, z, log_prob, _ = policy.sample(obs_t, goal_onehot)
        mode_k_v = int(mode_k.item())
        z_np = z.squeeze(0).cpu().numpy()

        collides, min_dist, bd_actual, fut_raw = evaluate_z(z_np)
        ep_achieved_bds.append(bd_actual)
        ep_collides_list.append(collides)

        r = shaping_reward(goal, bd_actual, min_dist, collides, t)
        done = (t == N_STEPS - 1) or collides
        ep_rewards.append(r)

        # 存 transition (s=z, a=z, g, k, r, done)
        # next state = next step's obs (但 obs 固定 sub60, 用 trajectory 最后位置作 s')
        if t < N_STEPS - 1:
            next_obs = obs_atk  # sub60 obs 固定
        else:
            next_obs = obs_atk  # done 时也用 obs (bootstrapping 用 Q)
        buffer.add(obs_atk, z_np, goal, mode_k_v, r, float(done), her=False)
        ep_transitions.append({
            'obs': obs_atk, 'z': z_np, 'goal': goal, 'mode_k': mode_k_v,
            'reward': r, 'done': float(done),
        })

        if collides and bd_actual == goal:
            hit_per_cell[goal] += 1
            if min_dist < best_min_dist_per_cell[goal]:
                best_min_dist_per_cell[goal] = min_dist
            key = (goal, mode_k_v)
            mode_hits[key] = mode_hits.get(key, 0) + 1

    # 3. HER: 跨步 future relabeling
    # 找 episode 中 successful 的 t
    success_t = [t for t, c in enumerate(ep_collides_list) if c]
    if success_t:
        # 对每 successful t, 把 (achieved_bd, k_t) 作 new goal 重标记
        # 用 (achieved_bd, k_t) 是因为 achieved_bd 是实际撞到的 cell
        for t in success_t:
            new_g = ep_achieved_bds[t]
            new_r = 1.0  # achieved_bd 撞到, r=1
            tr = ep_transitions[t]
            # 重标记成 achieved_bd + 原本的 mode_k
            buffer.add(tr['obs'], tr['z'], new_g, tr['mode_k'], new_r,
                        tr['done'], her=True)
            # 对其他 15 个 goal 存 r=proximity (无碰撞)
            for other_g in range(16):
                if other_g != new_g:
                    r_o = max(0, 1.0 - tr['reward'] * 0.5)  # 用 r 做 proximity proxy
                    buffer.add(tr['obs'], tr['z'], other_g, tr['mode_k'],
                                r_o, tr['done'], her=True)

    episode_rewards.append(sum(ep_rewards))

    # 4. SAC update
    if len(buffer) >= BATCH_SIZE:
        for _ in range(UPDATES_PER_EPISODE):
            batch = buffer.sample(BATCH_SIZE)
            obs_b = batch['obs'].to(device)
            z_b = batch['z'].to(device)
            goal_b = batch['goal'].to(device)
            mode_k_b = batch['mode_k'].to(device)
            reward_b = batch['reward'].to(device)
            done_b = batch['done'].to(device)
            goal_onehot_b = F.one_hot(goal_b, num_classes=16).float()
            mode_onehot_b = F.one_hot(mode_k_b, num_classes=N_MODES).float()

            # 4.1 Critic update
            with torch.no_grad():
                # Sample next z from policy
                _, next_z, next_log_prob, _ = policy.sample(obs_b, goal_onehot_b)
                target_q1 = q1_target(obs_b, next_z, goal_onehot_b, mode_onehot_b)
                target_q2 = q2_target(obs_b, next_z, goal_onehot_b, mode_onehot_b)
                target_q = torch.min(target_q1, target_q2)
                # TD target: r + γ * (1-done) * (Q(s', a') - α * log π)
                alpha = log_alpha.exp().detach()
                td_target = reward_b + gamma * (1 - done_b) * (
                    target_q - alpha * next_log_prob
                )
                # 关键: 裁剪 target 防止 Q 爆炸
                td_target = torch.clamp(td_target, -1.0, 2.0)

            q1_pred = q1(obs_b, z_b, goal_onehot_b, mode_onehot_b)
            q2_pred = q2(obs_b, z_b, goal_onehot_b, mode_onehot_b)
            q_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

            q1_optim.zero_grad()
            q2_optim.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(q1.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(q2.parameters(), grad_clip)
            q1_optim.step()
            q2_optim.step()

            # 4.2 Actor update (SAC style)
            new_log_prob, entropy = policy.get_log_prob_entropy(
                obs_b, goal_onehot_b, mode_k_b, z_b
            )
            q1_new = q1(obs_b, z_b, goal_onehot_b, mode_onehot_b)
            q2_new = q2(obs_b, z_b, goal_onehot_b, mode_onehot_b)
            q_new = torch.min(q1_new, q2_new)
            # π_loss = α * log_prob - Q
            pi_loss = (alpha * new_log_prob - q_new).mean()

            policy_optim.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            policy_optim.step()

            # 4.3 Alpha update
            alpha_loss = -(log_alpha * (new_log_prob.detach() + target_entropy)).mean()
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()

            # 4.4 Soft target update
            with torch.no_grad():
                for p, p_t in zip(q1.parameters(), q1_target.parameters()):
                    p_t.data.mul_(1 - tau).add_(tau * p.data)
                for p, p_t in zip(q2.parameters(), q2_target.parameters()):
                    p_t.data.mul_(1 - tau).add_(tau * p.data)

            losses_q.append(q_loss.item())
            losses_pi.append(pi_loss.item())
            losses_alpha.append(alpha_loss.item())

    if (ep + 1) % LOG_INTERVAL == 0:
        elapsed = time.time() - start_t
        recent_r = episode_rewards[-LOG_INTERVAL:] if len(episode_rewards) >= LOG_INTERVAL else episode_rewards
        avg_r = float(np.mean(recent_r)) if recent_r else 0.0
        her_frac = buffer.her_frac()
        n_cells_filled = sum(1 for m in best_min_dist_per_cell.values() if m < float('inf'))
        last_q = np.mean(losses_q[-100:]) if losses_q else 0
        last_pi = np.mean(losses_pi[-100:]) if losses_pi else 0
        alpha_val = log_alpha.exp().item()
        print(f"  ep {ep+1}/{N_EPISODES} ({elapsed:.0f}s): "
              f"avg_R={avg_r:.3f}, her_frac={her_frac:.2f}, "
              f"cells_filled={n_cells_filled}/16, "
              f"Q={last_q:.3f}, pi={last_pi:.3f}, alpha={alpha_val:.3f}")
        for b in range(16):
            if best_min_dist_per_cell[b] < float('inf'):
                cell_modes = [(k, v) for (g, k), v in mode_hits.items() if g == b]
                cell_modes.sort(key=lambda x: -x[1])
                mode_str = f"mode{[k for k,v in cell_modes[:2]]}"
                print(f"    cell {b:2d}: hit={hit_per_cell[b]:2d}, md={best_min_dist_per_cell[b]:.3f}m, {mode_str}")

# ===== Save =====
torch.save(policy.state_dict(), os.path.join(OUT, "policy.pt"))
torch.save(q1.state_dict(), os.path.join(OUT, "q1.pt"))
torch.save(q2.state_dict(), os.path.join(OUT, "q2.pt"))
print(f"\nPolicy + Q1 + Q2 saved: {OUT}/")

with open(os.path.join(OUT, "training_stats.json"), 'w') as f:
    json.dump({
        'n_episodes': N_EPISODES,
        'n_steps_per_ep': N_STEPS,
        'n_modes': N_MODES,
        'best_min_dist_per_cell': {str(k): float(v) for k, v in best_min_dist_per_cell.items()},
        'hit_per_cell': hit_per_cell,
        'mode_hits': {f"{g}_{k}": v for (g, k), v in mode_hits.items()},
        'episode_rewards': episode_rewards,
        'losses_q': losses_q,
        'losses_pi': losses_pi,
        'losses_alpha': losses_alpha,
    }, f, indent=2)

# ===== Viz =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(episode_rewards, alpha=0.4)
if len(episode_rewards) >= 20:
    axes[0].plot(np.convolve(episode_rewards, np.ones(20)/20, mode='valid'), 'r-', linewidth=2)
axes[0].set_title('Episode Reward (v9 multi-step SAC)')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[0].grid(True, alpha=0.3)

axes[1].plot(losses_q, alpha=0.5, label='Q')
axes[1].plot(losses_pi, alpha=0.5, label='pi')
axes[1].plot(losses_alpha, alpha=0.5, label='alpha')
axes[1].set_yscale('symlog')
axes[1].set_title('Losses')
axes[1].set_xlabel('Update step')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

bds = list(range(16))
mds = [best_min_dist_per_cell[b] if best_min_dist_per_cell[b] < 100 else 0 for b in bds]
colors = ['green' if best_min_dist_per_cell[b] < 1.0 else 'red' for b in bds]
axes[2].bar(bds, mds, color=colors)
axes[2].set_title('Best min_dist per cell (v9)')
axes[2].set_xlabel('BD cell')
axes[2].set_ylabel('min_dist (m)')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=120, bbox_inches='tight')
plt.close()
print(f"Curves saved: {OUT}/training_curves.png")

# ===== Eval 4 cells =====
print("\n=== Policy inference: 4 cells (best mode per cell) ===")
policy.eval()
with torch.no_grad():
    obs_t = torch.tensor(obs_atk, device=device, dtype=torch.float32).unsqueeze(0)
    cell_results = {}

    for g in [2, 6, 10, 14]:
        gh = torch.zeros(1, 16, device=device); gh[0, g] = 1.0
        mu, sigma, pi_logits = policy.forward(obs_t, gh)
        best_md = float('inf')
        best_fut = None
        best_mode = -1

        # 试每个 mode + 多次 sample
        for k in range(N_MODES):
            for trial in range(8):
                mu_k = mu[0, k, :]
                sigma_k = sigma[0, k, :]
                z_k = mu_k + sigma_k * 0.5 * torch.randn_like(mu_k)
                z_np = z_k.cpu().numpy()
                collides, min_dist, bd_actual, fut = evaluate_z(z_np)
                if collides and bd_actual == g and min_dist < best_md:
                    best_md = min_dist
                    best_fut = fut
                    best_mode = k

        if best_fut is not None:
            print(f"  goal {g:2d}: mode {best_mode}, md={best_md:.3f}m")
        else:
            # Stochastic fallback
            for trial in range(16):
                mode_k_s, z, _, _ = policy.sample(obs_t, gh, deterministic=False)
                z_np = z.squeeze(0).cpu().numpy()
                collides, min_dist, bd_actual, fut = evaluate_z(z_np)
                if collides and bd_actual == g and min_dist < best_md:
                    best_md = min_dist
                    best_fut = fut
                    best_mode = mode_k_s.item()
                    break
            if best_fut is not None:
                print(f"  goal {g:2d}: mode {best_mode} (stoch), md={best_md:.3f}m")
            else:
                print(f"  goal {g:2d}: miss")

        cell_results[g] = (best_md, best_fut, best_mode)

# Save
with open(os.path.join(OUT, "policy_eval.json"), 'w') as f:
    json.dump([{"goal": g, "mode_k": m, "min_dist": md}
               for g, (md, _, m) in cell_results.items()], f, indent=2)

# Viz
VIZ_BOUNDS = [-60.0, -60.0, 60.0, 60.0]
car_colors = nutils.get_adv_coloring(NA, atk_idx, 0)

with torch.no_grad():
    dec_prior = model.decode_embedding(prior_mu, embed_info, sg, mi, map_env)
    fut_prior = dec_prior["future_pred"]
nutils.viz_scene_graph(
    sg, mi, map_env, 0, os.path.join(OUT, "prior_before"),
    norm, model.get_att_normalizer(),
    future_pred=fut_prior,
    viz_traj=True, make_video=False, show_gt=False,
    viz_bounds=VIZ_BOUNDS, center_viz=True,
    car_colors=car_colors,
)
print(f"Saved: {OUT}/prior_before.png")

for g, (md, fut, mode_k) in cell_results.items():
    if fut is None:
        continue
    nutils.viz_scene_graph(
        sg, mi, map_env, 0, os.path.join(OUT, f"cell_{g:02d}_after"),
        norm, model.get_att_normalizer(),
        future_pred=fut,
        viz_traj=True, make_video=False, show_gt=False,
        viz_bounds=VIZ_BOUNDS, center_viz=True,
        car_colors=car_colors,
    )
    print(f"Saved: {OUT}/cell_{g:02d}_after.png (md={md:.3f}m, mode {mode_k})")

print("\nDone.")
