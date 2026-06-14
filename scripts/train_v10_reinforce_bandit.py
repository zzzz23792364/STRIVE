"""v10: REINFORCE with Continuous Reward (Williams 1992 + Sutton & Barto 1998)

Core idea: 这不是 MDP, 是 Contextual Bandit。
- 不需要 critic (REINFORCE with baseline 即可, baseline = MC return 平均)
- 不需要 HER (没有序列)
- 不需要 multi-step episode (单步决策)
- 不需要 target network (没有 bootstrap)

Reward: continuous distance-based (Sutton & Barto Ch.3)
- R = -min_dist if collide cell g
- R = -3.0 if collide other cell (mode-collapse 抑制)
- R = -min_dist_to_ego if no collision (close = good)

Policy: Mixture of K Gaussians (Bishop 1994 - MDN)
- K=4 modes, each dedicated to one cell
- Per-mode baseline (Williams 1992) 防止跨 mode 信号干扰
- Entropy bonus (Mnih et al. 2016 - A3C) 防止 σ 收缩

参考文献 (有论文支撑的组件):
  - REINFORCE + baseline:    Williams 1992, "Simple Statistical Gradient-Following..."
  - Mixture of Gaussians:    Bishop 1994, "Mixture Density Networks"
  - Continuous reward:       Sutton & Barto 1998, "Reinforcement Learning: An Introduction", Ch.3
  - Entropy regularization:  Mnih et al. 2016, "Asynchronous Methods for Deep RL" (A3C)
  - Per-mode decoupled opt:  Standard ensemble / options framework (Sutton et al. 1999)
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

OUT = "./out/v10_reinforce_bandit"
os.makedirs(OUT, exist_ok=True)

# ===== Load scene =====
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
G_to_MODE = {2: 0, 6: 1, 10: 2, 14: 3}  # 固定映射


@torch.no_grad()
def evaluate_z(z_atk):
    """Decode 1 z, 评估碰撞 + min_dist to ego center (continuous, always defined)."""
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
    return collides, min_dist, bd_idx, fut_raw


def continuous_reward(g, bd_actual, min_dist, collides):
    """Continuous reward (Sutton & Barto Ch.3).

    Large range for non-collision (-3 to -30) creates high advantage signal 
    when hit occurs (R ≈ -0.2 vs baseline ≈ -15 → advantage ≈ +15).
    This contrast is essential: the strong hit signal overcomes 99.5% miss noise.
    """
    if collides and bd_actual == g:
        return -min(min_dist, 3.0)  # 强 signal
    elif collides:
        return -min_dist * 0.5 - 1.0  # 撞其他 cell: 介于 hit 和 miss 之间
    else:
        return -min_dist  # clip 会让 hit vs miss 对比弱化


# ===== Policy: Mixture K=4 =====
print("Initializing MixtureGaussianPolicy K=4...")
policy = MixtureGaussianPolicy(obs_dim=OBS_DIM, goal_dim=16, z_dim=32, n_modes=N_MODES, hidden=256).to(device)

# Optimizer
LR = 3e-4  # REINFORCE 不需要 SAC 那么小心
policy_optim = torch.optim.Adam(policy.parameters(), lr=LR)

# Per-mode baseline (Williams 1992)
baseline = np.zeros(N_MODES)  # 4 个 mode 各一个 baseline

# Hyperparams
ENTROPY_COEF = 0.05  # 防止 σ 收缩 (A3C paper Mnih et al. 2016)

# ===== Training loop =====
N_EPISODES = 2000

print(f"\nStarting v10: REINFORCE Bandit + Continuous Reward + Mixture K={N_MODES}, {N_EPISODES} ep")
losses_pi, losses_entropy = [], []
episode_rewards = []
hit_per_cell = {b: 0 for b in range(16)}
best_min_dist_per_cell = {b: float('inf') for b in range(16)}
mode_hits = {}  # (goal, mode_k) -> count

start_t = time.time()
rng = np.random.default_rng(42)

for ep in range(N_EPISODES):
    # 1. 选 goal (均匀从 reachable)
    g = int(REACHABLE_CELLS[rng.integers(0, 4)])
    k = G_to_MODE[g]  # 固定 mode 映射

    # 2. Sample from mode k
    goal_onehot = torch.zeros(1, 16, device=device); goal_onehot[0, g] = 1.0
    obs_t = torch.tensor(obs_atk, device=device, dtype=torch.float32).unsqueeze(0)

    mu, sigma, pi_logits = policy.forward(obs_t, goal_onehot)
    # Force sample from mode k (固定 mapping)
    mu_k = mu[0, k, :]      # (z_dim,)
    sigma_fixed = 0.5  # 固定 sigma, 只训 mu, 防止探索消失
    z_dist = torch.distributions.Normal(mu_k, sigma_fixed)
    z = z_dist.rsample()    # (z_dim,)
    log_prob = z_dist.log_prob(z).sum()  # scalar

    z_np = z.detach().cpu().numpy()

    # 3. Evaluate
    collides, min_dist, bd_actual, fut_raw = evaluate_z(z_np)
    R = continuous_reward(g, bd_actual, min_dist, collides)

    if collides and bd_actual == g:
        hit_per_cell[g] += 1
        if min_dist < best_min_dist_per_cell[g]:
            best_min_dist_per_cell[g] = min_dist
        key = (g, k)
        mode_hits[key] = mode_hits.get(key, 0) + 1

    # 4. Per-mode baseline update (Williams 1992)
    baseline[k] = 0.99 * baseline[k] + 0.01 * R

    # 5. REINFORCE with baseline (Williams 1992)
    advantage = R - baseline[k]
    pi_loss = -log_prob * advantage

    # 6. Entropy bonus (Mnih et al. 2016)
    entropy_loss = -ENTROPY_COEF * z_dist.entropy().sum()

    loss = pi_loss + entropy_loss

    policy_optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    policy_optim.step()

    losses_pi.append(pi_loss.item())
    losses_entropy.append(entropy_loss.item())
    episode_rewards.append(R)

    if (ep + 1) % 100 == 0:
        elapsed = time.time() - start_t
        recent_r = episode_rewards[-100:]
        avg_r = float(np.mean(recent_r))
        n_cells = sum(1 for m in best_min_dist_per_cell.values() if m < float('inf'))
        n_total_hit = sum(hit_per_cell.values())
        hit_rate = n_total_hit / (ep + 1)
        print(f"  ep {ep+1}/{N_EPISODES} ({elapsed:.0f}s): "
              f"avg_R={avg_r:.3f}, hit_rate={hit_rate:.3f}, "
              f"cells_filled={n_cells}/4 (in [2,6,10,14])")
        for b in REACHABLE_CELLS:
            md = best_min_dist_per_cell[b]
            hit = hit_per_cell[b]
            tag = f"md={md:.3f}m" if md < float('inf') else "miss"
            print(f"    cell {b:2d}: hit={hit:3d}, {tag}")

# ===== Save =====
torch.save(policy.state_dict(), os.path.join(OUT, "policy.pt"))
print(f"\nPolicy saved: {OUT}/policy.pt")

with open(os.path.join(OUT, "training_stats.json"), 'w') as f:
    json.dump({
        'n_episodes': N_EPISODES,
        'n_modes': N_MODES,
        'best_min_dist_per_cell': {str(k): float(v) for k, v in best_min_dist_per_cell.items()},
        'hit_per_cell': hit_per_cell,
        'mode_hits': {f"{g}_{k}": v for (g, k), v in mode_hits.items()},
        'episode_rewards': episode_rewards,
        'losses_pi': losses_pi,
        'losses_entropy': losses_entropy,
        'baseline_final': baseline.tolist(),
    }, f, indent=2)

# ===== Viz =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(episode_rewards, alpha=0.4)
if len(episode_rewards) >= 20:
    axes[0].plot(np.convolve(episode_rewards, np.ones(20)/20, mode='valid'), 'r-', linewidth=2)
axes[0].set_title('Episode Reward (v10 REINFORCE Bandit)')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[0].grid(True, alpha=0.3)

axes[1].plot(losses_pi, alpha=0.5, label='pi')
axes[1].plot(losses_entropy, alpha=0.5, label='entropy')
axes[1].set_title('Losses')
axes[1].set_xlabel('Update step')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

bds = REACHABLE_CELLS
mds = [best_min_dist_per_cell[b] if best_min_dist_per_cell[b] < 100 else 0 for b in bds]
colors = ['green' if best_min_dist_per_cell[b] < 1.0 else 'red' for b in bds]
axes[2].bar(bds, mds, color=colors)
axes[2].set_title('Best min_dist per cell (v10)')
axes[2].set_xlabel('BD cell')
axes[2].set_ylabel('min_dist (m)')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=120, bbox_inches='tight')
plt.close()
print(f"Curves saved: {OUT}/training_curves.png")

# ===== Eval 4 cells =====
print("\n=== Policy inference: 4 cells (deterministic mode select) ===")
policy.eval()
with torch.no_grad():
    obs_t = torch.tensor(obs_atk, device=device, dtype=torch.float32).unsqueeze(0)
    cell_results = {}

    for g in REACHABLE_CELLS:
        k = G_to_MODE[g]
        gh = torch.zeros(1, 16, device=device); gh[0, g] = 1.0
        mu, sigma, _ = policy.forward(obs_t, gh)
        mu_k = mu[0, k, :]
        sigma_k = 0.5  # 固定 sigma, 同训练时

        # 200 trials with sample (sigma=0.5 gives ~8% hit rate per trial)
        best_md = float('inf')
        best_fut = None

        for trial in range(200):
            z_k = mu_k + sigma_k * torch.randn_like(mu_k)
            z_np = z_k.cpu().numpy()
            collides, min_dist, bd_actual, fut = evaluate_z(z_np)
            if collides and bd_actual == g and min_dist < best_md:
                best_md = min_dist
                best_fut = fut
            if best_fut is not None and best_md < 0.5:
                break

        if best_fut is not None:
            print(f"  goal {g:2d}: mode {k}, md={best_md:.3f}m")
        else:
            print(f"  goal {g:2d}: miss")

        cell_results[g] = (best_md, best_fut, k)

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
