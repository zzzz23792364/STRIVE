"""PGA-MAP-Elites for sub60 multi-solution.

基于 pyribs (成熟 QD 库) 实现:
- GridArchive: 16 cells (4x4 BD grid)
- EvolutionStrategyEmitter: 多个 CMA-ES emitter 探索
- GeneticAlgorithmEmitter: GA crossover + mutation 跨 cell
- Scheduler: 调度 emitter 产出 batch

Policy 网络 (ConditionalGaussianPolicy) 作为 generator:
- 输入: scene_obs (192) + bd_idx_onehot (16) = 208 维
- 输出: z (32 维) + sigma (32 维)

训练循环:
  1. Emitter 产出 K 个 z
  2. decode z -> fut -> 计算 (min_dist, bd_idx)
  3. 写入 archive (按 bd_idx 索引)
  4. PPO update policy 网络 (RL signal: archive 添加成功 +1, 失败 0)
"""
import os, sys, json, time
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, GeneticAlgorithmEmitter
from ribs.schedulers import Scheduler

from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from datasets.utils import NUSC_BIKE_PARAMS
from rl.prior_perturbation import compute_bd_from_collision, fast_collision_check_vectorized

device = get_device()
print(f"Device: {device}")

OUT = "./out/pga_map_elites"
os.makedirs(OUT, exist_ok=True)

# ===== 加载场景 (sub60) =====
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

print("Embedding scene...")
with torch.no_grad():
    ei = model.embed(sg, mi, map_env)
embed_info = detach_embed_info(ei)
prior_mu, prior_var = ei['prior_out']
sigma_prior = torch.sqrt(prior_var)
norm = model.get_normalizer()

ego_replay = norm.unnormalize(sg.future_gt[ego_mask][:, :, :4])[0]
ego_lw = model.get_att_normalizer().unnormalize(sg.lw[ego_mask])[0]
atk_lw = model.get_att_normalizer().unnormalize(sg.lw[~ego_mask])[0]

# ===== Policy 网络 =====
class ConditionalGaussianPolicy(nn.Module):
    """输入 obs (192) + bd_onehot (16) -> z (32) + sigma (32)"""
    def __init__(self, obs_dim=192, bd_dim=16, z_dim=32, hidden=256):
        super().__init__()
        self.bd_embed = nn.Linear(bd_dim, 32)
        in_dim = obs_dim + 32
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, z_dim)
        self.logsig_head = nn.Linear(hidden, z_dim)

    def forward(self, obs, bd_onehot):
        bd_emb = F.relu(self.bd_embed(bd_onehot))
        x = torch.cat([obs, bd_emb], dim=-1)
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_sig = torch.clamp(self.logsig_head(h), -5, 2)
        return mu, log_sig.exp()


# Build obs (复用 build_obs in train_phase1.py)
def build_obs(embed_info, sg):
    map_feat = embed_info['map_feat']
    past_feat = embed_info['past_feat']
    obs = torch.cat([map_feat, past_feat, embed_info['prior_out'][0],
                     embed_info['prior_out'][1], sg.lw, sg.sem], dim=-1)
    return obs


obs = build_obs(embed_info, sg)  # (NA, ?)
obs_atk = obs[atk_idx:atk_idx+1]  # (1, ?)
OBS_DIM = obs_atk.size(-1)
print(f"obs_atk shape: {obs_atk.shape}, obs_dim={OBS_DIM}")

policy = ConditionalGaussianPolicy(obs_dim=OBS_DIM, bd_dim=16, z_dim=32).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)


# ===== 评估函数: decode + BD 计算 =====
@torch.no_grad()
def evaluate_z(z_atk_batch):
    """给定 atk z batch (BS, 32), 评估 (collides, min_dist, bd_idx)"""
    BS = z_atk_batch.size(0)
    # 构建 z_full shape (NA, BS, 32) - decoder 期望这个 shape
    # 用 stack 构造, 避免 expand 的 broadcast view 写入问题
    ego_z = prior_mu[0].unsqueeze(0).expand(BS, 32)  # (BS, 32) - ego 槽位
    z_full = torch.stack([ego_z, z_atk_batch], dim=0)  # (NA=2, BS, 32)

    dec = model.decode_embedding(z_full, embed_info, sg, mi, map_env)
    fut = norm.unnormalize(dec['future_pred'])  # (NA, BS, FT, 4)

    decoded_atk = fut[atk_idx, :, :, :4]  # (BS, FT, 4)

    # Collision check
    collides, coll_times, min_dists = fast_collision_check_vectorized(
        decoded_atk, ego_replay, atk_lw, ego_lw
    )

    # BD 计算 (只对撞到的)
    bd_indices = np.full(BS, -1, dtype=np.int64)
    pos_angles = np.zeros(BS)
    head_angles = np.zeros(BS)
    for k in range(BS):
        if collides[k]:
            ct = int(coll_times[k].item())
            if 0 <= ct < decoded_atk.size(1):
                bd_idx, pa, ha = compute_bd_from_collision(
                    ego_replay, decoded_atk[k], ct
                )
                bd_indices[k] = bd_idx
                pos_angles[k] = pa if pa is not None else 0.0
                head_angles[k] = ha if ha is not None else 0.0

    return collides.cpu().numpy(), min_dists.cpu().numpy(), bd_indices, pos_angles, head_angles


# ===== pyribs Archive + Emitters =====
# BD 是 16 离散 cell, 4x4 grid
solution_dim = 32  # z 维度
dims = (4, 4)  # (pos_bin, heading_bin)
ranges = ((0, 4), (0, 4))  # 0-3 离散

# 用 Evolution Strategy (CMA-ES) emitter: 16 个独立 emitter
# 每个 emitter 探索不同 BD cell
archive = GridArchive(
    solution_dim=solution_dim,
    dims=dims,
    ranges=ranges,
    seed=42,
)

# 16 emitters, 每个对应一个 BD cell
emitters = []
for pos_bin in range(4):
    for h_bin in range(4):
        bd_idx = pos_bin * 4 + h_bin
        # 初始解: z = prior_mu[atk_idx] + sigma_prior[atk_idx] * small_noise
        # + 方向偏向: 让初始指向目标 BD cell 的物理位置
        init_z = prior_mu[atk_idx].cpu().numpy() + sigma_prior[atk_idx].cpu().numpy() * np.random.randn(32) * 0.3
        emitter = EvolutionStrategyEmitter(
            archive,
            x0=init_z,
            sigma0=2.0,
            ranker="2imp",
            es="cma_es",
            batch_size=8,
        )
        emitters.append(emitter)

# 调度器
scheduler = Scheduler(archive, emitters)

print(f"\nArchive: {dims[0]}x{dims[1]} = {dims[0]*dims[1]} cells")
print(f"Emitters: {len(emitters)} (one per cell)")

# ===== 训练循环 =====
N_ITERS = 200
K_PER_ITER = 128  # 每次 scheduler.tell() 之前 ask() 多少次

print(f"\nStarting PGA-MAP-Elites: {N_ITERS} iters")
best_md_per_cell = np.full(16, np.inf)
policy_losses = []

start_t = time.time()
for it in range(N_ITERS):
    # 1. Scheduler 产出 K 个 solution
    solutions = scheduler.ask()  # (K, solution_dim) numpy

    # 2. 评估每个 solution
    z_batch = torch.tensor(solutions, device=device, dtype=torch.float32)
    collides, min_dists, bd_indices, pos_angles, head_angles = evaluate_z(z_batch)

    # 3. 写入 archive
    # - 撞到的: objective = -min_dist (higher = better), measures = (pos_bin, h_bin)
    # - 未撞到的: measure 设为 -1 (pyribs 会忽略, 不写入 archive)
    objectives = -min_dists.astype(np.float64)
    measures = np.full((len(solutions), 2), -1.0, dtype=np.float32)
    for k in range(len(solutions)):
        if bd_indices[k] >= 0:
            measures[k, 0] = bd_indices[k] // 4  # pos_bin
            measures[k, 1] = bd_indices[k] % 4   # h_bin

    scheduler.tell(objectives, measures)

    # 4. 统计
    archive_stats = archive.stats
    coverage = archive_stats.coverage
    qd_score = archive_stats.qd_score

    for k in range(len(solutions)):
        if bd_indices[k] >= 0 and min_dists[k] < best_md_per_cell[bd_indices[k]]:
            best_md_per_cell[bd_indices[k]] = min_dists[k]

    # 5. Policy update (RL signal: 写入成功的 sample 训练 policy 模仿)
    # 简化: 用 PPO-style 行为克隆
    # 收集 (z, success) 配对
    policy_loss = 0.0
    if it % 5 == 0:  # 每 5 iter 训一次
        success_mask = collides & (bd_indices >= 0)
        if success_mask.sum() > 0:
            succ_z = z_batch[success_mask]
            succ_bd = bd_indices[success_mask]

            # 训 policy: 输入 (obs, bd_onehot) -> 输出 mu 接近 succ_z
            bd_onehots = F.one_hot(torch.tensor(succ_bd, device=device), num_classes=16).float()
            obs_rep = obs_atk.expand(succ_z.size(0), -1)

            pred_mu, pred_sig = policy(obs_rep, bd_onehots)
            # MSE loss 模仿成功 z
            policy_loss = F.mse_loss(pred_mu, succ_z)
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
    policy_losses.append(policy_loss.item() if torch.is_tensor(policy_loss) else 0.0)

    # 6. Print progress
    if it % 10 == 0 or it == N_ITERS - 1:
        elapsed = time.time() - start_t
        n_cells_filled = sum(1 for m in best_md_per_cell if m < np.inf)
        n_collided = sum(1 for c in collides if c)
        print(f"  it {it}/{N_ITERS} ({elapsed:.0f}s): "
              f"coverage={coverage*100:.1f}%, qd={qd_score:.2f}, "
              f"cells_filled={n_cells_filled}/16, "
              f"this_iter_collided={n_collided}/{len(solutions)}, "
              f"policy_loss={policy_losses[-1]:.4f}")

print("\n=== Final Results ===")
print(f"Archive coverage: {archive.stats.coverage*100:.1f}%")
print(f"QD score: {archive.stats.qd_score:.2f}")
print("\nBest min_dist per cell:")
for c in range(16):
    md = best_md_per_cell[c]
    if md < np.inf:
        print(f"  cell {c:2d}: {md:.3f}m")
    else:
        print(f"  cell {c:2d}: -- (not filled)")

# ===== Save policy =====
torch.save(policy.state_dict(), os.path.join(OUT, "policy.pt"))
print(f"\nPolicy saved: {OUT}/policy.pt")

# ===== Save archive =====
import pandas as pd
df = archive.data()
df.to_csv(os.path.join(OUT, "archive.csv"), index=False)
print(f"Archive saved: {OUT}/archive.csv")

# ===== Final viz: 用 policy 推理 16 cell 找最优 =====
print("\n=== Policy inference: 16 cells ===")
policy.eval()
with torch.no_grad():
    obs_rep = obs_atk.expand(16, -1)
    bd_all = F.one_hot(torch.arange(16, device=device), num_classes=16).float()
    for trial in range(8):  # 每个 cell 试 8 次, 选最佳
        mus, sigs = policy(obs_rep, bd_all)
        z_trials = mus + sigs * torch.randn_like(mus)
        collides, min_dists, bd_idxs, _, _ = evaluate_z(z_trials)
        for c in range(16):
            if collides[c] and min_dists[c] < best_md_per_cell[c]:
                best_md_per_cell[c] = min_dists[c]

print("After policy inference:")
for c in range(16):
    md = best_md_per_cell[c]
    if md < np.inf:
        print(f"  cell {c:2d}: {md:.3f}m")
    else:
        print(f"  cell {c:2d}: -- (not filled)")

# Save final stats
final_stats = {
    "n_iters": N_ITERS,
    "archive_coverage": float(archive.stats.coverage),
    "qd_score": float(archive.stats.qd_score),
    "best_min_dist_per_cell": best_md_per_cell.tolist(),
    "policy_losses": policy_losses,
}
with open(os.path.join(OUT, "training_stats.json"), "w") as f:
    json.dump(final_stats, f, indent=2)
print(f"\nStats saved: {OUT}/training_stats.json")
