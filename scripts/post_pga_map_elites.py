"""后处理 PGA-MAP-Elites 训练结果: 重新跑 archive + policy 推理 + 保存 csv/可视化."""
import os, sys, json
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
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

OUT = "./out/pga_map_elites"
os.makedirs(OUT, exist_ok=True)

# ===== 加载 =====
print("Loading scene...")
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

print("Embedding...")
with torch.no_grad():
    ei = model.embed(sg, mi, map_env)
embed_info = detach_embed_info(ei)
prior_mu, prior_var = ei['prior_out']
sigma_prior = torch.sqrt(prior_var)
norm = model.get_normalizer()

ego_replay = norm.unnormalize(sg.future_gt[ego_mask][:, :, :4])[0]
ego_lw = model.get_att_normalizer().unnormalize(sg.lw[ego_mask])[0]
atk_lw = model.get_att_normalizer().unnormalize(sg.lw[~ego_mask])[0]


# ===== 评估函数 =====
@torch.no_grad()
def evaluate_z(z_atk_batch):
    BS = z_atk_batch.size(0)
    ego_z = prior_mu[0].unsqueeze(0).expand(BS, 32)
    z_full = torch.stack([ego_z, z_atk_batch], dim=0)  # (NA=2, BS, 32)

    dec = model.decode_embedding(z_full, embed_info, sg, mi, map_env)
    fut = norm.unnormalize(dec['future_pred'])  # (NA, BS, FT, 4)

    decoded_atk = fut[atk_idx, :, :, :4]  # (BS, FT, 4)

    collides, coll_times, min_dists = fast_collision_check_vectorized(
        decoded_atk, ego_replay, atk_lw, ego_lw
    )

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


# ===== 重新跑 PGA-ME 拿 archive (因为之前的 archive 在内存里没保存) =====
print("\nRe-running PGA-ME to save archive (fast: 100 iter)...")
solution_dim = 32
dims = (4, 4)
ranges = ((0, 4), (0, 4))
archive = GridArchive(solution_dim=solution_dim, dims=dims, ranges=ranges, seed=42)
emitters = []
for pos_bin in range(4):
    for h_bin in range(4):
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
scheduler = Scheduler(archive, emitters)

qd_history = []
cov_history = []
import time
t0 = time.time()
for it in range(200):
    sols = scheduler.ask()
    z_batch = torch.tensor(sols, device=device, dtype=torch.float32)
    collides, min_dists, bd_indices, _, _ = evaluate_z(z_batch)

    objectives = -min_dists.astype(np.float64)
    measures = np.full((len(sols), 2), -1.0, dtype=np.float32)
    for k in range(len(sols)):
        if bd_indices[k] >= 0:
            measures[k, 0] = bd_indices[k] // 4
            measures[k, 1] = bd_indices[k] % 4
    scheduler.tell(objectives, measures)

    stats = archive.stats
    cov_history.append(float(stats.coverage))
    qd_history.append(float(stats.qd_score))

    if it % 20 == 0:
        print(f"  it {it}: coverage={stats.coverage*100:.1f}%, qd={stats.qd_score:.2f}, t={time.time()-t0:.0f}s")

print(f"\nFinal: coverage={archive.stats.coverage*100:.1f}%, qd={archive.stats.qd_score:.2f}")

# ===== 保存 archive DataFrame =====
df = archive.data(return_type="pandas")
print(f"Archive shape: {df.shape}")
print(df.head())

# 提取每个 cell 的最优 solution
best_per_cell = {}
for _, row in df.iterrows():
    # measures_0 = pos_bin, measures_1 = h_bin
    if row['measures_0'] < 0 or row['measures_1'] < 0:
        continue  # placeholder
    pos_bin = int(row['measures_0'])
    h_bin = int(row['measures_1'])
    bd_idx = pos_bin * 4 + h_bin
    if bd_idx >= 16:
        continue
    obj = row['objective']
    if bd_idx not in best_per_cell or obj > best_per_cell[bd_idx]['objective']:
        sol = np.array([row[f'solution_{i}'] for i in range(32)])
        best_per_cell[bd_idx] = {
            'objective': float(obj),
            'min_dist': float(-obj),  # objective = -min_dist
            'pos_bin': pos_bin,
            'h_bin': h_bin,
            'solution': sol.tolist(),
        }

print(f"\nBest per cell (from saved archive):")
for c in range(16):
    if c in best_per_cell:
        print(f"  cell {c}: min_dist={best_per_cell[c]['min_dist']:.3f}m, "
              f"objective={best_per_cell[c]['objective']:.3f}")
    else:
        print(f"  cell {c}: --")

# 保存 archive.csv
df.to_csv(os.path.join(OUT, "archive.csv"), index=False)
print(f"Saved: {OUT}/archive.csv")

# 保存 best_per_cell.json
with open(os.path.join(OUT, "best_per_cell.json"), 'w') as f:
    out = {str(c): best_per_cell.get(c, {'min_dist': None, 'solution': None})
           for c in range(16)}
    json.dump(out, f, indent=2)
print(f"Saved: {OUT}/best_per_cell.json")

# 保存 training history
with open(os.path.join(OUT, "training_stats.json"), 'w') as f:
    json.dump({
        'n_iters': 200,
        'archive_coverage': float(archive.stats.coverage),
        'qd_score': float(archive.stats.qd_score),
        'qd_history': qd_history,
        'coverage_history': cov_history,
    }, f, indent=2)
print(f"Saved: {OUT}/training_stats.json")

# ===== 可视化: coverage/qd evolution =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(cov_history, 'b-', linewidth=2)
axes[0].set_title('Archive Coverage over Iterations')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Coverage (fraction of 16 cells)')
axes[0].set_ylim([0, 0.4])
axes[0].grid(True, alpha=0.3)

axes[1].plot(qd_history, 'r-', linewidth=2)
axes[1].set_title('QD Score over Iterations')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('QD Score')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/training_curves.png")

# ===== 可视化: 4 cell 可视化 =====
print("\n=== Viz 4 filled cells ===")
class ConditionalGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, bd_dim=16, z_dim=32, hidden=256):
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

obs_dim = 196  # hardcode (we know)
policy = ConditionalGaussianPolicy(obs_dim=obs_dim, bd_dim=16, z_dim=32).to(device)
policy.load_state_dict(torch.load(os.path.join(OUT, "policy.pt"), map_location=device))
policy.eval()

# Load obs
def build_obs(embed_info, sg):
    map_feat = embed_info['map_feat']
    past_feat = embed_info['past_feat']
    obs = torch.cat([map_feat, past_feat, embed_info['prior_out'][0],
                     embed_info['prior_out'][1], sg.lw, sg.sem], dim=-1)
    return obs

obs = build_obs(embed_info, sg)
obs_atk = obs[atk_idx:atk_idx+1]

VIZ_BOUNDS = [-60.0, -60.0, 60.0, 60.0]
car_colors = nutils.get_adv_coloring(NA, atk_idx, 0)

# Prior baseline
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

# 4 cell: use best z from archive
for c in [2, 6, 10, 14]:
    if c not in best_per_cell:
        print(f"  cell {c}: not filled, skip viz")
        continue
    z_atk = torch.tensor(best_per_cell[c]['solution'], device=device, dtype=torch.float32)
    z_atk_batch = z_atk.unsqueeze(0)  # (1, 32)
    ego_z = prior_mu[0].unsqueeze(0)  # (1, 32)
    z_full = torch.stack([ego_z, z_atk_batch], dim=0)  # (NA=2, 1, 32)

    with torch.no_grad():
        dec = model.decode_embedding(z_full, embed_info, sg, mi, map_env)
        fut = dec["future_pred"]  # (NA=2, 1, FT, 4) NORMALIZED

    out_prefix = os.path.join(OUT, f"cell_{c}_after")
    nutils.viz_scene_graph(
        sg, mi, map_env, 0, out_prefix,
        norm, model.get_att_normalizer(),
        future_pred=fut,
        viz_traj=True, make_video=False, show_gt=False,
        viz_bounds=VIZ_BOUNDS, center_viz=True,
        car_colors=car_colors,
    )
    print(f"  cell {c}: md={best_per_cell[c]['min_dist']:.3f}m -> {out_prefix}.png")

print("\nDone.")
