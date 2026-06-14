"""Experiment 1: Prior Perturbation Probe on val subseq 60.

按 docs/exp1_prior_perturbation_plan.md 执行:
- 6 档 sigma (1, 3, 5, 7, 9, 11)
- 128 sample/档
- 单种子 42
- 只做 prior
- 输出 4 个可视化 + 2 个 json
"""
import os, sys, json
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from datasets.utils import NUSC_BIKE_PARAMS
from rl.prior_perturbation import probe_prior_sigma, summarize_results

device = get_device()
print(f"Device: {device}")

OUT = "./out/prior_perturbation"
os.makedirs(OUT, exist_ok=True)

print("Loading scene (val subseq 60)...")
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
atk_idx = 1  # NA=2, only non-ego

print(f"Scene: NA={NA}, map_idx={mi.item()}")

print("Loading model...")
model = TrafficModel(4, 12, 256, len(dataset.categories), output_bicycle=True,
                     conv_kernel_list=[7,5,5,3,3,3],
                     conv_stride_list=[2,2,2,2,2,2],
                     conv_filter_list=[16,32,32,64,128,128]).to(device) \
    if False else TrafficModel(4, 12, 256, len(dataset.categories), output_bicycle=True,
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
print(f"Prior mu shape: {prior_mu.shape}, var shape: {prior_var.shape}")
print(f"Prior mu range: [{prior_mu.min().item():.3f}, {prior_mu.max().item():.3f}]")
print(f"Prior sigma range: [{prior_var.sqrt().min().item():.3f}, {prior_var.sqrt().max().item():.3f}]")

print("\n" + "="*60)
print("Running prior perturbation probe...")
print("="*60)
SIGMAS = (1, 3, 5, 7, 9, 11)
N_SAMPLES = 128

results = probe_prior_sigma(
    sg, mi, map_env, model, embed_info,
    ego_mask, atk_idx,
    sigmas=SIGMAS, n_samples=N_SAMPLES, seed=42,
)

summary = summarize_results(results)

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"{'sigma':>6} | {'coll_rate':>10} | {'md_median':>10} | "
      f"{'md_q25':>8} | {'md_q75':>8} | {'bd_cov':>7} | {'avg_t':>7}")
print("-"*70)
for mult in SIGMAS:
    s = summary[mult]
    avg_t = f"{s['avg_coll_time']:.1f}" if s['avg_coll_time'] is not None else "N/A"
    print(f"{mult:>6} | {s['collision_rate']*100:>9.1f}% | "
          f"{s['min_dist_median']:>10.3f} | {s['min_dist_q25']:>8.3f} | "
          f"{s['min_dist_q75']:>8.3f} | {s['bd_coverage']:>5}/16 | {avg_t:>7}")

# ===== 保存 json =====
with open(os.path.join(OUT, "sigma_stats.json"), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSaved: {OUT}/sigma_stats.json")

with open(os.path.join(OUT, "all_samples.json"), 'w') as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)
print(f"Saved: {OUT}/all_samples.json")

# ===== 可视化 A: BD 散点图 =====
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(SIGMAS)))
for i, mult in enumerate(SIGMAS):
    bds = [(s['bd_idx'] % 4, s['bd_idx'] // 4)
           for s in results[mult]
           if s['collides'] and s['bd_idx'] >= 0]
    if bds:
        xs = [b[0] + 0.5 for b in bds]
        ys = [b[1] + 0.5 for b in bds]
        ax.scatter(xs, ys, c=[colors[i]], s=30, alpha=0.6,
                   label=f'σ={mult}× ({len(bds)} hits)', edgecolors='black', linewidth=0.3)

ax.set_xticks([0.5, 1.5, 2.5, 3.5])
ax.set_xticklabels(['0°(前)', '90°(右)', '180°(后)', '270°(左)'])
ax.set_yticks([0.5, 1.5, 2.5, 3.5])
ax.set_yticklabels(['0°(前)', '90°(右)', '180°(后)', '270°(左)'])
ax.set_xlabel('pos_bin (attack position in ego frame)')
ax.set_ylabel('heading_bin (attack heading in ego frame)')
ax.set_title(f'BD scatter on val subseq 60 (NA=2)\n128 samples × 6 σ levels')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bd_scatter.png"), dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/bd_scatter.png")

# ===== 可视化 B: min_dist 直方图 =====
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx, mult in enumerate(SIGMAS):
    ax = axes[idx // 3, idx % 3]
    all_md = [s['min_dist'] for s in results[mult]]
    coll_md = [s['min_dist'] for s in results[mult] if s['collides']]
    non_coll_md = [s['min_dist'] for s in results[mult] if not s['collides']]

    if non_coll_md:
        ax.hist(non_coll_md, bins=30, alpha=0.5, color='gray', label=f'no-coll ({len(non_coll_md)})')
    if coll_md:
        ax.hist(coll_md, bins=20, alpha=0.7, color='steelblue', label=f'coll ({len(coll_md)})')

    ax.set_title(f'σ={mult}× (coll_rate={summary[mult]["collision_rate"]*100:.1f}%)')
    ax.set_xlabel('min_dist (m)')
    ax.set_ylabel('count')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('min_dist distribution per σ level')
plt.tight_layout()
plt.savefig(os.path.join(OUT, "min_dist_dist.png"), dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/min_dist_dist.png")

# ===== 可视化 C: 碰撞时刻分布 =====
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx, mult in enumerate(SIGMAS):
    ax = axes[idx // 3, idx % 3]
    cts = [s['coll_time'] for s in results[mult]
           if s['collides'] and s['coll_time'] is not None]
    if cts:
        ax.hist(cts, bins=range(13), alpha=0.7, color='coral', edgecolor='black')
    ax.set_title(f'σ={mult}× (n_coll={len(cts)})')
    ax.set_xlabel('collision time step (0-11)')
    ax.set_ylabel('count')
    ax.set_xticks(range(0, 12, 2))
    ax.grid(True, alpha=0.3)

plt.suptitle('collision time distribution per σ level')
plt.tight_layout()
plt.savefig(os.path.join(OUT, "coll_time_dist.png"), dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/coll_time_dist.png")

# ===== 可视化 D: sigma-cell 热力图 =====
heat = np.zeros((len(SIGMAS), 16))
for i, mult in enumerate(SIGMAS):
    for s in results[mult]:
        if s['collides'] and s['bd_idx'] >= 0:
            heat[i, s['bd_idx']] += 1

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
im = ax.imshow(heat, aspect='auto', cmap='YlOrRd')
ax.set_yticks(range(len(SIGMAS)))
ax.set_yticklabels([f'σ={m}×' for m in SIGMAS])
ax.set_xticks(range(16))
ax.set_xticklabels(range(16), rotation=0, fontsize=8)
ax.set_xlabel('BD cell index (pos_bin × 4 + heading_bin)')
ax.set_title('sigma × BD_cell heatmap (collision count, max 128)')
for i in range(len(SIGMAS)):
    for j in range(16):
        if heat[i, j] > 0:
            ax.text(j, i, f'{int(heat[i,j])}', ha='center', va='center',
                    fontsize=7, color='black' if heat[i,j] < heat.max() * 0.6 else 'white')
plt.colorbar(im, ax=ax, label='collision count')
plt.tight_layout()
plt.savefig(os.path.join(OUT, "sigma_cell_heatmap.png"), dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/sigma_cell_heatmap.png")

print("\n" + "="*60)
print("Done.")
print("="*60)
