"""Visualize 4 best collision samples (one per BD cell) on val subseq 60.

基于 docs/exp1_prior_perturbation_results.md 找出的 4 cell 最佳样本,
重 decode, 画 BEV 轨迹图 (2x2 + 单图).

快: 只 decode 4 个 z (每个 cell 1 个), 不重跑整个实验.
"""
import os, sys, json
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from datasets.utils import NUSC_BIKE_PARAMS

device = get_device()

OUT = "./out/prior_perturbation"
os.makedirs(OUT, exist_ok=True)

# ===== 加载场景 =====
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

prior_mu, prior_var = ei['prior_out']  # (NA, 32)
sigma_prior = torch.sqrt(prior_var)   # (NA, 32)

norm = model.get_normalizer()

# ===== 4 cell 最佳样本 (sigma, sample_id) =====
# 跨 sigma 选 min_dist 最小的
with open(os.path.join(OUT, "all_samples.json")) as f:
    all_data = json.load(f)

best_per_cell = {}
for mult, samples in all_data.items():
    for s in samples:
        if s['collides'] and s['bd_idx'] >= 0:
            c = s['bd_idx']
            if c not in best_per_cell or s['min_dist'] < best_per_cell[c]['min_dist']:
                best_per_cell[c] = {**s, 'sigma_str': mult}

print("\nBest per cell:")
for c in sorted(best_per_cell):
    s = best_per_cell[c]
    print(f"  cell {c}: sigma={s['sigma_str']}×, sample_id={s['sample_id']}, "
          f"md={s['min_dist']:.3f}m")

# Cell 名称 (基于 docs/bd_design_spec.md)
CELL_NAMES = {
    2: "Cell 2: Head-On (pos=front, h=back)",
    6: "Cell 6: T-bone from right (pos=right, h=back)",
    10: "Cell 10: Rear-end (pos=behind, h=back)",
    14: "Cell 14: T-bone from left (pos=left, h=back)",
}

# ===== 提取 ego_replay + prior_decode =====
ego_replay = norm.unnormalize(sg.future_gt[ego_mask][:, :, :4])[0]  # (FT, 4)
ego_lw = model.get_att_normalizer().unnormalize(sg.lw[ego_mask])[0]
atk_lw = model.get_att_normalizer().unnormalize(sg.lw[~ego_mask])[0]

# Prior (未扰动) decode — 用作 baseline
with torch.no_grad():
    dec_prior = model.decode_embedding(prior_mu, embed_info, sg, mi, map_env)
    fut_prior = norm.unnormalize(dec_prior['future_pred'])  # (NA, FT, 4)
prior_atk = fut_prior[atk_idx, :, :4]   # (FT, 4)
prior_ego = fut_prior[0, :, :4]        # (FT, 4)

# Past 轨迹 (4 步) for context
past = sg.past.cpu().numpy()  # (NA, PT, 4)
past_ego = past[0, :, :4]     # (PT, 4)
past_atk = past[1, :, :4]     # (PT, 4)

# ===== 重现 4 个 z 并 decode =====
print("\nReproducing 4 z and decoding...")

# 关键: 用 seed=42 重新生成同样的 randn
torch.manual_seed(42)

cell_results = {}
for c in sorted(best_per_cell.keys()):
    s = best_per_cell[c]
    mult = float(s['sigma_str'])
    sample_id = s['sample_id']

    # 重现 sample_id 个 sample (sample_id 之前的都被消耗, 所以要 skip 掉)
    # 这与原实验一致: 每 sigma 重新从 sample 0 开始
    eps = torch.randn(NA, prior_mu.size(-1), device=device) * mult
    # skip 掉 sample_id 之前的所有 sample
    for _ in range(sample_id):
        _ = torch.randn(NA, prior_mu.size(-1), device=device) * mult
    eps_target = torch.randn(NA, prior_mu.size(-1), device=device) * mult
    z = prior_mu + sigma_prior * eps_target  # (NA, 32)

    with torch.no_grad():
        dec = model.decode_embedding(z, embed_info, sg, mi, map_env)
        fut = norm.unnormalize(dec['future_pred'])  # (NA, FT, 4)

    decoded_atk = fut[atk_idx, :, :4]
    cell_results[c] = {
        'z': z.cpu(),
        'decoded_atk': decoded_atk.cpu(),
        'mult': mult,
        'sample_id': sample_id,
        'min_dist': s['min_dist'],
        'pos_angle': s['pos_angle'],
        'heading_angle': s['heading_angle'],
        'coll_time': s['coll_time'],
    }
    print(f"  cell {c}: z generated, md_actual={s['min_dist']:.3f}m")

ego_replay_np = ego_replay.cpu().numpy()
prior_atk_np = prior_atk.cpu().numpy()
prior_ego_np = prior_ego.cpu().numpy()
ego_lw_np = ego_lw.cpu().numpy()
atk_lw_np = atk_lw.cpu().numpy()


def draw_box(ax, x, y, hx, hy, lw, color, lw_line=2, alpha=0.8):
    """Draw a vehicle bounding box given (x, y, hx, hy) heading vector and (length, width)."""
    L, W = lw
    # heading 方向 (归一化)
    h = np.array([hx, hy])
    h = h / (np.linalg.norm(h) + 1e-8)
    # 垂直方向
    n = np.array([-h[1], h[0]])
    # 4 个角
    c = np.array([x, y])
    corners = c + (L / 2) * h + (W / 2) * n
    corners = np.array([
        corners,
        c + (L / 2) * h - (W / 2) * n,
        c - (L / 2) * h - (W / 2) * n,
        c - (L / 2) * h + (W / 2) * n,
    ])
    ax.add_patch(plt.Polygon(corners, closed=True, fill=True,
                              facecolor=color, edgecolor='black',
                              linewidth=lw_line, alpha=alpha))


def plot_cell(ax, c, info, ego_replay, prior_ego, prior_atk, decoded_atk,
              past_ego, past_atk, ego_lw, atk_lw, draw_map_bg=True):
    """Draw one cell's trajectories on ax. 每个时刻都画 vehicle box, 透明度随时间渐变."""
    # Map 背景
    if draw_map_bg:
        try:
            ax.set_facecolor('#e8e8e8')  # light gray base
        except Exception:
            pass

    ct = info['coll_time']
    FT = decoded_atk.shape[0]

    # ===== 1) Past (灰色虚线, 起点 box) =====
    ax.plot(past_ego[:, 0], past_ego[:, 1], 'k-', linewidth=1.5, alpha=0.3)
    ax.plot(past_atk[:, 0], past_atk[:, 1], 'k-', linewidth=1.5, alpha=0.3)
    # Past 终点 box
    draw_box(ax, past_ego[-1, 0], past_ego[-1, 1], past_ego[-1, 2], past_ego[-1, 3],
             ego_lw, 'lightblue', lw_line=1.5, alpha=0.8)
    draw_box(ax, past_atk[-1, 0], past_atk[-1, 1], past_atk[-1, 2], past_atk[-1, 3],
             atk_lw, 'lightyellow', lw_line=1.5, alpha=0.8)

    # ===== 2) Prior (虚线, 较暗) =====
    ax.plot(prior_ego[:, 0], prior_ego[:, 1], 'b--', linewidth=1.5, alpha=0.4,
            label='ego prior (no perturb)')
    ax.plot(prior_atk[:, 0], prior_atk[:, 1], 'c--', linewidth=1.5, alpha=0.4,
            label='atk prior (no perturb)')

    # ===== 3) 所有时刻画 box (透明度渐变: 越靠后越不透明) =====
    for t in range(FT):
        # 透明度: 0.15 -> 0.95
        alpha = 0.15 + 0.80 * (t / max(FT - 1, 1))
        # 碰撞时刻最显眼
        is_collide_t = (t == ct)
        if is_collide_t:
            alpha = 1.0
        # 边线
        lw_line = 2.5 if is_collide_t else 0.8
        # 颜色: 越靠后越深 (白->灰, 浅红->红)
        ego_color = plt.cm.Greys(0.3 + 0.6 * (t / max(FT - 1, 1)))
        atk_color = plt.cm.Reds(0.2 + 0.7 * (t / max(FT - 1, 1)))

        draw_box(ax, ego_replay[t, 0], ego_replay[t, 1],
                 ego_replay[t, 2], ego_replay[t, 3],
                 ego_lw, ego_color, lw_line=lw_line, alpha=alpha)
        draw_box(ax, decoded_atk[t, 0], decoded_atk[t, 1],
                 decoded_atk[t, 2], decoded_atk[t, 3],
                 atk_lw, atk_color, lw_line=lw_line, alpha=alpha)

    # ===== 4) 轨迹线 (在 box 之上) =====
    ax.plot(ego_replay[:, 0], ego_replay[:, 1], 'k-', linewidth=2.0,
            label='ego logger replay', alpha=0.7, zorder=5)
    ax.plot(decoded_atk[:, 0], decoded_atk[:, 1], 'r-', linewidth=2.0,
            label='atk decoded (perturbed)', alpha=0.7, zorder=5)

    # ===== 5) 起点和终点标记 =====
    ax.plot(ego_replay[0, 0], ego_replay[0, 1], 'k^', markersize=10, zorder=20,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(decoded_atk[0, 0], decoded_atk[0, 1], 'r^', markersize=10, zorder=20,
            markeredgecolor='white', markeredgewidth=1.5)

    # ===== 6) 碰撞时刻黄星 + min_dist 标注 =====
    if ct is not None and 0 <= ct < FT:
        # 两车中点
        cx = (decoded_atk[ct, 0] + ego_replay[ct, 0]) / 2
        cy = (decoded_atk[ct, 1] + ego_replay[ct, 1]) / 2
        # 距离线
        ax.annotate('', xy=(decoded_atk[ct, 0], decoded_atk[ct, 1]),
                    xytext=(ego_replay[ct, 0], ego_replay[ct, 1]),
                    arrowprops=dict(arrowstyle='<->', color='orange', lw=2.5,
                                    shrinkA=0, shrinkB=0))
        # 距离文字
        ax.text(cx, cy + 0.8, f'  min_dist={info["min_dist"]:.3f}m at t={ct}',
                fontsize=12, color='black', fontweight='bold',
                bbox=dict(facecolor='yellow', edgecolor='orange',
                          boxstyle='round,pad=0.3'), zorder=25)

    # ===== 7) 标题 (英文避免乱码) =====
    title = (f"{CELL_NAMES[c]}\n"
             f"sigma={info['mult']}x, sample_id={info['sample_id']}, "
             f"min_dist={info['min_dist']:.3f}m, t={ct}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


# ===== 画 2x2 对比图 =====
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
for idx, c in enumerate(sorted(cell_results.keys())):
    ax = axes[idx // 2, idx % 2]
    info = cell_results[c]
    plot_cell(ax, c, info,
              ego_replay_np, prior_ego_np, prior_atk_np,
              info['decoded_atk'].numpy(),
              past_ego, past_atk, ego_lw_np, atk_lw_np)
    ax.legend(loc='upper left', fontsize=8)

# 设置 view bounds (consistent)
all_x = []
all_y = []
for info in cell_results.values():
    arr = info['decoded_atk'].numpy()
    all_x.extend(arr[:, 0].tolist())
    all_y.extend(arr[:, 1].tolist())
all_x.extend(ego_replay_np[:, 0].tolist())
all_y.extend(ego_replay_np[:, 1].tolist())
margin = 5
xmin, xmax = min(all_x) - margin, max(all_x) + margin
ymin, ymax = min(all_y) - margin, max(all_y) + margin
for ax in axes.flat:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

plt.suptitle('4 BD cells: best collision sample per cell\nval subseq 60, NA=2',
             fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "4cells_comparison.png"), dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUT}/4cells_comparison.png")

# ===== 画每 cell 单独 1 张高清 (用新 plot_cell, 包含全时刻 box) =====
for c in sorted(cell_results.keys()):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    info = cell_results[c]
    plot_cell(ax, c, info,
              ego_replay_np, prior_ego_np, prior_atk_np,
              info['decoded_atk'].numpy(),
              past_ego, past_atk, ego_lw_np, atk_lw_np)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    fname = f"cell_{c:02d}_trajectory.png"
    plt.savefig(os.path.join(OUT, fname), dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT}/{fname}")

# ===== 保存 cell metadata (for future reuse) =====
cell_meta = {}
for c, info in cell_results.items():
    cell_meta[c] = {
        'sigma_mult': info['mult'],
        'sample_id': info['sample_id'],
        'min_dist': info['min_dist'],
        'pos_angle': info['pos_angle'],
        'heading_angle': info['heading_angle'],
        'coll_time': info['coll_time'],
        'z_atk': info['z'][atk_idx].tolist(),
    }
with open(os.path.join(OUT, "best_per_cell.json"), 'w') as f:
    json.dump(cell_meta, f, indent=2)
print(f"Saved: {OUT}/best_per_cell.json")

print("\nDone.")
