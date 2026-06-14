"""Fast viz: reuse nutils.viz_scene_graph to show perturbed trajectories for 4 BD cells on val subseq 60.

复用 STRIVE 自带的 nutils.viz_scene_graph, 它内部已经 unnormalize + 画 map 背景.
每个 cell 输出:
  - prior_before.png: prior (no perturb) - 4 cell 共享
  - cell_NN_after.png:  perturbed (best sample per cell)
"""
import os, sys, json
sys.path.insert(0, "src")

import torch
import numpy as np

from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from datasets.utils import NUSC_BIKE_PARAMS
from rl.prior_perturbation import probe_prior_sigma

device = get_device()

OUT = "./out/prior_perturbation"
os.makedirs(OUT, exist_ok=True)

# ===== 加载场景 + 模型 =====
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
atk_idx = 1  # NA=2

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
norm = model.get_normalizer()

# ===== Probe: 拿到 future_pred 全部样本 =====
print("Running probe_prior_sigma...")
results = probe_prior_sigma(
    sg, mi, map_env, model, embed_info,
    ego_mask, atk_idx,
    sigmas=[1, 3, 5, 7, 9, 11],
    n_samples=128, seed=42, batch_size=32,
)

# ===== 4 cell 最佳样本 =====
with open(os.path.join(OUT, "best_per_cell.json")) as f:
    cell_meta = json.load(f)

cells = sorted(cell_meta.keys(), key=int)
print(f"\nCells to viz: {cells}")

VIZ_BOUNDS = [-60.0, -60.0, 60.0, 60.0]
car_colors = nutils.get_adv_coloring(NA, atk_idx, 0)
print(f"car_colors: {car_colors}")

# ===== 画 prior (1 张) + 4 cell (各 1 张) =====
print("\n=== Generating before (prior) ===")
with torch.no_grad():
    dec_prior = model.decode_embedding(ei["prior_out"][0], embed_info, sg, mi, map_env)
    fut_prior = dec_prior["future_pred"]
    fp = norm.unnormalize(fut_prior)
    md_prior = torch.norm(fp[1:, :, :2] - fp[:1, :, :2], dim=-1).min().item()
    print(f"  prior min_dist: {md_prior:.3f}m")

nutils.viz_scene_graph(
    sg, mi, map_env, 0, os.path.join(OUT, "prior_before"),
    norm, model.get_att_normalizer(),
    future_pred=fut_prior,
    viz_traj=True, make_video=False, show_gt=False,
    viz_bounds=VIZ_BOUNDS, center_viz=True,
    car_colors=car_colors,
)
print(f"  Saved: {OUT}/prior_before.png")

print("\n=== Generating 4 cells (perturbed) ===")
for c in cells:
    info = cell_meta[c]
    mult = int(info['sigma_mult'])
    sample_id = info['sample_id']
    sample = results[mult][sample_id]
    # future_pred is (NA, FT, 4) normalized - 直接传给 viz
    fut = sample['future_pred'].to(device)  # (NA, FT, 4) normalized

    out_prefix = os.path.join(OUT, f"cell_{c}_after")
    nutils.viz_scene_graph(
        sg, mi, map_env, 0, out_prefix,
        norm, model.get_att_normalizer(),
        future_pred=fut,
        viz_traj=True, make_video=False, show_gt=False,
        viz_bounds=VIZ_BOUNDS, center_viz=True,
        car_colors=car_colors,
    )
    print(f"  cell {c}: sigma={mult}x, sample_id={sample_id}, "
          f"md={sample['min_dist']:.3f}m -> {out_prefix}.png")

print("\nDone.")
