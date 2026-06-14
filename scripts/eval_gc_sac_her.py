"""独立评估脚本: 用 v6 训好的 policy 做 16 goal 推理 (deterministic + stochastic)."""
import os, sys, json
sys.path.insert(0, "src")

import torch
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
from rl.conditional_policy import ConditionalPolicy

device = get_device()
print(f"Device: {device}")

OUT = "./out/gc_sac_her"

# ===== 加载 sub60 =====
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


@torch.no_grad()
def evaluate_z_full(z_atk):
    z_atk_t = torch.tensor(z_atk, device=device, dtype=torch.float32).unsqueeze(0)
    ego_z = prior_mu[0].unsqueeze(0)
    z_full = torch.stack([ego_z, z_atk_t], dim=0)
    dec = model.decode_embedding(z_full, embed_info, sg, mi, map_env)
    fut = norm.unnormalize(dec['future_pred'])
    decoded_atk = fut[atk_idx, 0, :, :4]

    collides, coll_times, min_dists = fast_collision_check_vectorized(
        decoded_atk.unsqueeze(0), ego_replay, atk_lw, ego_lw
    )
    collides = bool(collides[0].item())
    ct = int(coll_times[0].item())
    min_dist = float(min_dists[0].item())

    if collides and 0 <= ct < decoded_atk.size(0):
        bd_idx, _, _ = compute_bd_from_collision(ego_replay, decoded_atk, ct)
    else:
        bd_idx = -1

    return collides, min_dist, bd_idx, fut  # 返回 fut 用于 viz


# ===== Load trained policy =====
print("Loading trained policy...")
policy = ConditionalPolicy(obs_dim=OBS_DIM, goal_dim=16, z_dim=32, hidden=256).to(device)
policy.load_state_dict(torch.load(os.path.join(OUT, "policy.pt"), map_location=device))
policy.eval()

# ===== Eval: 16 goals, deterministic first then stochastic =====
print("\n=== Policy inference: 16 goals (deterministic + stochastic) ===")
policy.eval()
with torch.no_grad():
    obs_t = torch.tensor(obs_atk, device=device, dtype=torch.float32).unsqueeze(0)
    all_z = []
    for g in range(16):
        gh = torch.zeros(1, 16, device=device); gh[0, g] = 1.0
        best_md = float('inf')
        best_z = None
        best_fut = None

        # 8 次 deterministic
        for trial in range(8):
            z, _, _ = policy.sample(obs_t, gh, deterministic=True)
            z_np = z.squeeze(0).cpu().numpy()
            collides, min_dist, bd_actual, fut = evaluate_z_full(z_np)
            if collides and bd_actual == g and min_dist < best_md:
                best_md = min_dist
                best_z = z_np
                best_fut = fut

        # 16 次 stochastic
        if best_z is None:
            for trial in range(16):
                z, _, _ = policy.sample(obs_t, gh, deterministic=False)
                z_np = z.squeeze(0).cpu().numpy()
                collides, min_dist, bd_actual, fut = evaluate_z_full(z_np)
                if collides and bd_actual == g and min_dist < best_md:
                    best_md = min_dist
                    best_z = z_np
                    best_fut = fut
                    break

        if best_z is not None:
            print(f"  goal {g:2d}: hit, md={best_md:.3f}m")
            all_z.append((g, best_z, best_md, best_fut))
        else:
            print(f"  goal {g:2d}: miss")

with open(os.path.join(OUT, "policy_eval.json"), 'w') as f:
    json.dump([{"goal": g, "min_dist": md} for g, _, md, _ in all_z], f, indent=2)

# ===== Viz 4 cell =====
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

for g, z_atk, md, fut in all_z:
    nutils.viz_scene_graph(
        sg, mi, map_env, 0, os.path.join(OUT, f"cell_{g:02d}_after"),
        norm, model.get_att_normalizer(),
        future_pred=fut,
        viz_traj=True, make_video=False, show_gt=False,
        viz_bounds=VIZ_BOUNDS, center_viz=True,
        car_colors=car_colors,
    )
    print(f"Saved: {OUT}/cell_{g:02d}_after.png (md={md:.3f}m)")

print("\nDone.")
