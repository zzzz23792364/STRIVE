import os
import sys
import torch
import numpy as np
from torch_geometric.data import DataLoader as GraphDataLoader
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from rl.policy import GaussianPolicy

device = get_device()
print(f"Device: {device}")

data_path = "./data/nuscenes/trainval"
map_env = NuScenesMapEnv(data_path, bounds=[-17.0, -38.5, 60.0, 38.5],
                         L=256, W=256,
                         layers=["drivable_area", "carpark_area",
                                 "road_divider", "lane_divider"],
                         device=device)
dataset = NuScenesDataset(data_path, map_env, version="trainval", split="val",
                           categories=["car", "truck"], npast=4, nfuture=12,
                           reduce_cats=False)
loader = GraphDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                          pin_memory=False)

model = TrafficModel(4, 12, 256, len(dataset.categories), output_bicycle=True,
                     conv_kernel_list=[7,5,5,3,3,3],
                     conv_stride_list=[2,2,2,2,2,2],
                     conv_filter_list=[16,32,64,64,128,128]).to(device)
load_state("./model_ckpt/traffic_model.pth", model, map_location=device)
model.set_normalizer(dataset.get_state_normalizer())
model.set_att_normalizer(dataset.get_att_normalizer())
from datasets.utils import NUSC_BIKE_PARAMS
model.set_bicycle_params(NUSC_BIKE_PARAMS)

# Load scene 0
for i, data in enumerate(loader):
    if i == 0:
        scene_graph, map_idx = data
        scene_graph, map_idx = scene_graph.to(device), map_idx.to(device)
        break
print(f"Scene: {scene_graph}")
NA = scene_graph.future_gt.size(0)
ptr = scene_graph.ptr
ego_inds = ptr[:-1]
ego_mask = torch.zeros((NA), dtype=torch.bool, device=device)
ego_mask[ego_inds] = True

model.eval()
with torch.no_grad():
    embed_info_attached = model.embed(scene_graph, map_idx, map_env)
embed_info = detach_embed_info(embed_info_attached)

# Identify attack agent
with torch.no_grad():
    z_post = embed_info["posterior_out"][0]
    dec = model.decode_embedding(z_post, embed_info, scene_graph, map_idx, map_env)
    fut = model.get_normalizer().unnormalize(dec["future_pred"])
    ego_pos = fut[0:1, :, :2]
    other_pos = fut[1:, :, :2]
    dists = torch.norm(other_pos - ego_pos, dim=-1).min(dim=-1)[0]
    attack_agt_idx = dists.argmin().item() + 1
print(f"Attack agent: {attack_agt_idx} (min_dist={dists.min().item():.3f}m)")

OUT = "./out/viz_results"

# ========== 1. Visualize POSTERIOR (before) ==========
print("Rendering posterior (before)...")
fut_post = dec["future_pred"]  # NORMALIZED
nutils.viz_scene_graph(
    scene_graph, map_idx, map_env, 0,
    os.path.join(OUT, "posterior"),
    model.get_normalizer(), model.get_att_normalizer(),
    future_pred=fut_post,
    viz_traj=True, make_video=False, show_gt=True,
    viz_bounds=[-60.0, -60.0, 60.0, 60.0], center_viz=True,
)

# ========== 2. Run GRPO sampling to find best adversarial z ==========
print("Running GRPO sampling (K=64, scale=10)...")
obs_full = torch.cat([embed_info["map_feat"], embed_info["past_feat"],
                       embed_info["prior_out"][0], embed_info["prior_out"][1],
                       scene_graph.lw, scene_graph.sem], dim=-1)
obs_atk = obs_full[attack_agt_idx:attack_agt_idx+1]
obs_dim = obs_atk.size(-1)
action_dim = 32

policy = GaussianPolicy(obs_dim, action_dim).to(device)
opt = torch.optim.Adam(policy.parameters(), lr=0.0003)

z_base = embed_info["posterior_out"][0].detach().clone()
z_base[ego_mask] = embed_info["prior_out"][0][ego_mask].detach()
norm = model.get_normalizer()

best_min_d = float("inf")
best_z_atk = None
best_fut = None

for grpo_it in range(100):
    with torch.no_grad():
        mu, std = policy(obs_atk)
        mu = mu.squeeze(0)
        std = std.squeeze(0)
        dist = torch.distributions.Normal(mu, std)
        delta_z = dist.sample((64,))
        z_attack = embed_info["posterior_out"][0][attack_agt_idx].unsqueeze(0).expand(64, -1) + torch.tanh(delta_z) * 10.0

    z_base_k = z_base.unsqueeze(0).expand(64, NA, 32).clone()
    z_base_k[:, attack_agt_idx, :] = z_attack
    z_full = z_base_k.transpose(0, 1)

    with torch.no_grad():
        dec_out = model.decode_embedding(z_full, embed_info, scene_graph, map_idx, map_env)
        fp = dec_out["future_pred"]
        atk_pos = norm.unnormalize(fp[attack_agt_idx])
        ego_pos_n = norm.unnormalize(fp[0])

    for k in range(64):
        md = torch.norm(atk_pos[k, :, :2] - ego_pos_n[k, :, :2], dim=-1).min().item()
        if md < best_min_d:
            best_min_d = md
            best_z_atk = z_attack[k].clone()
            best_fut = fp[:, k]
            print(f"  iter {grpo_it}, sample {k}: min_dist={md:.4f}m")

print(f"\nBest min_dist found: {best_min_d:.4f}m")

# ========== 3. Visualize ADVERSARIAL (after) ==========
print("Rendering adversarial (after)...")
z_full_best = z_base.clone()
z_full_best[attack_agt_idx] = embed_info["posterior_out"][0][attack_agt_idx].detach() + torch.tanh(best_z_atk) * 10.0
with torch.no_grad():
    dec_best = model.decode_embedding(z_full_best.unsqueeze(1), embed_info, scene_graph, map_idx, map_env)
    fut_adv = dec_best["future_pred"][:, 0]

nutils.viz_scene_graph(
    scene_graph, map_idx, map_env, 0,
    os.path.join(OUT, "adversarial"),
    model.get_normalizer(), model.get_att_normalizer(),
    future_pred=fut_adv.unsqueeze(1),
    viz_traj=True, make_video=False, show_gt=True,
    viz_bounds=[-60.0, -60.0, 60.0, 60.0], center_viz=True,
)

# ========== 4. Side-by-side comparison ==========
print("Creating comparison...")
from PIL import Image
post_img = Image.open(os.path.join(OUT, "posterior.png"))
adv_img = Image.open(os.path.join(OUT, "adversarial.png"))
w = post_img.width
comp = Image.new("RGB", (w * 2 + 10, max(post_img.height, adv_img.height)))
comp.paste(post_img, (0, 0))
comp.paste(adv_img, (w + 10, 0))
comp.save(os.path.join(OUT, "comparison.png"))
print(f"Saved comparison to {OUT}/comparison.png")
print(f"Posterior: {os.path.join(OUT, 'posterior.png')}")
print(f"Adversarial: {os.path.join(OUT, 'adversarial.png')}")
