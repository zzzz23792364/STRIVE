"""
Run original gradient-based adversarial optimization on mini sub024 (NA=3).
"""
import sys, os
sys.path.insert(0, "src")

import torch
import torch.optim as optim
from torch_geometric.data import DataLoader as GraphDataLoader
from torch_geometric.data import Batch as GraphBatch

from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from losses.traffic_model import compute_coll_rate_env, compute_coll_rate_veh
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from utils.common import dict2obj, mkdir
from utils.logger import Logger
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from utils.adv_gen_optim import run_adv_gen_optim, compute_adv_gen_success
from utils.init_optim import run_init_optim
from utils.config import get_parser, add_base_args

device = get_device()
print(f"Device: {device}")

# Load mini data
data_path = "./data/nuscenes/mini"
map_env = NuScenesMapEnv(data_path, bounds=[-17.0, -38.5, 60.0, 38.5],
                         L=256, W=256,
                         layers=["drivable_area", "carpark_area",
                                 "road_divider", "lane_divider"],
                         device=device)
dataset = NuScenesDataset(data_path, map_env, version="mini", split="train",
                           categories=["car", "truck"], npast=4, nfuture=12,
                           reduce_cats=False)
loader = GraphDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                          pin_memory=False)

# Load model
model = TrafficModel(4, 12, 256, len(dataset.categories), output_bicycle=True,
                     conv_kernel_list=[7,5,5,3,3,3],
                     conv_stride_list=[2,2,2,2,2,2],
                     conv_filter_list=[16,32,64,64,128,128]).to(device)
load_state("./model_ckpt/traffic_model.pth", model, map_location=device)
model.set_normalizer(dataset.get_state_normalizer())
model.set_att_normalizer(dataset.get_att_normalizer())
from datasets.utils import NUSC_BIKE_PARAMS
model.set_bicycle_params(NUSC_BIKE_PARAMS)

# Load sub024 (NA=3)
for i, data in enumerate(loader):
    if i == 24:
        scene_graph, map_idx = data
        scene_graph, map_idx = scene_graph.to(device), map_idx.to(device)
        break

NA = scene_graph.future_gt.size(0)
B = map_idx.size(0)
ptr = scene_graph.ptr
ego_inds = ptr[:-1]
ego_mask = torch.zeros((NA), dtype=torch.bool, device=device)
ego_mask[ego_inds] = True

print(f"Scene: NA={NA}, ptr={ptr.tolist()}")

# Embed (model.train() needed for GRU backward)
model.train()
with torch.no_grad():
    embed_info_attached = model.embed(scene_graph, map_idx, map_env)
embed_info = detach_embed_info(embed_info_attached)

# Loss weights (same as adv_gen_rule_based.cfg)
loss_weights = {
    "coll_veh": 20.0, "coll_veh_plan": 20.0, "coll_env": 20.0,
    "motion_prior": 1.0, "motion_prior_atk": 0.005,
    "init_z": 0.5, "init_z_atk": 0.05,
    "motion_prior_ext": 0.0001, "match_ext": 10.0,
    "adv_crash": 2.0,
    "sol_coll_veh": 10.0, "sol_coll_env": 10.0,
    "sol_motion_prior": 0.005, "sol_init_z": 0.0,
    "sol_motion_prior_ext": 0.001, "sol_match_ext": 10.0,
    "init_match_ext": 10.0, "init_motion_prior_ext": 0.01,
}

z_init = embed_info_attached["posterior_out"][0].detach()
tgt_prior = (embed_info["prior_out"][0][ego_mask],
             embed_info["prior_out"][1][ego_mask])
other_prior = (embed_info["prior_out"][0][~ego_mask],
               embed_info["prior_out"][1][~ego_mask])

print("Running adversarial optimization (200 iters)...")
result = run_adv_gen_optim(
    z_init, 0.05, loss_weights, model, scene_graph, map_env, map_idx,
    200, embed_info, "ego", tgt_prior, other_prior,
    feasibility_time=0, feasibility_infront_min=None,
)
cur_z, final_traj, decoder_out, min_agt, min_t = result

# Check success
success = False
if min_agt is not None and len(min_agt) > 0:
    scene_sg = scene_graph.to_data_list()[0]
    success = compute_adv_gen_success(
        final_traj, model,
        GraphBatch.from_data_list([scene_sg]),
        min_agt[0] - scene_graph.ptr[0].item(),
    )

# Min distance
with torch.no_grad():
    model.eval()
    fut = model.get_normalizer().unnormalize(final_traj[:, 0])
    tgt = model.get_normalizer().unnormalize(
        scene_graph.future_gt[ego_mask][:, :, :4])
    others_warp = torch.cat([
        tgt[b:b+1].expand(
            (ptr[b+1]-ptr[b]-1), -1, -1)
        for b in range(B)
    ], dim=0)
    dists = torch.norm(fut[~ego_mask][:, :, :2] - others_warp[:, :, :2], dim=-1)
    min_d = dists.min().item()

print(f"\nResults:")
print(f"  Success: {success}")
print(f"  Attack agent (local): {min_agt}")
print(f"  Attack time: {min_t}")
print(f"  Min crash distance: {min_d:.4f}m")

# Also compute initial min distance (before optimization)
with torch.no_grad():
    dec_init = model.decode_embedding(z_init, embed_info, scene_graph, map_idx, map_env)
    init_fut = model.get_normalizer().unnormalize(dec_init["future_pred"])
    init_dists = torch.norm(init_fut[~ego_mask][:, :, :2] - others_warp[:, :, :2], dim=-1)
    init_min = init_dists.min().item()
print(f"  Initial min distance: {init_min:.4f}m")
