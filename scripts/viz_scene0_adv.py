"""Generate adversarial trajectory viz for scene 0 (NA=7) using GRPO + residual."""
import os, sys, torch
sys.path.insert(0, "src")

from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from datasets.utils import NUSC_BIKE_PARAMS

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
model.set_bicycle_params(NUSC_BIKE_PARAMS)
model.eval()

for i, data in enumerate(loader):
    if i == 0:
        sg, mi = data
        sg, mi = sg.to(device), mi.to(device)
        break

with torch.no_grad():
    ei = model.embed(sg, mi, map_env)

NA = sg.future_gt.size(0)
ptr = sg.ptr
ego_mask = torch.zeros(NA, dtype=torch.bool, device=device)
ego_mask[ptr[:-1]] = True

# Find closest non-ego
with torch.no_grad():
    z_post = ei["posterior_out"][0]
    dec = model.decode_embedding(z_post, ei, sg, mi, map_env)
    fu = model.get_normalizer().unnormalize(dec["future_pred"])
    dists = torch.norm(fu[1:,:,:2] - fu[:1,:,:2], dim=-1).min(dim=-1)[0]
    atk_idx = dists.argmin().item() + 1
print(f"Attack agent: {atk_idx}, initial min_dist={dists.min().item():.3f}m")

# Build obs and z_base
obs_full = torch.cat([ei["map_feat"], ei["past_feat"],
                       ei["prior_out"][0], ei["prior_out"][1],
                       sg.lw, sg.sem], dim=-1)
obs_atk = obs_full[atk_idx:atk_idx+1]
z_base = ei["posterior_out"][0].detach().clone()
z_base[ego_mask] = ei["prior_out"][0][ego_mask].detach()

from rl.policy import GaussianPolicy
from rl.grpo import GRPO

policy = GaussianPolicy(obs_atk.size(-1), 32).to(device)
opt = torch.optim.Adam(policy.parameters(), lr=0.0003)
grpo = GRPO(policy, opt, group_size=32, clip_eps=0.3, device=device)
norm = model.get_normalizer()

best_md = float("inf")
best_z = None
for it in range(200):
    dz, lp, adv, rw, bz = grpo.sample_and_score(obs_atk, z_base, atk_idx, 10.0,
                                                  model, ei, sg, mi, map_env)
    stats = grpo.update(obs_atk, dz, lp, adv)

    zf = z_base.clone()
    zf[atk_idx] = ei["posterior_out"][0][atk_idx].detach() + torch.tanh(bz) * 10.0
    with torch.no_grad():
        d = model.decode_embedding(zf.unsqueeze(1), ei, sg, mi, map_env)
        fp = norm.unnormalize(d["future_pred"][:, 0])
        md = torch.norm(fp[atk_idx,:,:2] - fp[0,:,:2], dim=-1).min().item()
    if md < best_md:
        best_md = md
        best_z = bz.clone()
    if it % 50 == 0:
        r_best = rw.max().item()
        print(f"  it {it}: R_best={r_best:.2f} best_min_d={best_md:.4f}m")

print(f"\nBest min_dist: {best_md:.4f}m")

OUT = "./out/viz_results"
os.makedirs(OUT, exist_ok=True)

# Render adversarial
zf = z_base.clone()
zf[atk_idx] = ei["posterior_out"][0][atk_idx].detach() + torch.tanh(best_z) * 10.0
with torch.no_grad():
    d2 = model.decode_embedding(zf.unsqueeze(1), ei, sg, mi, map_env)
    fut_adv = d2["future_pred"][:, 0]

nutils.viz_scene_graph(
    sg, mi, map_env, 0, os.path.join(OUT, "adversarial"),
    model.get_normalizer(), model.get_att_normalizer(),
    future_pred=fut_adv.unsqueeze(1),
    viz_traj=True, make_video=False, show_gt=True,
    viz_bounds=[-60.0, -60.0, 60.0, 60.0], center_viz=True,
)

print(f"Saved: {OUT}/adversarial.png")
print(f"Posterior: {OUT}/posterior.png (already exists)")
print(f"Scene 0 NA=7 | Initial min_dist=4.08m | Adversarial min_dist={best_md:.4f}m")
