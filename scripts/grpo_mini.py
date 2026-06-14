"""GRPO on mini sub072 (NA=14, init min_dist=2.7m)"""
import os, sys, torch
sys.path.insert(0, "src")

from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from datasets.utils import NUSC_BIKE_PARAMS
from rl.policy import GaussianPolicy
from rl.grpo import GRPO

device = get_device()
print(f"Device: {device}")

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
    if i == 72:
        sg, mi = data
        sg, mi = sg.to(device), mi.to(device)
        break

with torch.no_grad():
    ei = model.embed(sg, mi, map_env)

NA = sg.future_gt.size(0)
ptr = sg.ptr
ego_mask = torch.zeros(NA, dtype=torch.bool, device=device)
ego_mask[ptr[:-1]] = True

with torch.no_grad():
    z_post = ei["posterior_out"][0]
    dec = model.decode_embedding(z_post, ei, sg, mi, map_env)
    fu = model.get_normalizer().unnormalize(dec["future_pred"])
    dists = torch.norm(fu[1:,:,:2] - fu[:1,:,:2], dim=-1).min(dim=-1)[0]
    atk_idx = dists.argmin().item() + 1
print(f"sub072: NA={NA}, attack={atk_idx}, init_min_dist={dists.min().item():.3f}m")

obs_full = torch.cat([ei["map_feat"], ei["past_feat"],
                       ei["prior_out"][0], ei["prior_out"][1],
                       sg.lw, sg.sem], dim=-1)
obs_atk = obs_full[atk_idx:atk_idx+1]
z_base = ei["posterior_out"][0].detach().clone()
z_base[ego_mask] = ei["prior_out"][0][ego_mask].detach()
norm = model.get_normalizer()

policy = GaussianPolicy(obs_atk.size(-1), 32).to(device)
opt = torch.optim.Adam(policy.parameters(), lr=0.0003)
grpo = GRPO(policy, opt, group_size=32, clip_eps=0.3, device=device)

best_md = float("inf")
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
    if md < best_md: best_md = md
    if it % 50 == 0:
        print(f"  it {it}: R_best={rw.max().item():.2f} best_min_d={best_md:.4f}m")

print(f"\n  Final best min_dist: {best_md:.4f}m")
