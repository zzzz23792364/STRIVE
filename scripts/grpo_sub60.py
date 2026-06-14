"""GRPO on val subseq 60 (NA=2, min_dist=6.42m) using corrected reparameterization."""
import os, sys, torch
sys.path.insert(0, "src")

from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from datasets.utils import NUSC_BIKE_PARAMS
from rl.policy import GaussianPolicy
from rl.grpo import GRPO

device = get_device()
print(f"Device: {device}")

# === Load scene (must match adv_scenario_gen.py params) ===
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

NA = sg.future_gt.size(0)
ptr = sg.ptr
ego_mask = torch.zeros(NA, dtype=torch.bool, device=device)
ego_mask[ptr[:-1]] = True

# Attack agent = the only non-ego (NA=2, so it's agent 1)
atk_idx = 1
prior_mu_atk = ei["prior_out"][0][atk_idx].detach()
with torch.no_grad():
    dec_init = model.decode_embedding(ei["posterior_out"][0], ei, sg, mi, map_env)
    fu_init = model.get_normalizer().unnormalize(dec_init["future_pred"])
    init_md = torch.norm(fu_init[1:,:,:2] - fu_init[:1,:,:2], dim=-1).min().item()
print(f"Subseq 60: NA={NA}, init_min_dist={init_md:.3f}m")

# Build obs for attack agent
obs_full = torch.cat([ei["map_feat"], ei["past_feat"],
                       ei["prior_out"][0], ei["prior_out"][1],
                       sg.lw, sg.sem], dim=-1)
obs_atk = obs_full[atk_idx:atk_idx+1]

# z_base: ego=prior, others=posterior
z_base = ei["posterior_out"][0].detach().clone()
z_base[ego_mask] = ei["prior_out"][0][ego_mask].detach()

# Policy
policy = GaussianPolicy(obs_atk.size(-1), 32).to(device)
opt = torch.optim.Adam(policy.parameters(), lr=0.0003)
grpo = GRPO(policy, opt, group_size=32, clip_eps=0.3, device=device)
norm = model.get_normalizer()

best_md = float("inf")
best_delta = None

print("Starting GRPO (500 iters, K=32)...")
for it in range(500):
    delta, lp, adv, rw, bd = grpo.sample_and_score(
        obs_atk, prior_mu_atk, z_base, atk_idx, model,
        ei, sg, mi, map_env,
    )
    stats = grpo.update(obs_atk, delta, lp, adv)

    # Evaluate best sample from this iteration
    zf = z_base.clone()
    zf[atk_idx] = prior_mu_atk + bd
    with torch.no_grad():
        d = model.decode_embedding(zf.unsqueeze(1), ei, sg, mi, map_env)
        fp = norm.unnormalize(d["future_pred"][:, 0])
        md = torch.norm(fp[atk_idx,:,:2] - fp[0,:,:2], dim=-1).min().item()
    if md < best_md:
        best_md = md
        best_delta = bd.clone()

    if it % 50 == 0:
        print(f"  it {it}: R_best={rw.max().item():.2f} best_min_d={best_md:.4f}m  "
              f"|delta_mu|={delta.mean().item():.4f}")

print(f"\nBest min_dist: {best_md:.4f}m")

# === Visualize ===
OUT = "./out/viz_grpo_sub60"
os.makedirs(OUT, exist_ok=True)

# Before (prior)
with torch.no_grad():
    z_prior = z_base.clone()
    z_prior[atk_idx] = prior_mu_atk
    d = model.decode_embedding(z_prior.unsqueeze(1), ei, sg, mi, map_env)
    nutils.viz_scene_graph(sg, mi, map_env, 0, os.path.join(OUT, "before"),
        norm, model.get_att_normalizer(),
        future_pred=d["future_pred"][:, 0].unsqueeze(1),
        viz_traj=True, make_video=False, show_gt=True,
        viz_bounds=[-60.0,-60.0,60.0,60.0], center_viz=True)

# After (best adversarial)
zf = z_base.clone()
zf[atk_idx] = prior_mu_atk + best_delta
with torch.no_grad():
    d = model.decode_embedding(zf.unsqueeze(1), ei, sg, mi, map_env)
    nutils.viz_scene_graph(sg, mi, map_env, 0, os.path.join(OUT, "after"),
        norm, model.get_att_normalizer(),
        future_pred=d["future_pred"][:, 0].unsqueeze(1),
        viz_traj=True, make_video=False, show_gt=True,
        viz_bounds=[-60.0,-60.0,60.0,60.0], center_viz=True)

print(f"Saved: {OUT}/before.png, {OUT}/after.png")
print(f"Initial (prior): {torch.norm(norm.unnormalize(ei['prior_out'][0][1:,:,:2] - ei['prior_out'][0][:1,:,:2]), dim=-1).min().item():.3f}m -> {best_md:.4f}m")
