"""Fast single-scene adversarial optimization (val subseq 60, NA=2)"""
import os, sys, torch
sys.path.insert(0, "src")

from torch_geometric.data import DataLoader as GraphDataLoader, Batch as GraphBatch
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from utils.adv_gen_optim import run_adv_gen_optim, compute_adv_gen_success
from datasets.utils import NUSC_BIKE_PARAMS

device = get_device()
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

model = TrafficModel(4, 12, 256, len(dataset.categories), output_bicycle=True,
                     conv_kernel_list=[7,5,5,3,3,3],
                     conv_stride_list=[2,2,2,2,2,2],
                     conv_filter_list=[16,32,64,64,128,128]).to(device)
load_state("./model_ckpt/traffic_model.pth", model)
model.set_normalizer(dataset.get_state_normalizer())
model.set_att_normalizer(dataset.get_att_normalizer())
model.set_bicycle_params(NUSC_BIKE_PARAMS)

model.train()
with torch.no_grad():
    ei = model.embed(sg, mi, map_env)
z_init = ei["posterior_out"][0].detach()
embed_info = detach_embed_info(ei)
other_prior = (ei["prior_out"][0][~ego_mask], ei["prior_out"][1][~ego_mask])
tgt_prior = (ei["prior_out"][0][ego_mask], ei["prior_out"][1][ego_mask])

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

model.eval()
with torch.no_grad():
    dec_init = model.decode_embedding(z_init, embed_info, sg, mi, map_env)
    fu_init = model.get_normalizer().unnormalize(dec_init["future_pred"])
    init_d = torch.norm(fu_init[~ego_mask][:,:,:2] - fu_init[:1,:,:2], dim=-1).min().item()
    print(f"Initial min_dist: {init_d:.3f}m")

model.train()
result = run_adv_gen_optim(z_init, 0.05, loss_weights, model, sg, map_env, mi,
                            200, embed_info, "ego", tgt_prior, other_prior,
                            feasibility_time=0, feasibility_infront_min=None)
cur_z, final_traj, decoder_out, min_agt, min_t = result

model.eval()
success = False
if min_agt is not None and len(min_agt) > 0:
    scene_sg = sg.to_data_list()[0]
    success = compute_adv_gen_success(final_traj, model,
                                       GraphBatch.from_data_list([scene_sg]),
                                       min_agt[0] - ptr[0].item())
with torch.no_grad():
    fut = model.get_normalizer().unnormalize(final_traj[:, 0])
    tgt = model.get_normalizer().unnormalize(sg.future_gt[ego_mask][:,:,:4])
    final_d = torch.norm(fut[~ego_mask][:,:,:2] - tgt[:,:,:2], dim=-1).min().item()

print(f"min_dist: {init_d:.4f} -> {final_d:.4f}m | success={success}")

# Viz
viz_out = "./out/viz_adv_sub60"
os.makedirs(viz_out, exist_ok=True)
with torch.no_grad():
    model.eval()
    zf = cur_z.clone()
    dec_adv = model.decode_embedding(zf.unsqueeze(1), embed_info, sg, mi, map_env)
    fut_adv = dec_adv["future_pred"][:, 0]
for label, fp in [("before", dec_init["future_pred"]), ("after", fut_adv.unsqueeze(1))]:
    nutils.viz_scene_graph(sg, mi, map_env, 0, os.path.join(viz_out, label),
        model.get_normalizer(), model.get_att_normalizer(),
        future_pred=fp, viz_traj=True, make_video=False, show_gt=True,
        viz_bounds=[-60.0,-60.0,60.0,60.0], center_viz=True)
print(f"Saved: {viz_out}/")
