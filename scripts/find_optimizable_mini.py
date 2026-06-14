"""Find which mini scenes respond to adversarial optimization"""
import os, sys, torch, json
sys.path.insert(0, "src")

from torch_geometric.data import DataLoader as GraphDataLoader, Batch as GraphBatch
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from utils.adv_gen_optim import run_adv_gen_optim, compute_adv_gen_success
from datasets.utils import NUSC_BIKE_PARAMS

device = get_device()
print(f"Device: {device}")

data_path = "./data/nuscenes"
map_env = NuScenesMapEnv(os.path.join(data_path, "mini"), bounds=[-17.0, -38.5, 60.0, 38.5],
                         L=256, W=256,
                         layers=["drivable_area", "carpark_area",
                                 "road_divider", "lane_divider"],
                         device=device)
dataset = NuScenesDataset(os.path.join(data_path, "mini"), map_env,
                           version="mini", split="train",
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

results = []
found_success = False
for i, data in enumerate(loader):
    if found_success and len(results) >= 5:
        break  # stop after finding 1 success + 4 others

    sg, mi = data
    sg, mi = sg.to(device), mi.to(device)
    NA = sg.future_gt.size(0)
    ptr = sg.ptr
    ego_mask = torch.zeros(NA, dtype=torch.bool, device=device)
    ego_mask[ptr[:-1]] = True

    model.eval()
    with torch.no_grad():
        ei = model.embed(sg, mi, map_env)
        z_post = ei["posterior_out"][0]
        dec = model.decode_embedding(z_post, ei, sg, mi, map_env)
        fu = model.get_normalizer().unnormalize(dec["future_pred"])
        dists = torch.norm(fu[1:,:,:2] - fu[:1,:,:2], dim=-1)
        init_min_d = dists.min().item()

    if init_min_d > 15.0:
        print(f"subseq {i}: NA={NA}, init_min_d={init_min_d:.1f}m - SKIP")
        continue

    print(f"subseq {i}: NA={NA}, init_min_d={init_min_d:.3f}m - OPT...", end=" ", flush=True)

    model.train()
    z_init = ei["posterior_out"][0].detach()
    other_prior = (ei["prior_out"][0][~ego_mask], ei["prior_out"][1][~ego_mask])
    tgt_prior = (ei["prior_out"][0][ego_mask], ei["prior_out"][1][ego_mask])
    embed_info = detach_embed_info(ei)

    try:
        cur_z, final_traj, dec_out, min_agt, min_t = run_adv_gen_optim(
            z_init, 0.05, loss_weights, model, sg, map_env, mi,
            100, embed_info, "ego", tgt_prior, other_prior,
            feasibility_time=0, feasibility_infront_min=None,
        )
        success = False
        if min_agt is not None and len(min_agt) > 0:
            scene_sg = sg.to_data_list()[0]
            success = compute_adv_gen_success(
                final_traj, model, GraphBatch.from_data_list([scene_sg]),
                min_agt[0] - ptr[0].item(),
            )
        with torch.no_grad():
            model.eval()
            fut = model.get_normalizer().unnormalize(final_traj[:, 0])
            tgt = model.get_normalizer().unnormalize(sg.future_gt[ego_mask][:,:,:4])
            others = torch.cat([tgt[b:b+1].expand((ptr[b+1]-ptr[b]-1), -1, -1) for b in range(mi.size(0))], dim=0)
            final_d = torch.norm(fut[~ego_mask][:,:,:2] - others[:,:,:2], dim=-1).min().item()
            print(f"final_d={final_d:.3f} succ={success}")
            results.append({"idx": i, "NA": NA, "init_d": init_min_d, "final_d": final_d, "success": success})
            if success:
                found_success = True
    except Exception as e:
        print(f"ERR: {e}")

print("\n" + "="*60)
print("RESULTS: scenes where final_d < 2.0m (near collision):")
for r in results:
    if r["final_d"] < 2.0 or r["success"]:
        print(f"  subseq {r['idx']}: NA={r['NA']} {r['init_d']:.3f}->{r['final_d']:.3f} success={r['success']}")
print("\nBest improvements:")
sorted_r = sorted(results, key=lambda r: r["init_d"] - r["final_d"], reverse=True)
for r in sorted_r[:5]:
    print(f"  subseq {r['idx']}: init={r['init_d']:.3f} -> final={r['final_d']:.3f} (delta={r['init_d']-r['final_d']:.3f}m) success={r['success']}")
