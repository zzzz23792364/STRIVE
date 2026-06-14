"""Viz all NA=2 scenes matching adv_scenario_gen.py params (randomize_val=True, seq_interval=10)"""
import os, sys, torch
sys.path.insert(0, "src")

from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from datasets.utils import NUSC_BIKE_PARAMS

device = get_device()
print(f"Device: {device}")

data_path = "./data/nuscenes/trainval"

OUT = "./out/viz_na2_advgen"
os.makedirs(OUT, exist_ok=True)

for split_name in ["val", "train"]:
    map_env = NuScenesMapEnv(data_path, bounds=[-17.0, -38.5, 60.0, 38.5],
                             L=256, W=256,
                             layers=["drivable_area", "carpark_area",
                                     "road_divider", "lane_divider"],
                             device=device)
    dataset = NuScenesDataset(data_path, map_env,
                               version="trainval", split=split_name,
                               categories=["car", "truck"],
                               npast=4, nfuture=12, reduce_cats=False,
                               seq_interval=10, randomize_val=True, val_size=400)
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = TrafficModel(4, 12, 256, len(dataset.categories), output_bicycle=True,
                         conv_kernel_list=[7,5,5,3,3,3],
                         conv_stride_list=[2,2,2,2,2,2],
                         conv_filter_list=[16,32,64,64,128,128]).to(device)
    load_state("./model_ckpt/traffic_model.pth", model, map_location=device)
    model.set_normalizer(dataset.get_state_normalizer())
    model.set_att_normalizer(dataset.get_att_normalizer())
    model.set_bicycle_params(NUSC_BIKE_PARAMS)
    model.eval()

    prev_na2_idx = -10
    for i, data in enumerate(loader):
        sg, mi = data
        sg, mi = sg.to(device), mi.to(device)
        na = sg.future_gt.size(0)
        if na != 2:
            continue

        # If gap is 1, same original scene (adjacent subseqs), skip
        if i - prev_na2_idx == 1:
            prev_na2_idx = i
            continue
        prev_na2_idx = i

        with torch.no_grad():
            ei = model.embed(sg, mi, map_env)
            dec = model.decode_embedding(ei["posterior_out"][0], ei, sg, mi, map_env)
            fut = dec["future_pred"]

        cats = [dataset.vec2cat[tuple(c)] for c in sg.sem.cpu().numpy().tolist()]
        op = os.path.join(OUT, f"{split_name}_sub{i:04d}_NA{na}")
        nutils.viz_scene_graph(
            sg, mi, map_env, 0, op,
            model.get_normalizer(), model.get_att_normalizer(),
            future_pred=fut,
            viz_traj=True, make_video=False, show_gt=True,
            viz_bounds=[-60.0, -60.0, 60.0, 60.0], center_viz=True,
        )

        fp_u = model.get_normalizer().unnormalize(fut)
        md = torch.norm(fp_u[1:,:,:2] - fp_u[:1,:,:2], dim=-1).min().item()
        print(f"[{split_name}] subseq {i}: NA={na}, cats={cats}, min_dist={md:.3f}m")

print(f"\nDone. Saved to {OUT}/")
