import os
import sys
import torch
from torch_geometric.data import DataLoader as GraphDataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state

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

model.eval()

OUT = "./out/viz_na2"
os.makedirs(OUT, exist_ok=True)

NA2_SCENES = [167, 168, 169, 170, 171, 172, 173, 174, 177, 190,
              191, 192, 353, 354, 355, 356, 361, 362, 363, 364]

count = 0
for i, data in enumerate(loader):
    if i not in NA2_SCENES:
        continue
    if count >= len(NA2_SCENES):
        break

    scene_graph, map_idx = data
    scene_graph, map_idx = scene_graph.to(device), map_idx.to(device)

    with torch.no_grad():
        embed_info_attached = model.embed(scene_graph, map_idx, map_env)
        dec = model.decode_embedding(embed_info_attached["posterior_out"][0],
                                      embed_info_attached, scene_graph, map_idx, map_env)
        fut = dec["future_pred"]

    out_path = os.path.join(OUT, f"scene_{i:04d}")
    nutils.viz_scene_graph(
        scene_graph, map_idx, map_env, 0, out_path,
        model.get_normalizer(), model.get_att_normalizer(),
        future_pred=fut,
        viz_traj=True, make_video=False, show_gt=True,
        viz_bounds=[-60.0, -60.0, 60.0, 60.0], center_viz=True,
    )

    # Also compute min distance info
    na = scene_graph.future_gt.size(0)
    fp_u = model.get_normalizer().unnormalize(fut)
    ego_pos = fp_u[0:1, :, :2]
    other_pos = fp_u[1:, :, :2]
    dists = torch.norm(other_pos - ego_pos, dim=-1)
    min_d = dists.min().item()
    print(f"Scene {i}: NA={na}, min_dist={min_d:.3f}m -> {out_path}.png")

    count += 1

print(f"\nDone. {count} visualizations saved to {OUT}/")
