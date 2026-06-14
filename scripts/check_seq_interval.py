"""Check NA count for specific subseq with seq_interval=10"""
import sys
sys.path.insert(0, "src")
import torch
from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from utils.torch import get_device

device = get_device()
data_path = "./data/nuscenes/trainval"
map_env = NuScenesMapEnv(data_path, bounds=[-17.0, -38.5, 60.0, 38.5],
                         L=256, W=256,
                         layers=["drivable_area", "carpark_area",
                                 "road_divider", "lane_divider"],
                         device=device)

for split_name in ["val", "train"]:
    dataset = NuScenesDataset(data_path, map_env, version="trainval",
                               split=split_name,
                               categories=["car", "truck"],
                               npast=4, nfuture=12, reduce_cats=False,
                               seq_interval=10)
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    for i, data in enumerate(loader):
        if i in [60, 247]:
            sg, mi = data
            na = sg.future_gt.size(0)
            cats = [dataset.vec2cat[tuple(c)] for c in sg.sem.cpu().numpy().tolist()]
            print(f"[{split_name} seq_int=10] subseq {i}: NA={na}, cats={cats}")
        if i > 250:
            break
