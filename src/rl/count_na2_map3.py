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
                               npast=4, nfuture=12, reduce_cats=False)
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    count = 0
    indices = []
    for i, data in enumerate(loader):
        sg, mi = data
        if sg.future_gt.size(0) == 2 and mi.item() == 3:
            count += 1
            indices.append(i)

    print(f"[{split_name}] NA=2 with map_idx=3: {count} subseqs")
    if indices:
        # group consecutive to show unique original scenes
        groups = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx > prev + 1:
                groups.append((start, prev))
                start = idx
            prev = idx
        groups.append((start, prev))
        print(f"  Unique original scenes: {len(groups)}")
        for s, e in groups:
            n = e - s + 1
            print(f"    indices {s}..{e} ({n} subseqs)")
