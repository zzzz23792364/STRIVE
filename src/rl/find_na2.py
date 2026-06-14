import sys
sys.path.insert(0, "src")
import torch
from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from utils.torch import get_device

device = get_device()
data_path = "./data/nuscenes/trainval"

for split_name in ["val", "train"]:
    map_env = NuScenesMapEnv(data_path, bounds=[-17.0,-38.5,60.0,38.5],
                             L=256, W=256,
                             layers=["drivable_area","carpark_area",
                                     "road_divider","lane_divider"],
                             device=device)
    dataset = NuScenesDataset(data_path, map_env, version="trainval",
                               split=split_name,
                               categories=["car","truck"], npast=4, nfuture=12,
                               reduce_cats=False)
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    na2_subseqs = []
    for i, data in enumerate(loader):
        sg, mi = data
        if sg.future_gt.size(0) == 2:
            na2_subseqs.append(i)

    # Group consecutive indices to find unique original scenes
    unique_groups = []
    if na2_subseqs:
        start = na2_subseqs[0]
        prev = na2_subseqs[0]
        for idx in na2_subseqs[1:]:
            if idx > prev + 1:
                unique_groups.append((start, prev))
                start = idx
            prev = idx
        unique_groups.append((start, prev))

    print(f"{split_name}: NA=2 subseq count={len(na2_subseqs)}")
    print(f"  Unique original scenes: {len(unique_groups)}")
    for s, e in unique_groups:
        n = e - s + 1
        print(f"    scene subseq indices {s}..{e} ({n} subsequences)")
