"""Scan NA=2 scenes matching adv_scenario_gen.py config (randomize_val=True, seq_interval=10)"""
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
    map_env = NuScenesMapEnv(data_path, bounds=[-17.0, -38.5, 60.0, 38.5],
                             L=256, W=256,
                             layers=["drivable_area", "carpark_area",
                                     "road_divider", "lane_divider"],
                             device=device)
    # Match adv_scenario_gen.py settings
    dataset = NuScenesDataset(data_path, map_env,
                               version="trainval",
                               split=split_name,
                               categories=["car", "truck"],
                               npast=4, nfuture=12,
                               reduce_cats=False,
                               seq_interval=10,
                               randomize_val=True,
                               val_size=400)
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    total = len(loader)
    na2_subseqs = []
    for i, data in enumerate(loader):
        sg, mi = data
        if sg.future_gt.size(0) == 2:
            na2_subseqs.append(i)

    # Group consecutive indices to find unique original scenes
    groups = []
    if na2_subseqs:
        start = na2_subseqs[0]
        prev = na2_subseqs[0]
        for idx in na2_subseqs[1:]:
            if idx > prev + 1:
                groups.append((start, prev))
                start = idx
            prev = idx
        groups.append((start, prev))

    print(f"\n[{split_name}] total subseq={total}")
    print(f"  NA=2 subseq count={len(na2_subseqs)}")
    print(f"  Unique original scenes={len(groups)}")
    if groups:
        print(f"  First subseq indices per scene:")
        for s, e in groups:
            print(f"    subseq {s}..{e} ({e-s+1} subseqs)")
