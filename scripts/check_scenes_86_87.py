"""Check raw nuScenes scenes"""
import sys
sys.path.insert(0, "src")
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version="v1.0-trainval", dataroot="./data/nuscenes/trainval", verbose=False)

for scene_idx in [86, 87]:
    scene = nusc.scene[scene_idx]
    token = scene['first_sample_token']
    cars_and_trucks = set()
    while token != '':
        sample = nusc.get('sample', token)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            name = ann['category_name']
            if 'car' in name or 'truck' in name:
                cars_and_trucks.add(ann['instance_token'])
        token = sample['next']

    print(f"Scene {scene_idx} ({scene['name']}): total car/truck instances = {len(cars_and_trucks)}")
