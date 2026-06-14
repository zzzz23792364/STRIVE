import os
import sys
import json
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader as GraphDataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets import nuscenes_utils as nutils
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from models.traffic_model import TrafficModel
from losses.traffic_model import compute_coll_rate_env, compute_coll_rate_veh
from utils.torch import get_device, load_state
from utils.common import dict2obj, mkdir
from utils.logger import Logger
from utils.scenario_gen import detach_embed_info
from utils.config import get_parser, add_base_args

from rl.policy import GaussianPolicy
from rl.grpo import GRPO


def main():
    parser = get_parser("Phase 1: GRPO + Residual training")
    parser = add_base_args(parser)
    parser.add_argument("--scene_idx", type=int, default=0)
    parser.add_argument("--num_iters", type=int, default=300)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--lr_rl", type=float, default=3e-4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--w_adv_crash", type=float, default=1.0)
    parser.add_argument("--w_motion_prior", type=float, default=0.0)
    parser.add_argument("--w_coll_veh", type=float, default=0.0)
    parser.add_argument("--w_coll_env", type=float, default=0.0)

    args = parser.parse_args()
    config_dict = vars(args)
    cfg = dict2obj(config_dict)

    out_dir = cfg.out
    mkdir(out_dir)
    log_path = os.path.join(out_dir, "phase1_grpo_log.txt")
    Logger.init(log_path)
    Logger.log("Args: " + str(json.dumps(config_dict, indent=2, default=str)))

    device = get_device()
    Logger.log(f"Device: {device}")

    data_path = os.path.join(cfg.data_dir, cfg.data_version)
    map_env = NuScenesMapEnv(data_path, bounds=cfg.map_obs_bounds,
                             L=cfg.map_obs_size_pix, W=cfg.map_obs_size_pix,
                             layers=cfg.map_layers, device=device)
    dataset = NuScenesDataset(data_path, map_env, version=cfg.data_version,
                              split="val", categories=cfg.agent_types,
                              npast=cfg.past_len, nfuture=cfg.future_len,
                              reduce_cats=cfg.reduce_cats)
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    model = TrafficModel(cfg.past_len, cfg.future_len, cfg.map_obs_size_pix,
                         len(dataset.categories), output_bicycle=cfg.model_output_bicycle,
                         conv_kernel_list=cfg.conv_kernel_list,
                         conv_stride_list=cfg.conv_stride_list,
                         conv_filter_list=cfg.conv_filter_list).to(device)
    if cfg.ckpt is not None:
        ckpt_epoch, _ = load_state(cfg.ckpt, model, map_location=device)
        Logger.log(f"Loaded checkpoint epoch {ckpt_epoch}")
    model.set_normalizer(dataset.get_state_normalizer())
    model.set_att_normalizer(dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    for i, data in enumerate(loader):
        if i == cfg.scene_idx:
            scene_graph, map_idx = data
            scene_graph, map_idx = scene_graph.to(device), map_idx.to(device)
            break
    Logger.log(f"Scene {cfg.scene_idx}: {scene_graph}")

    model.eval()
    with torch.no_grad():
        embed_info_attached = model.embed(scene_graph, map_idx, map_env)
    embed_info = detach_embed_info(embed_info_attached)

    NA = scene_graph.future_gt.size(0)
    B = map_idx.size(0)
    ptr = scene_graph.ptr
    ego_inds = ptr[:-1]
    ego_mask = torch.zeros((NA), dtype=torch.bool, device=device)
    ego_mask[ego_inds] = True

    # Identify attack agent: closest non-ego to ego using posterior decode
    with torch.no_grad():
        z_post = embed_info["posterior_out"][0]
        dec = model.decode_embedding(z_post, embed_info, scene_graph, map_idx, map_env)
        fut = model.get_normalizer().unnormalize(dec["future_pred"])
        ego_pos = fut[0:1, :, :2]
        other_pos = fut[1:, :, :2]
        dists = torch.norm(other_pos - ego_pos, dim=-1).min(dim=-1)[0]
        attack_agt_idx = dists.argmin().item() + 1

    Logger.log(f"Attack agent: index {attack_agt_idx} (min_dist={dists.min().item():.3f}m)")

    # Build obs for attack agent only
    obs_full = torch.cat([embed_info["map_feat"], embed_info["past_feat"],
                           embed_info["prior_out"][0], embed_info["prior_out"][1],
                           scene_graph.lw, scene_graph.sem], dim=-1)
    obs_atk = obs_full[attack_agt_idx:attack_agt_idx+1]
    obs_dim = obs_atk.size(-1)
    action_dim = cfg.latent_size

    # Build z_base: ego=prior, others=posterior (except attacker will be replaced)
    z_base = embed_info["posterior_out"][0].detach().clone()
    z_base[ego_mask] = embed_info["prior_out"][0][ego_mask].detach()

    # Policy
    policy = GaussianPolicy(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr_rl)
    grpo = GRPO(policy, optimizer, clip_eps=cfg.clip_eps,
                group_size=cfg.group_size, device=device)

    # GRPO training loop
    Logger.log(f"Starting GRPO ({cfg.num_iters} iters, K={cfg.group_size})...")

    best_reward = -float("inf")
    rewards_history = []
    min_dists = []

    for it in range(cfg.num_iters):
        delta_z, log_probs, adv, rewards, best_z_attack = grpo.sample_and_score(
            obs_atk, z_base, attack_agt_idx, cfg.scale, model,
            embed_info_attached, scene_graph, map_idx, map_env,
        )
        stats = grpo.update(obs_atk, delta_z, log_probs, adv)

        r_mean = rewards.mean().item()
        r_best = rewards.max().item()
        rewards_history.append(r_mean)

        if r_best > best_reward:
            best_reward = r_best

        # Check min distance for best sample
        with torch.no_grad():
            z_full = z_base.clone()
            post_atk = embed_info_attached["posterior_out"][0][attack_agt_idx]
            z_full[attack_agt_idx] = post_atk.detach() + torch.tanh(best_z_attack.unsqueeze(0)) * cfg.scale
            dec = model.decode_embedding(z_full.unsqueeze(1), embed_info_attached,
                                          scene_graph, map_idx, map_env)
            fp = model.get_normalizer().unnormalize(dec["future_pred"][:, 0])
            ego_pos = fp[0:1, :, :2]
            atk_pos = fp[attack_agt_idx:attack_agt_idx+1, :, :2]
            min_d = torch.norm(atk_pos - ego_pos, dim=-1).min().item()
            min_dists.append(min_d)

        if it % 20 == 0 or it == cfg.num_iters - 1:
            msg = (f"It {it:4d} | R_avg={r_mean:+.2f} R_best={r_best:+.2f} "
                   f"min_d={min_d:.3f}m"
                   f" | policy_loss={stats.get('policy_loss', 0):.4f}")
            if not stats.get("skipped", False):
                pass
            Logger.log(msg)
            print(msg)

    final_min_d = np.min(min_dists[-50:]) if len(min_dists) >= 50 else np.min(min_dists)
    Logger.log("=" * 60)
    Logger.log(f"Phase 1 GRPO Results")
    Logger.log(f"  Iterations: {cfg.num_iters}")
    Logger.log(f"  Group size: {cfg.group_size}")
    Logger.log(f"  Best reward: {best_reward:+.2f}")
    Logger.log(f"  Best min_dist: {np.min(min_dists):.3f}m")
    Logger.log(f"  Final min_dist (avg last 50): {final_min_d:.3f}m")
    Logger.log("=" * 60)

    ckpt_path = os.path.join(out_dir, "policy.pth")
    torch.save(policy.state_dict(), ckpt_path)
    results = {
        "args": config_dict,
        "rewards": rewards_history,
        "min_dists": min_dists,
        "best_reward": best_reward,
        "best_min_dist": float(np.min(min_dists)),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    Logger.log("Done.")


if __name__ == "__main__":
    main()
