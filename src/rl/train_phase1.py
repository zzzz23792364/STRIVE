import os
import sys
import time
import json

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader as GraphDataLoader
from torch_geometric.data import Batch as GraphBatch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets import nuscenes_utils as nutils
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.common import dict2obj, mkdir
from utils.logger import Logger
from utils.scenario_gen import detach_embed_info
from utils.config import get_parser, add_base_args

from rl.policy import GaussianPolicy
from rl.reward import RLReward
from rl.reinforce import REINFORCE
from rl.ppo import PPO


def build_obs(embed_info, scene_graph):
    map_feat = embed_info["map_feat"]
    past_feat = embed_info["past_feat"]
    prior_mu, prior_var = embed_info["prior_out"]
    obs = torch.cat(
        [map_feat, past_feat, prior_mu, prior_var, scene_graph.lw, scene_graph.sem],
        dim=-1,
    )
    return obs


def compute_adv_success(future_pred, scene_graph, model, attack_agt_idx):
    from losses.adv_gen_nusc import check_single_veh_coll

    NA = future_pred.size(0)
    ego_idx = 0
    traj = model.get_normalizer().unnormalize(future_pred)
    lw = model.get_att_normalizer().unnormalize(scene_graph.lw)
    planner_traj = traj[ego_idx]
    planner_lw = lw[ego_idx]
    other_traj = traj[1:]
    other_lw = lw[1:]
    coll, _ = check_single_veh_coll(planner_traj, planner_lw, other_traj, other_lw)
    return bool(coll[attack_agt_idx - 1])


def run_gradient_baseline(model, scene_graph, map_idx, map_env, embed_info,
                           loss_weights, num_iters=300, lr=0.05, device="cuda"):
    from utils.adv_gen_optim import run_adv_gen_optim, compute_adv_gen_success

    NA = scene_graph.future_gt.size(0)
    B = map_idx.size(0)
    ego_inds = scene_graph.ptr[:-1]
    ego_mask = torch.zeros((NA), dtype=torch.bool)
    ego_mask[ego_inds] = True

    z_init = embed_info["posterior_out"][0].detach()

    tgt_prior = (embed_info["prior_out"][0][ego_mask], embed_info["prior_out"][1][ego_mask])
    other_prior = (embed_info["prior_out"][0][~ego_mask], embed_info["prior_out"][1][~ego_mask])

    result = run_adv_gen_optim(
        z_init, lr, loss_weights, model, scene_graph, map_env, map_idx,
        num_iters, embed_info, "ego", tgt_prior, other_prior,
        feasibility_time=0, feasibility_infront_min=None,
    )
    cur_z, final_traj, decoder_out, min_agt, min_t = result

    success = False
    scene_sg = scene_graph.to_data_list()[0]
    if min_agt is not None and len(min_agt) > 0:
        success = compute_adv_gen_success(
            final_traj, model,
            GraphBatch.from_data_list([scene_sg]),
            min_agt[0] - scene_graph.ptr[0].item(),
        )
    return {
        "success": success,
        "z": cur_z,
        "traj": final_traj,
        "min_agt": min_agt,
        "min_t": min_t,
    }


def warmup_policy(policy, target_z, obs, optimizer, steps=50, lr=1e-3, device="cuda"):
    target_z = target_z.detach()
    for step in range(steps):
        optimizer.zero_grad()
        mu, std = policy(obs)
        loss = ((mu - target_z) ** 2).mean()
        loss.backward()
        optimizer.step()
    Logger.log(f"Warmup done: final imitation loss = {loss.item():.6f}")


def main():
    parser = get_parser("Phase 1: RL single-scenario training")
    parser = add_base_args(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene_idx", type=int, default=0)
    parser.add_argument("--rl_algo", type=str, default="ppo", choices=["reinforce", "ppo"])
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--lr_rl", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--grad_iters", type=int, default=300)
    parser.add_argument("--compare_baseline", type=lambda x: x.lower() == "true", default=True)

    args = parser.parse_args()
    config_dict = vars(args)
    cfg = dict2obj(config_dict)

    out_dir = cfg.out
    mkdir(out_dir)
    log_path = os.path.join(out_dir, "phase1_log.txt")
    Logger.init(log_path)
    Logger.log("Args: " + str(json.dumps(config_dict, indent=2, default=str)))

    device = get_device()
    Logger.log(f"Using device: {device}")

    data_path = os.path.join(cfg.data_dir, cfg.data_version)
    map_env = NuScenesMapEnv(
        data_path,
        bounds=cfg.map_obs_bounds,
        L=cfg.map_obs_size_pix,
        W=cfg.map_obs_size_pix,
        layers=cfg.map_layers,
        device=device,
    )
    dataset = NuScenesDataset(
        data_path, map_env,
        version=cfg.data_version,
        split="val",
        categories=cfg.agent_types,
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        reduce_cats=cfg.reduce_cats,
    )

    loader = GraphDataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=False,
    )

    model = TrafficModel(
        cfg.past_len, cfg.future_len, cfg.map_obs_size_pix, len(dataset.categories),
        map_feat_size=cfg.map_feat_size, past_feat_size=cfg.past_feat_size,
        future_feat_size=cfg.future_feat_size, latent_size=cfg.latent_size,
        output_bicycle=cfg.model_output_bicycle,
        conv_channel_in=map_env.num_layers,
        conv_kernel_list=cfg.conv_kernel_list,
        conv_stride_list=cfg.conv_stride_list,
        conv_filter_list=cfg.conv_filter_list,
    ).to(device)

    if cfg.ckpt is not None:
        ckpt_epoch, _ = load_state(cfg.ckpt, model, map_location=device)
        Logger.log(f"Loaded checkpoint from epoch {ckpt_epoch}")
    model.set_normalizer(dataset.get_state_normalizer())
    model.set_att_normalizer(dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    for i, data in enumerate(loader):
        if i == cfg.scene_idx:
            scene_graph, map_idx = data
            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)
            break
    Logger.log(f"Scene {cfg.scene_idx}: {scene_graph}")
    NA = scene_graph.future_gt.size(0)
    B = map_idx.size(0)
    ego_inds = scene_graph.ptr[:-1]

    model.eval()
    with torch.no_grad():
        embed_info_attached = model.embed(scene_graph, map_idx, map_env)
    embed_info = detach_embed_info(embed_info_attached)

    obs_full = build_obs(embed_info, scene_graph)
    obs_dim = obs_full.size(-1)
    action_dim = cfg.latent_size
    Logger.log(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    ego_mask = torch.zeros((NA), dtype=torch.bool).to(device)
    ego_mask[ego_inds] = True

    loss_weights = {
        "adv_crash": 2.0,
        "motion_prior": 1.0,
        "coll_veh": 20.0,
        "coll_env": 20.0,
    }
    reward_fn = RLReward(
        loss_weights,
        model.get_att_normalizer().unnormalize(scene_graph.lw),
        map_idx[scene_graph.batch],
        map_env,
        scene_graph.ptr,
    ).to(device)

    policy = GaussianPolicy(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr_rl)

    # Warmup: initialize policy to output posterior z (imitation learning)
    posterior_z = embed_info_attached["posterior_out"][0].detach()
    Logger.log(f"Warming up policy with posterior init ({cfg.warmup_steps} steps)...")
    warmup_policy(policy, posterior_z, obs_full, optimizer, steps=cfg.warmup_steps, device=device)

    # RL
    reinforce = REINFORCE(policy, optimizer, gamma=cfg.gamma, device=device)
    ppo_algo = PPO(policy, optimizer, gamma=cfg.gamma, clip_eps=cfg.clip_eps,
                   entropy_coef=cfg.entropy_coef, device=device)

    # Gradient baseline
    grad_result = None
    if cfg.compare_baseline:
        Logger.log("Running gradient baseline (300 iters)...")
        bl_loss_weights = {
            "adv_crash": 2.0, "coll_veh": 20.0, "coll_env": 20.0,
            "motion_prior": 1.0, "motion_prior_atk": 0.005,
            "init_z": 0.5, "init_z_atk": 0.05,
            "coll_veh_plan": 20.0,
            "motion_prior_ext": 0.0001, "match_ext": 10.0,
        }
        grad_result = run_gradient_baseline(
            model, scene_graph, map_idx, map_env, embed_info,
            bl_loss_weights, num_iters=cfg.grad_iters, device=device,
        )
        Logger.log(f"Gradient baseline success: {grad_result['success']}")

    Logger.log(f"Starting RL training ({cfg.rl_algo.upper()}, {cfg.num_episodes} episodes)...")
    model.eval()

    episode_rewards = []
    episode_successes = []
    best_reward = -float("inf")

    # Pre-compute when possible
    tgt_traj = scene_graph.future_gt[ego_mask][:, :, :4]
    prior_ego = (embed_info["prior_out"][0][ego_mask], embed_info["prior_out"][1][ego_mask])

    for ep in range(cfg.num_episodes):
        obs_list, z_list, log_prob_list, reward_list, value_list = [], [], [], [], []

        z_non_ego, log_prob, mu, std = policy.sample(obs_full[~ego_mask])

        z_full = torch.cat([embed_info["prior_out"][0][ego_mask], z_non_ego.detach()], dim=0)

        with torch.no_grad():
            dec_out = model.decode_embedding(
                z_full, embed_info, scene_graph, map_idx, map_env
            )
            future_pred = dec_out["future_pred"]

        z_other = z_full[~ego_mask]
        prior_other = (
            embed_info["prior_out"][0][~ego_mask],
            embed_info["prior_out"][1][~ego_mask],
        )

        reward, r_info = reward_fn(
            model.get_normalizer().unnormalize(future_pred),
            model.get_normalizer().unnormalize(tgt_traj),
            z_other, prior_other,
        )

        obs_list.append(obs_full[~ego_mask])
        z_list.append(z_non_ego)
        log_prob_list.append(log_prob)
        reward_list.append(reward.item())

        if cfg.rl_algo == "ppo":
            value_list.append(0.0)

        if reward.item() > best_reward:
            best_reward = reward.item()

        attack_agt = 1
        success = compute_adv_success(
            model.get_normalizer().unnormalize(future_pred), scene_graph, model, attack_agt
        )
        episode_successes.append(success)
        episode_rewards.append(reward.item())

        if cfg.rl_algo == "reinforce":
            reward_list_centered = [r - np.mean(episode_rewards[-50:]) if len(episode_rewards) > 0 else r for r in reward_list]
            stats = reinforce.update(reward_list_centered, log_prob_list)
        elif cfg.rl_algo == "ppo":
            stats = ppo_algo.update(obs_list, z_list, reward_list, log_prob_list, value_list)
            if stats.get("skipped", False):
                Logger.log(f"Ep {ep}: PPO update skipped (NaN loss)")

        if ep % 50 == 0 or ep == cfg.num_episodes - 1:
            recent_r = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            recent_sr = np.mean(episode_successes[-50:]) if len(episode_successes) >= 50 else np.mean(episode_successes)
            msg = f"Ep {ep:4d} | R={reward.item():+.1f} | avgR={recent_r:+.1f} | SR={recent_sr:.2f}"
            for k, v in r_info.items():
                if k != "reward":
                    msg += f" | {k}={v:.2f}"
            if stats:
                for k, v in stats.items():
                    msg += f" | {k}={v:.4f}"
            Logger.log(msg)
            print(msg)

    final_sr = np.mean(episode_successes[-100:]) if len(episode_successes) >= 100 else np.mean(episode_successes)
    Logger.log("=" * 60)
    Logger.log(f"Phase 1 Results ({cfg.rl_algo.upper()})")
    Logger.log(f"  Avg reward (last 100): {np.mean(episode_rewards[-100:]):+.1f}")
    Logger.log(f"  Success rate (last 100): {final_sr:.3f}")
    if grad_result is not None:
        Logger.log(f"  Gradient baseline success: {grad_result['success']}")
    Logger.log("=" * 60)

    ckpt_path = os.path.join(out_dir, "policy.pth")
    torch.save(policy.state_dict(), ckpt_path)

    results = {
        "args": config_dict,
        "episode_rewards": episode_rewards,
        "episode_successes": [int(s) for s in episode_successes],
        "final_success_rate": final_sr,
        "best_reward": best_reward,
        "grad_baseline_success": grad_result["success"] if grad_result else None,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    Logger.log("Results saved.")


if __name__ == "__main__":
    main()
