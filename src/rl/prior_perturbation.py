"""Prior Perturbation Probe: 用 sigma 倍数扰动 prior_z, decode, 检查碰撞.

实验目的 (见 docs/exp1_prior_perturbation_plan.md):
  1. sigma 几倍能撞到
  2. 每 sigma 能填多少 BD cell
  3. min_dist 分布
  4. prior 物理可塑性
"""
import torch
import numpy as np
from utils.transforms import transform2frame


def angle_to_bin(angle_deg):
    """[-180, 180) -> {0,1,2,3}, 中心 0/90/180/270 度, 边在 +/-45."""
    a = (angle_deg + 45.0) % 360.0
    return int(a // 90.0)


def fast_collision_check_vectorized(decoded_atk, ego_replay, atk_lw, ego_lw,
                                     center_threshold=1.5):
    """快速向量化碰撞检测：用中心距离 + 简单包围盒.

    真实 shapely IoU 太慢 (768 次调用 ~10 分钟), 改用近似:
      - 算 atk 和 ego 在每个时刻的中心距离
      - 用 lw 的几何平均做阈值
      - 比 shapely 严格: 距离 < (avg_lw) 视为碰撞

    Args:
        decoded_atk: (n_samples, FT, 4) unnormalized
        ego_replay: (FT, 4) unnormalized
        atk_lw, ego_lw: (2,) unnormalized [length, width]
    Returns:
        collides: (n_samples,) bool
        coll_times: (n_samples,) int
        min_dists: (n_samples,) float
    """
    n_samples = decoded_atk.size(0)
    FT = decoded_atk.size(1)

    # 中心距离: (n_samples, FT)
    dist = torch.norm(decoded_atk[:, :, :2] - ego_replay[:, :2].unsqueeze(0), dim=-1)
    min_dists, min_idx = dist.min(dim=-1)  # (n_samples,)

    # 阈值: 用 lw 的几何近似碰撞
    avg_lw = (atk_lw.mean() + ego_lw.mean()) / 2.0
    threshold = avg_lw * 0.5  # 保守一点 (车长 4m, 阈值 1.0m)

    collides = min_dists < threshold
    coll_times = min_idx  # 第一次距离最小 = 接近时刻
    return collides, coll_times, min_dists


def compute_bd_from_collision(ego_replay_traj, decoded_atk_traj, coll_time):
    """在 collision moment 算 (pos_angle, heading_angle) 在 ego 局部坐标系.

    Args:
        ego_replay_traj: (FT, 4) unnormalized [x, y, hx, hy]
        decoded_atk_traj: (FT, 4) unnormalized
        coll_time: int, T index of first collision
    Returns:
        bd_idx: int in [0, 15] or -1
        pos_angle_deg: float
        heading_angle_deg: float
    """
    ego_state = ego_replay_traj[coll_time].unsqueeze(0)  # (1, 4)
    atk_state = decoded_atk_traj[coll_time].unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
    local_atk = transform2frame(ego_state, atk_state)[0, 0]  # (4,) = [x, y, hx, hy]

    pos_angle = torch.atan2(local_atk[1], local_atk[0]).item() * 180.0 / np.pi
    heading_angle = torch.atan2(local_atk[3], local_atk[2]).item() * 180.0 / np.pi

    pos_bin = angle_to_bin(pos_angle)
    h_bin = angle_to_bin(heading_angle)
    bd_idx = pos_bin * 4 + h_bin
    return bd_idx, pos_angle, heading_angle


def probe_prior_sigma(
    sg, mi, map_env, model, embed_info,
    ego_mask, atk_idx,
    sigmas=(1, 3, 5, 7, 9, 11),
    n_samples=128,
    seed=42,
    batch_size=32,
):
    """核心探测函数 (向量化版本).

    Args:
        sg, mi, map_env, model, embed_info: 已加载的场景和模型
        ego_mask: (NA,) bool, True for ego
        atk_idx: int, attack agent index
        sigmas: tuple of sigma multipliers
        n_samples: int, samples per sigma
        seed: int, random seed for reproducibility
        batch_size: int, decode batch size
    Returns:
        dict[sigma] -> list of sample dicts
    """
    device = sg.future_gt.device
    norm = model.get_normalizer()
    att_norm = model.get_att_normalizer()

    prior_mu, prior_var = embed_info['prior_out']  # (NA, 32)
    sigma_prior = torch.sqrt(prior_var)            # (NA, 32)

    ego_replay = norm.unnormalize(sg.future_gt[ego_mask][:, :, :4])  # (1, FT, 4)
    ego_replay_traj = ego_replay[0]                                    # (FT, 4)
    ego_lw = att_norm.unnormalize(sg.lw[ego_mask])[0]                  # (2,)
    atk_lw = att_norm.unnormalize(sg.lw[~ego_mask])[0]                 # (2,)

    NA = sg.future_gt.size(0)

    torch.manual_seed(seed)
    results = {}

    for mult in sigmas:
        samples = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        sample_idx = 0
        for b in range(n_batches):
            cur_bs = min(batch_size, n_samples - sample_idx)
            # 只扰动 atk (atk_idx), ego (其他非 atk 索引) 保持 prior_mu 不变
            eps = torch.randn(NA, cur_bs, prior_mu.size(-1), device=device) * mult
            # z shape: (NA, BS, 32) - 默认保持 prior_mu, 只在 atk 槽位加扰动
            z = prior_mu.unsqueeze(1) + sigma_prior.unsqueeze(1) * eps
            # 修复: ego (非 atk) 不扰动
            ego_idx_list = [i for i in range(NA) if i != atk_idx]
            z[ego_idx_list] = prior_mu[ego_idx_list].unsqueeze(1)

            with torch.no_grad():
                dec = model.decode_embedding(z, embed_info, sg, mi, map_env)
                fut = norm.unnormalize(dec['future_pred'])  # (NA, BS, FT, 4)

            decoded_atk = fut[atk_idx, :, :, :4]  # (BS, FT, 4)

            # 向量化碰撞检测
            collides, coll_times, min_dists = fast_collision_check_vectorized(
                decoded_atk, ego_replay_traj, atk_lw, ego_lw
            )

            for k in range(cur_bs):
                c = bool(collides[k].item())
                ct = int(coll_times[k].item()) if c else -1
                md = float(min_dists[k].item())

                if c and 0 <= ct < decoded_atk.size(1):
                    bd_idx, pos_ang, h_ang = compute_bd_from_collision(
                        ego_replay_traj, decoded_atk[k], ct
                    )
                else:
                    bd_idx, pos_ang, h_ang = -1, None, None

                samples.append({
                    'sample_id': sample_idx,
                    'sigma_mult': mult,
                    'collides': c,
                    'coll_time': ct if c else None,
                    'min_dist': md,
                    'bd_idx': bd_idx,
                    'pos_angle': pos_ang,
                    'heading_angle': h_ang,
                    'z': z[:, k, :].cpu(),  # (NA, 32) - full z matrix
                    'future_pred': dec['future_pred'][:, k, :, :].cpu(),  # (NA, FT, 4) NORMALIZED for viz
                })
                sample_idx += 1

        results[mult] = samples

    return results


def summarize_results(results):
    """统计每档 sigma 的关键指标."""
    summary = {}
    for mult, samples in results.items():
        n = len(samples)
        n_coll = sum(1 for s in samples if s['collides'])
        coll_md = [s['min_dist'] for s in samples if s['collides']]
        bd_filled = set(s['bd_idx'] for s in samples if s['collides'])
        coll_times = [s['coll_time'] for s in samples if s['collides']

                     and s['coll_time'] is not None]

        if coll_md:
            md_arr = np.array(coll_md)
            md_q25, md_med, md_q75 = np.percentile(md_arr, [25, 50, 75])
        else:
            md_q25 = md_med = md_q75 = float('nan')

        summary[mult] = {
            'n_samples': n,
            'n_collide': n_coll,
            'collision_rate': n_coll / n,
            'min_dist_median': float(md_med),
            'min_dist_q25': float(md_q25),
            'min_dist_q75': float(md_q75),
            'bd_coverage': len(bd_filled),
            'bd_filled_list': sorted(bd_filled),
            'avg_coll_time': float(np.mean(coll_times)) if coll_times else None,
        }
    return summary
