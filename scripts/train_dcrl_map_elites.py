"""DCRL-MAP-Elites for sub60 multi-solution.

 基于 Faldor et al. 2023 (DCG-MAP-Elites) 和 Faldor et al. 2024 (DCRL-MAP-Elites).
 
 三个 variation operators:
   - GA: Genetic Algorithm (CMA-ES emitters) — 随机探索
   - PG: Policy Gradient with descriptor-conditioned critic — 定向改良
   - AI: Actor Injection — descriptor-conditioned actor 作为 generative model

 区别于 PGA-MAP-Elites:
   - Critic 是 descriptor-conditioned 的: Q(obs, z | bd_target)
   - PG variation 使用 critic gradient 朝 target descriptor 方向变异
   - Actor 通过 TD3 训练, 同时完成 archive distillation
   - AI: actor π(obs | d) 生成 progeny 注入 offspring batch
"""
import os, sys, json, time
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.torch import get_device, load_state
from utils.scenario_gen import detach_embed_info
from datasets.utils import NUSC_BIKE_PARAMS
from rl.prior_perturbation import compute_bd_from_collision, fast_collision_check_vectorized

device = get_device()
print(f"Device: {device}")

OUT = "./out/dcrl_map_elites"
os.makedirs(OUT, exist_ok=True)

# ===== 加载场景 =====
print("Loading val subseq 60...")
data_path = "./data/nuscenes/trainval"
map_env = NuScenesMapEnv(data_path, bounds=[-17.0, -38.5, 60.0, 38.5],
                         L=256, W=256,
                         layers=["drivable_area", "carpark_area",
                                 "road_divider", "lane_divider"],
                         device=device)
dataset = NuScenesDataset(data_path, map_env, version="trainval", split="val",
                           categories=["car", "truck"], npast=4, nfuture=12,
                           reduce_cats=False, seq_interval=10,
                           randomize_val=True, val_size=400)
loader = GraphDataLoader(dataset, batch_size=1, shuffle=False,
                          num_workers=0, pin_memory=False)
for i, data in enumerate(loader):
    if i == 60:
        sg, mi = data; sg, mi = sg.to(device), mi.to(device)
        break

NA = sg.future_gt.size(0)
ptr = sg.ptr
ego_mask = torch.zeros(NA, dtype=torch.bool, device=device)
ego_mask[ptr[:-1]] = True
atk_idx = 1

print("Loading model...")
model = TrafficModel(4, 12, 256, len(dataset.categories), output_bicycle=True,
                     conv_kernel_list=[7,5,5,3,3,3],
                     conv_stride_list=[2,2,2,2,2,2],
                     conv_filter_list=[16,32,64,64,128,128]).to(device)
load_state("./model_ckpt/traffic_model.pth", model)
model.set_normalizer(dataset.get_state_normalizer())
model.set_att_normalizer(dataset.get_att_normalizer())
model.set_bicycle_params(NUSC_BIKE_PARAMS)
model.eval()

with torch.no_grad():
    ei = model.embed(sg, mi, map_env)
embed_info = detach_embed_info(ei)
prior_mu, prior_var = ei['prior_out']
sigma_prior = torch.sqrt(prior_var)
norm = model.get_normalizer()

ego_replay = norm.unnormalize(sg.future_gt[ego_mask][:, :, :4])[0]
ego_lw = model.get_att_normalizer().unnormalize(sg.lw[ego_mask])[0]
atk_lw = model.get_att_normalizer().unnormalize(sg.lw[~ego_mask])[0]


def build_obs(embed_info, sg):
    map_feat = embed_info['map_feat']
    past_feat = embed_info['past_feat']
    obs = torch.cat([map_feat, past_feat, embed_info['prior_out'][0],
                     embed_info['prior_out'][1], sg.lw, sg.sem], dim=-1)
    return obs

obs = build_obs(embed_info, sg)
obs_atk = obs[atk_idx:atk_idx+1]
OBS_DIM = obs_atk.size(-1)
obs_atk_np = obs_atk.cpu().numpy()[0]
print(f"obs_atk shape: {obs_atk.shape}, obs_dim={OBS_DIM}")


# ===== Evaluate =====
@torch.no_grad()
def evaluate_z(z_atk_batch):
    """给定 atk z batch (BS, 32), 评估 (collides, min_dist, bd_idx)."""
    BS = z_atk_batch.size(0)
    ego_z = prior_mu[0].unsqueeze(0).expand(BS, 32)
    z_full = torch.stack([ego_z, z_atk_batch], dim=0)
    dec = model.decode_embedding(z_full, embed_info, sg, mi, map_env)
    fut = norm.unnormalize(dec['future_pred'])
    decoded_atk = fut[atk_idx, :, :, :4]
    collides, coll_times, min_dists = fast_collision_check_vectorized(
        decoded_atk, ego_replay, atk_lw, ego_lw
    )
    bd_indices = np.full(BS, -1, dtype=np.int64)
    for k in range(BS):
        if collides[k]:
            ct = int(coll_times[k].item())
            if 0 <= ct < decoded_atk.size(1):
                bd_idx, _, _ = compute_bd_from_collision(ego_replay, decoded_atk[k], ct)
                bd_indices[k] = bd_idx
    return collides.cpu().numpy(), min_dists.cpu().numpy(), bd_indices, fut


# ===== Networks =====
class DescriptorConditionedCritic(nn.Module):
    """Q(obs, z | bd_target). 评估: z 能达成 bd_target 吗?"""
    def __init__(self, obs_dim, z_dim=32, bd_dim=16, hidden=256):
        super().__init__()
        self.bd_embed = nn.Linear(bd_dim, 32)
        in_dim = obs_dim + z_dim + 32
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, obs, z, bd_onehot):
        bd_emb = F.relu(self.bd_embed(bd_onehot))
        x = torch.cat([obs, z, bd_emb], dim=-1)
        return self.net(x).squeeze(-1)

class DescriptorConditionedActor(nn.Module):
    """π(obs | bd_target) → z. Descriptor-conditioned policy."""
    def __init__(self, obs_dim, bd_dim=16, z_dim=32, hidden=256):
        super().__init__()
        self.bd_embed = nn.Linear(bd_dim, 32)
        in_dim = obs_dim + 32
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, z_dim)
    def forward(self, obs, bd_onehot):
        bd_emb = F.relu(self.bd_embed(bd_onehot))
        x = torch.cat([obs, bd_emb], dim=-1)
        h = self.backbone(x)
        return self.mu_head(h)


actor = DescriptorConditionedActor(obs_dim=OBS_DIM, bd_dim=16, z_dim=32).to(device)
critic1 = DescriptorConditionedCritic(obs_dim=OBS_DIM, z_dim=32, bd_dim=16).to(device)
critic2 = DescriptorConditionedCritic(obs_dim=OBS_DIM, z_dim=32, bd_dim=16).to(device)
import copy
critic1_target = copy.deepcopy(critic1)
critic2_target = copy.deepcopy(critic2)
for p in critic1_target.parameters(): p.requires_grad = False
for p in critic2_target.parameters(): p.requires_grad = False
actor_target = copy.deepcopy(actor)
for p in actor_target.parameters(): p.requires_grad = False

actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
critic_optim = torch.optim.Adam(
    list(critic1.parameters()) + list(critic2.parameters()), lr=3e-4
)


# ===== Replay Buffer + pyribs Archive =====
class DCRLReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    def add(self, obs, z, reward, bd_observed, bd_target):
        self.buffer.append({
            'obs': obs, 'z': z, 'reward': reward,
            'bd_observed': bd_observed, 'bd_target': bd_target
        })
    def __len__(self): return len(self.buffer)
    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        return {
            'obs': torch.tensor(np.stack([b['obs'] for b in batch]), dtype=torch.float32).to(device),
            'z': torch.tensor(np.stack([b['z'] for b in batch]), dtype=torch.float32).to(device),
            'reward': torch.tensor([b['reward'] for b in batch], dtype=torch.float32).to(device),
            'bd_obs': torch.tensor([b['bd_observed'] for b in batch], dtype=torch.long).to(device),
            'bd_target': torch.tensor([b['bd_target'] for b in batch], dtype=torch.long).to(device),
        }

buffer = DCRLReplayBuffer(capacity=50000)

solution_dim = 32
archive = GridArchive(solution_dim=solution_dim, dims=(4, 4), ranges=((0, 4), (0, 4)), seed=42)

emitters = []
for pos_bin in range(4):
    for h_bin in range(4):
        init_z = prior_mu[atk_idx].cpu().numpy() + sigma_prior[atk_idx].cpu().numpy() * np.random.randn(32) * 0.3
        emitter = EvolutionStrategyEmitter(archive, x0=init_z, sigma0=2.0,
                                            ranker="2imp", es="cma_es", batch_size=8)
        emitters.append(emitter)
scheduler = Scheduler(archive, emitters)

# ===== Hyperparams =====
N_ITERS = 200
BATCH_SIZE = 64       # critic training batch
GAIN_PENALTY = -10.0  # 撞错 cell 惩罚
ACTOR_INJECTION_START = 30
POLICY_UPDATE_DELAY = 2  # TD3-style: actor update every N steps
TAU = 0.005

print(f"\nDCRL-MAP-Elites: {N_ITERS} iters, startup={ACTOR_INJECTION_START}")
best_md_per_cell = np.full(16, np.inf)
rng = np.random.default_rng(42)
critic_losses, actor_losses = [], []
update_step = 0

start_t = time.time()

for it in range(N_ITERS):
    # ===== 1. GA variation (CMA-ES) =====
    solutions_ga = scheduler.ask()

    # ===== 2. PG variation (descriptor-conditioned critic gradient) =====
    solutions_pg = []
    if len(buffer) >= BATCH_SIZE and it > 10:
        for k in range(16):  # 每个 cell 一个 PG 候选
            bd_target = k
            bd_oh = torch.zeros(1, 16, device=device); bd_oh[0, bd_target] = 1.0
            obs_t = torch.tensor(obs_atk_np, device=device, dtype=torch.float32).unsqueeze(0)
            # 从 archive 里选一个该 cell 的 existing solution 做 parent
            parent_z = prior_mu[atk_idx].cpu().numpy() + np.random.randn(32) * 0.5
            z_t = torch.tensor(parent_z, device=device, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
            # PG: maximize critic Q toward target descriptor
            q_val = critic1(obs_t, z_t, bd_oh)
            grad_z = torch.autograd.grad(q_val.sum(), z_t, create_graph=False)[0]
            z_mutated = z_t.detach() + 0.1 * grad_z
            solutions_pg.append(z_mutated.squeeze(0).cpu().numpy())

    # ===== 3. AI: Actor Injection =====
    solutions_ai = []
    if it >= ACTOR_INJECTION_START:
        actor.eval()
        with torch.no_grad():
            # 均匀覆盖 4 个 reachable cells
            for g in [2, 6, 10, 14]:
                for _ in range(4):
                    bd_oh = torch.zeros(1, 16, device=device); bd_oh[0, g] = 1.0
                    obs_t = torch.tensor(obs_atk_np, device=device, dtype=torch.float32).unsqueeze(0)
                    z = actor(obs_t, bd_oh) + 0.3 * torch.randn(32, device=device)
                    solutions_ai.append(z.squeeze(0).cpu().numpy())
        actor.train()

    # ===== 4. Evaluate all =====
    all_solutions = solutions_ga.tolist() + solutions_pg + solutions_ai
    if len(all_solutions) == 0:
        all_solutions = solutions_ga

    z_batch = torch.tensor(np.array(all_solutions), device=device, dtype=torch.float32)
    collides, min_dists, bd_indices, fut = evaluate_z(z_batch)

    # Write to archive
    objectives = -min_dists.astype(np.float64)
    measures = np.full((len(all_solutions), 2), -1.0, dtype=np.float32)
    for k in range(len(all_solutions)):
        if bd_indices[k] >= 0:
            measures[k, 0] = bd_indices[k] // 4
            measures[k, 1] = bd_indices[k] % 4
    scheduler.tell(objectives[:len(solutions_ga)], measures[:len(solutions_ga)])
    # PG and AI solutions added manually to archive
    for k in range(len(solutions_ga), len(all_solutions)):
        bd_idx = bd_indices[k]
        if bd_idx >= 0:
            obj = objectives[k]
            m = (bd_idx // 4, bd_idx % 4)
            archive.add_single(all_solutions[k], obj, m)

    # Update best
    for k in range(len(all_solutions)):
        if bd_indices[k] >= 0 and min_dists[k] < best_md_per_cell[bd_indices[k]]:
            best_md_per_cell[bd_indices[k]] = min_dists[k]

    # ===== 5. Store transitions in replay buffer =====
    for k in range(len(all_solutions)):
        reward = -min_dists[k]
        bd_obs = int(bd_indices[k]) if bd_indices[k] >= 0 else -1
        # target descriptor: the cell this z was supposed to hit
        if k < len(solutions_ga):
            bd_target = int(measures[k, 0]) * 4 + int(measures[k, 1]) if measures[k, 0] >= 0 else int(rng.integers(0, 16))
        elif k - len(solutions_ga) < len(solutions_pg):
            bd_target = (k - len(solutions_ga)) % 16  # PG: one per cell
        else:
            ai_idx = k - len(solutions_ga) - len(solutions_pg)
            bd_target = [2,6,10,14][(ai_idx // 4) % 4]  # AI: reachable cells
        buffer.add(obs_atk_np, all_solutions[k], float(reward), bd_obs, bd_target)

    # ===== 6. Actor-Critic Training (TD3) =====
    if len(buffer) >= BATCH_SIZE:
        for _ in range(2):  # 2 updates per iter
            batch = buffer.sample(BATCH_SIZE)
            bd_obs_oh = F.one_hot(batch['bd_obs'].clamp(min=0), num_classes=16).float()
            bd_target_oh = F.one_hot(batch['bd_target'], num_classes=16).float()

            # Descriptor-aware reward shaping (DCRL Eq.5):
            # S(d, d') = 1 if bd_observed == bd_target else 0
            s_mask = (batch['bd_obs'] == batch['bd_target']).float()  # successor mask
            shaped_reward = s_mask * batch['reward'] + (1 - s_mask) * GAIN_PENALTY

            # Critic update
            with torch.no_grad():
                target_z = actor_target(batch['obs'], bd_target_oh)
                noise = torch.clamp(torch.randn_like(target_z) * 0.2, -0.5, 0.5)
                target_z = target_z + noise
                q1_t = critic1_target(batch['obs'], target_z, bd_target_oh)
                q2_t = critic2_target(batch['obs'], target_z, bd_target_oh)
                q_target = shaped_reward + 0.99 * torch.min(q1_t, q2_t)

            q1_pred = critic1(batch['obs'], batch['z'], bd_target_oh)
            q2_pred = critic2(batch['obs'], batch['z'], bd_target_oh)
            critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), 1.0)
            critic_optim.step()
            critic_losses.append(critic_loss.item())

            # Actor update (delayed, TD3-style)
            if update_step % POLICY_UPDATE_DELAY == 0:
                pred_z = actor(batch['obs'], bd_target_oh)
                actor_loss = -critic1(batch['obs'], pred_z, bd_target_oh).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optim.step()
                actor_losses.append(actor_loss.item())

                # Soft target update
                with torch.no_grad():
                    for p, p_t in zip(critic1.parameters(), critic1_target.parameters()):
                        p_t.mul_(1 - TAU).add_(TAU * p)
                    for p, p_t in zip(critic2.parameters(), critic2_target.parameters()):
                        p_t.mul_(1 - TAU).add_(TAU * p)
                    for p, p_t in zip(actor.parameters(), actor_target.parameters()):
                        p_t.mul_(1 - TAU).add_(TAU * p)

            update_step += 1

    # ===== 7. Print =====
    if it % 20 == 0 or it == N_ITERS - 1:
        elapsed = time.time() - start_t
        n_filled = sum(1 for m in best_md_per_cell if m < np.inf)
        n_col = sum(1 for c in collides if c)
        last_c = np.mean(critic_losses[-50:]) if critic_losses else 0
        last_a = np.mean(actor_losses[-50:]) if actor_losses else 0
        print(f"  it {it}/{N_ITERS} ({elapsed:.0f}s): "
              f"cells_filled={n_filled}/16, this_collided={n_col}/{len(all_solutions)}, "
              f"critic={last_c:.3f}, actor={last_a:.3f}")
        for c in range(16):
            if best_md_per_cell[c] < np.inf:
                print(f"    cell {c:2d}: md={best_md_per_cell[c]:.3f}m")

# ===== Save =====
print("\n=== Final Results ===")
for c in range(16):
    if best_md_per_cell[c] < np.inf:
        print(f"  cell {c:2d}: {best_md_per_cell[c]:.3f}m")

torch.save(actor.state_dict(), os.path.join(OUT, "actor.pt"))
torch.save(critic1.state_dict(), os.path.join(OUT, "critic.pt"))

stats = {
    "n_iters": N_ITERS, "best_md_per_cell": {str(k): float(v) for k, v in best_md_per_cell.items()},
    "critic_losses": critic_losses, "actor_losses": actor_losses,
}
with open(os.path.join(OUT, "training_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

# ===== Inference =====
print("\n=== Actor inference: 4 reachable cells ===")
actor.eval()
with torch.no_grad():
    obs_t = torch.tensor(obs_atk_np, device=device, dtype=torch.float32).unsqueeze(0)
    for g in [2, 6, 10, 14]:
        bd_oh = torch.zeros(1, 16, device=device); bd_oh[0, g] = 1.0
        mu = actor(obs_t, bd_oh)
        best_md = float('inf')
        for trial in range(200):
            z = mu + 0.5 * torch.randn_like(mu)
            z_np = z.squeeze(0).cpu().numpy()
            c, md, bi, _ = evaluate_z(torch.tensor(z_np, device=device).unsqueeze(0))
            if c[0] and bi[0] == g and md[0] < best_md:
                best_md = md[0]
            if best_md < 0.5: break
        if best_md < float('inf'):
            print(f"  goal {g:2d}: md={best_md:.3f}m")
        else:
            print(f"  goal {g:2d}: miss")

print(f"\nDone. Output: {OUT}/")
