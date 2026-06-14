# Phase 0: RL 基础设施实现记录

## 架构概览

```
Scene Graph + Map
       │
       ▼
┌───────────────┐     ┌──────────────────────────┐
│ TrafficModel  │     │  RL Policy (GaussianPolicy)│
│  (embed)      │────→│  π(a|obs)                 │
│  prior,feats  │     │  → μ_z (32), σ_z (32)    │
└───────────────┘     └──────────┬───────────────┘
                                 │ sample z ~ N(μ, σ) via rsample
                                 ▼
┌──────────────────────────────────────────────┐
│ TrafficModel (decode_embedding, frozen)       │
│ decoder(z, map_feat, past_feat) → trajectory  │
└────────────────────────┬─────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────┐
│ RLReward (复用原版 VehCollLoss/EnvCollLoss/   │
│          MotionPriorLoss)                     │
│ R = -[crash_dist + prior_NLL + penalties]     │
└──────────────────────────────────────────────┘
```

## 文件说明

### `src/rl/policy.py` — 策略网络

```python
class GaussianPolicy(nn.Module):
    # MLP: obs_dim → 256 → 256 → action_dim*2 (μ_z + log_σ_z)
    # forward(obs) → mu, std
    # sample(obs) → z, log_prob, mu, std
```

- Obs 构成: `concat([map_feat (64), past_feat (64), prior_mu (32), prior_var (32), lw (2), sem (NC)])` → 196 dim
- Action: z ∈ R³²，输出多元高斯参数
- 初始化: orthogonal init + 末层 weight *= 0.01 防止初始输出爆炸
- log_std clamp [-5, 2] → std ∈ [0.0067, 7.39]
- NaN 输入自动替换为 0

### `src/rl/reward.py` — 奖励函数

封装原版 loss 组件:

| 项 | 权重 (默认) | 说明 |
|------|--------|------|
| `adv_crash` | 2.0 | 攻击者与目标之间的最小距离 |
| `motion_prior` | 1.0 | 潜变量 z 在运动先验下的负对数似然 |
| `coll_veh` | 20.0 | 车辆间碰撞惩罚 |
| `coll_env` | 20.0 | 车辆与环境碰撞惩罚 |

Reward = -[Σ(weight_i * loss_i)]

### `src/rl/reinforce.py` — REINFORCE 算法

- 折扣回报 G_t = Σ γ^k r_{t+k}
- 支持 baseline 减方差
- Gradient clipping: max_norm=1.0
- 单步 episode 时跳过 returns 归一化

### `src/rl/ppo.py` — PPO 算法

- GAE(λ=0.95) 优势估计
- Clipped surrogate objective (ε=0.2)
- Entropy bonus
- 自动跳过 NaN 梯度更新
- ratio clamp [0, 10] 防溢出

### `src/rl/train_phase1.py` — 训练入口

关键参数:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--scene_idx` | 0 | 使用第几个场景 |
| `--rl_algo` | ppo | reinforce / ppo |
| `--num_episodes` | 1000 | 训练 episode 数 |
| `--warmup_steps` | 50 | 初始化阶段模仿后验 z |
| `--lr_rl` | 3e-4 | 学习率 |
| `--compare_baseline` | True | 是否跑梯度基线对比 |

执行流程:
1. 加载模型 checkpoint + nuScenes val 数据集
2. 加载第 `scene_idx` 个场景
3. 运行 `model.embed()` 获取观测
4. Warmup: 通过 MSE 监督让策略输出接近后验 z
5. （可选）运行梯度基线 (300 iters Adam on z)
6. RL 训练循环: policy.sample → decode → reward → update

## 数据流

```python
# 1. Embed scene → get per-agent obs
embed_info = model.embed(scene_graph, map_idx, map_env)
obs = build_obs(embed_info, scene_graph)  # NA x 196

# 2. Policy samples z for non-ego agents
z_non_ego, log_prob, mu, std = policy.sample(obs[~ego_mask])

# 3. Decode full scene
z_full = concat(prior_ego_z, z_non_ego)
future_pred = model.decode_embedding(z_full, embed_info, ...)

# 4. Compute reward
reward = reward_fn(unnormalize(future_pred), unnormalize(gt_traj), z, prior)

# 5. Update policy
reinforce.update([reward], [log_prob])
```

## 测试结果

- 管线跑通: 数据加载 → embed → sample → decode → reward → update 全链路 OK
- REINFORCE 30 episodes: 无崩溃，`policy_loss` 从 0 增长到 -2066（梯度正在起作用）
- 后续优化方向:
  - 增加 episode 数 (500-5000)
  - 调整 reward 权重 (增大 adv_crash, 减小 motion_prior)
  - 尝试多种 learning rate
  - 长训练时切换 PPO

## 使用方式

### 快速测试 (5 episodes)
```bash
cd /root/autodl-tmp/STRIVE
python src/rl/train_phase1.py \
  --config ./configs/phase1_rl.cfg \
  --out ./out/phase1_test \
  --compare_baseline false \
  --rl_algo reinforce \
  --warmup_steps 50 \
  --num_episodes 5
```

### 正式训练 + 梯度基线对比
```bash
python src/rl/train_phase1.py \
  --config ./configs/phase1_rl.cfg \
  --out ./out/phase1_full \
  --rl_algo ppo \
  --num_episodes 1000 \
  --lr_rl 0.0003
```

### 配置说明

配置文件 `configs/phase1_rl.cfg` 复用了 `test_traffic.cfg` 的参数结构，额外添加了 RL 参数。

## 与梯度基线的区别

| 维度 | 梯度优化 (Phase 2) | RL (Phase 0-1) |
|------|-------------------|----------------|
| 优化变量 | z (直接) | policy 参数 θ |
| 每场景 | 独立 300 iter | 共享 policy |
| 推理 | 300 iter | 1× forward |
| 跨场景泛化 | ❌ | ✅ |
| 多解 | ❌ | ✅ |

## 已知问题

1. **PPO 单步不稳定**: 1-step episode 下 GAE 退化，PPO 易产生 NaN。当前已加 NaN 跳过保护，建议先用 REINFORCE
2. **Reward 尺度**: prior_nll 约 30-400，crash_dist 约 0-6，collision 约 0-1，尺度差异大，需要权重调优
3. **探索不足**: 如果 policy std 太小，策略可能困在局部最优
