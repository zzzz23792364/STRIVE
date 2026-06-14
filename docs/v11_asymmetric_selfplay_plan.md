# v11: Asymmetric Self-Play 方案

> 日期: 2026-06-15
> 状态: **计划中** — 待铲屎官审批
> 论文支撑: Zhang et al. 2025 (Waabi) "Learning to Drive via Asymmetric Self-Play"
> 关联: `docs/algorithm_decisions.md`, `docs/v10_reinforce_bandit_results.md`

## 1. 为什么 v10 失败后需要 v11?

v10 的核心问题: **0.5% 固定碰撞率 = μ 推不动**。

```
v10: ego GT固定 → 碰撞率永远是 0.5%
      → per mode ~2.5 hits → 32D gradient 方向抵消 → μ 原地踏步
```

**v11 的基本思路**: 让 ego 不再固定, 而是和 attacker 共同学习。这就是 **Self-Play**:

```
v11: ego learns to dodge, attacker learns to collide
      → ego 初始弱 (容易撞) → collision rate ~5-10% → many hits
      → attacker 变强 → ego 被迫变强 → harder scenarios
      → 自动 curriculum → mode 分化
```

**这不是 multi-agent RL, 是 Self-Play** — 两个 agent 共享同一个 policy 网络, 通过 role conditioning 区分。

## 2. 理论支撑

### 2.1 Asymmetric Self-Play (Zhang et al., Waabi 2025)

论文核心机制:

```
Teacher (场景生成者):
  目标 1: 让 Student 失败 (Student collisions ↑)
  目标 2: Teacher 自己能 by pass (solvability constraint)
  目标 3: 场景 keep realistic (realism regularizer)

Student (驾驶策略):
  目标 1: 在 Teacher 场景中存活 (no collisions)
  目标 2: 行为 realistic
  目标 3: 达成 navigation goals

训练时: Teacher-Student 轮流更新
均衡时: Student is α-β-optimal (Theorem 1)
```

**关键 insight**: Teacher 找到 **Student 能力边界** 的场景 → 不是 impossible, 也不是 trivial。这会自动形成 curriculum。

### 2.2 映射到我们的问题

| Waabi Paper | 我们的 v11 |
|-------------|-----------|
| Teacher 生成场景 | ego 学习"让 attacker 失败"的轨迹 |
| Student 驾驶策略 | attacker K modes 学习 "撞 cell g" |
| Teacher pass constraint | ego 保持 near GT trajectory |
| Realism regularizer | ego 不偏离 GT 太远 |
| Student survival | attacker 避开错误 cell |
| Navigation goals | attacker 达成 cell g |

**区别**: Waabi 的 Teacher generates scenarios (adversarial), 我们的 Teacher is ego (co-player in the same scene)。但数学模型完全同构。

### 2.3 和 AlphaGo 的类比

| AlphaGo | v11 Asymmetric Self-Play |
|---------|--------------------------|
| 两个对弈 AI 互相学习 | ego vs attacker 共同进化 |
| 对弈中对手变强 → 发现新棋 | ego 变强 → attacker 被迫发现新碰撞路径 |
| SL 提供 initial policy | prior (N(0, I)) 提供 initial behavior |
| RL 做 improvement + discovery | RL 做 improvement + discovery |

**和我们之前 AlphaGo 讨论的一致性**: v11 的 RL 不是 pure search — 它从 prior 的 easy collisions 出发, 然后 Self-Play 的对抗循环让两者共同进化到更精准的 collision 和 avoidance。这和 AlphaGo 从 SL policy 出发, RL 让 policy 超越人类的过程同构。

## 3. 具体设计

### 3.1 Architecture

```
Shared Policy: MixtureGaussianPolicy (unchanged from v10)
  输入: (obs, role_conditioning)
  输出: K=4 (μ, σ, π) for each mode

Role Conditioning:
  role = "ego": goal = "no_collision" → 任意 mode
  role = "attacker_cell_k": goal = cell_k → mode = k
```

### 3.2 Training Loop

```python
for ep in range(N_EPISODES):
    g = sample([2, 6, 10, 14])
    k = g // 4  # attacker mode

    # 1. Sample z for both roles
    z_ego = policy.sample(obs_ego, role="ego")
    z_atk = policy.sample(obs_atk, role="attacker", goal=g)

    # 2. Decode jointly
    z_full = stack([z_ego, z_atk])  # (NA=2, z_dim)
    trajectories = decoder(z_full)

    # 3. Collision check
    collides, bd_actual, min_dist = collision_check(trajectories)

    # 4. Asymmetric Reward
    if collides and bd_actual == g:
        R_atk = +5.0       # attacker 成功撞对 cell
        R_ego = -5.0        # ego 没躲开
    elif collides and bd_actual != g:
        R_atk = -10.0       # 撞错 cell (mode collapse penalty)
        R_ego = -2.0        # 也撞了但不是 attacker 的目标
    else:
        R_atk = -min_dist   # miss — 靠近 ego 越好
        R_ego = +1.0        # ego 成功躲开

    # 5. Realism regularizer (for ego)
    ego_deviation = ||z_ego - z_ego_GT||  # 距 GT 的偏移
    R_ego += -0.1 * ego_deviation

    # 6. Per-mode baseline + REINFORCE
    advantage_atk = R_atk - baseline_atk[g]
    advantage_ego = R_ego - baseline_ego

    loss = -log_prob(z_atk) * advantage_atk - log_prob(z_ego) * advantage_ego

    loss.backward()
    optimizer.step()
```

### 3.3 Key Design Choices

| Component | Design | Rationale |
|-----------|--------|-----------|
| σ | fixed 0.5 (same as v10) | 防止探索消失 |
| Reward for wrong cell collision | -10 (heavy penalty) | 强制 mode 分化 |
| Ego realism regularizer | 0.1 × deviation from GT | 防止 ego 飞到 unrealistic regions |
| Baseline | per-mode (attacker) + per-role (ego) | 信用分配 |
| Entropy bonus | 0.05 (same as v10) | A3C-style, keep σ from shrinking |

### 3.4 Expected Curriculum Evolution

```
Phase 1 (ep 0-300):
  ego from prior: starts near GT, doesn't dodge well
  attacker from prior: starts with random perturbations
  collision rate: 2-5% (ego doesn't dodge)
  → attacker gets many hits → starts learning

Phase 2 (ep 300-800):
  ego starts learning basic dodging
  attacker mode 1 (cell 6, easiest) gets most hits
  collision rate drops to 1-3%
  → other modes (2,10,14) get some signal

Phase 3 (ep 800-2000):
  ego becomes competent at dodging
  attacker must specialize: different cells require different trajectories
  modes begin to differentiate
  → cell 14 (hardest) starts getting pressure
```

### 3.5 Why This Should Solve the 0.5% Bottleneck

```
v10: ego固定 → collision rate = 0.5% (constant)
v11: ego learns + attacker learns
  → Phase 1: ego弱 → collision rate ~5% (10x v10!)
  → Phase 2: ego强 → collision rate ~1-3% (still 3-6x v10!)
  → Phase 3: 均衡 → collision rate may drop but attacker has learned
```

**Initial boost**: ego的随机初始状态为 attacker 提供 easy collisions, 信号密度比 v10 高 10-20x。

**长期信号**: 即使 collision rate 最终 drop, attacker 已经在 Phase 1-2 学到了 sufficient signal。

## 4. Implementation Plan

### Phase 1: Single-Scene Validation (v11a)

- 1 scene (sub60), 2000 ep
- Shared policy, asymmetric reward
- Goal: 训练 4/4 cell, 推理 4/4 cell
- 预期时间: ~15 min (1 GPU)

### Phase 2: Multi-Scene Scale-Up (v11b, if v11a works)

- 5-10 subseqs, 5000 ep per scene
- Shared policy across scenes (obs varies)
- Goal: 验证跨场景泛化潜力

### Phase 3: Generalization Test (v11c, if v11b works)

- New unseen scenes
- Zero-shot inference
- Goal: 验证 backbone 学到了 (obs, g) → z 的通用映射

## 5. Risks & Mitigations

| 风险 | Mitigation |
|------|-----------|
| **Ego exploits asymmetry**: ego drives unrealistically far → never hit | Realism regularizer (||z_ego - z_GT||) |
| **Mode collapse**: all attacker modes converge to cell 6 (easiest) | Heavy penalty (-10) for wrong cell collision |
| **Training instability**: two adversarial agents → gradient oscillation | Joint update (simultaneous, not alternating) |
| **Too slow**: decode 2 agents → 2x compute | Batch decode (256 samples × 2 agents simultaneously) |
| **Ego σ collapse**: ego stops exploring | Entropy bonus + fixed σ for both roles |

## 6. 代码改动量

| 文件 | 改动 | 行数 |
|------|------|------|
| `scripts/train_v11_selfplay.py` | 新建, 基于 v10 修改 | ~250 行 |
| `src/rl/mixture_policy.py` | 无需改动 (已支持 mode selection) | 0 |
| `src/rl/prior_perturbation.py` | 无需改动 | 0 |

总改动: ~250 行新建文件, 0 行修改现有文件。
