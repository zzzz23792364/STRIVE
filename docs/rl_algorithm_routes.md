# RL 算法选型路线讨论记录

## 1. 需求回顾

| 需求 | 对应 Phase | 说明 |
|------|-----------|------|
| 单场景单解 | Phase 1 | 替代梯度基线，训练 policy 找到 1 个碰撞解 |
| 单场景多解 | Phase 2 | 对同一场景输出 ≥K 个不同的碰撞解 |
| 多场景单解 | Phase 3 | 跨场景泛化，zero-shot 对新场景输出解 |
| 多场景多解 | Phase 4 | Phase 2 + Phase 3 组合 |

碰撞模式按 **碰撞位置 + 碰撞角度** 粗分为 16 类（4×4 网格）。

## 2. 候选算法路线

### A. GC-SAC + HER

**Core Idea**: Goal-conditioned RL + Hindsight Experience Replay

```
定义 16 个 goal（每种碰撞模式一个 one-hot）
Policy: π(z | obs, g)
Reward: goal 达成奖励
HER: 实际状态 → 重标记为新 goal

训练后推理：遍历 16 个 goal → 得到 16 个解
```

| 维度 | 评估 |
|------|------|
| 多解能力 | ⚠️ 被动。只在 rollout 自然出现的 goal 上有效 |
| 稀有模式 | ❌ 从未出现的模式，HER 无素材可重标记 |
| 样本效率 | ✅ 高（HER 重标记提升密度） |
| 实现成本 | 中等（需要 goal 嵌入 + SAC + HER） |
| 独立 Phase 2 可行性 | ❌ 需要额外探索机制 (curiosity/curriculum) |

**结论**: 不适合独立做 Phase 2，但适合作为 QD 框架内部的 PG backbone（样本效率高）。

### B. PGA-MAP-Elites (QD-RL)

**Core Idea**: Archive + GA variation + PG variation

```
Archive: 16 cells (BD = 位置 bin × 角度 bin)
GA variation:  两父代参数交叉 + 自适应 noise → 填充空 cell
PG variation:  Critic 梯度驱动 → 精化已有 cell
Replay buffer: 存所有个体 transition → 训练 Critic
```

| 维度 | 评估 |
|------|------|
| 多解能力 | ✅ **原生支持**。archive 天然存储多解 |
| 稀有模式 | ✅ GA variation 的 crossover + noise 能跳跃到未填充区域 |
| 样本效率 | 中等（比纯 off-policy 差，但有 replay buffer） |
| 实现成本 | 较高（archive + 双 variation + critic） |
| 独立 Phase 2 可行性 | ✅ 最直接匹配 |

**结论**: Phase 2/4 的最佳选型方向。

### C. PPO + Diversity Bonus (轻量替代)

**Core Idea**: 在 PPO 的 reward 中加一个 BD-novelty 项

```
Reward = -AdvGenLoss + w · BD_novelty
BD_novelty = 当前轨迹 BD 与 archive 中最近 cell 的距离
```

| 维度 | 评估 |
|------|------|
| 多解能力 | ⚠️ 可用，但不如 QD-RL 系统化 |
| 稀有模式 | ⚠️ 依赖 w 调参，噪声探索随机 |
| 样本效率 | 低（on-policy） |
| 实现成本 | ✅ 低（复用 PPO 实现） |
| 独立 Phase 2 可行性 | ✅ 可作为 MVP |

**结论**: Phase 1→2 的最快过渡方案。

## 3. 算法对比总表

| 维度 | PPO baseline | PPO + DB | GC-SAC + HER | PGA-MAP-Elites |
|------|:----------:|:--------:|:-----------:|:--------------:|
| Phase 1 单解 | ✅ | ✅ | ✅ | ✅ |
| Phase 2 多解 | ❌ | ⚠️ | ❌ | ✅ |
| 稀有模式覆盖 | ❌ | ⚠️ | ❌ | ✅ |
| 样本效率 | 低 | 低 | 高 | 中 |
| 实现成本 | 低 | 低 | 中 | 高 |
| 与现有代码集成 | ✅ PPO已有 | ✅ 加一行 | 新算法 | 新框架 |

## 4. 路线建议

```
推荐渐进路线：

Phase 1 (当前) ─→ Phase 2 MVP ─→ Phase 2 正式 ─→ Phase 3/4
    PPO              PPO + DB       PGA-MAP-Elites   条件式 QD
                                                      (场景ID + BD)

               ↓                 ↓                 ↓
           快速验证"多样性     核心多解方案      跨场景泛化
           reward 有效"       archive + 双优化器   + 多解
```

### 为什么要渐进

1. **PPO + DB**：在 PPO 基线基础上加几行代码就能验证多样性信号是否有用。如果连加了 novelty 都找不到新解，说明问题不在算法而在 reward 或 BD 设计。
2. **PGA-MAP-Elites**：确认多样性路径可行后再投入实现，避免造大轮子却发现根本走不通。
3. **GC-SAC+HER 作为 QD 内的 PG backbone**：如果样本效率成为瓶颈，把 QD 内部的 PG 部分换成 SAC+HER。

## 5. 关键待定问题

- BD 设计：16 类 (4×4) 够吗？还是需要连续 BD 空间？
- Reward 中多样性权重的尺度？与 -AdvGenLoss 如何平衡？
- GA noise 的初始幅值和自适应策略？
- 是否需要先 warmup policy 到能稳定产出碰撞（Phase 1 成果）再启动 archive？

> 讨论日期: 2026-06-14 (初始) / 2026-06-15 (v10 + 论文调研更新)
> 关联文档: `docs/rl_approach_plan.md`, `docs/phase0_rl_impl.md`, `docs/algorithm_decisions.md`

---

## 6. 实验总结 (2026-06-15)

### 6.1 Route A (GC-SAC+HER) — 已放弃

v3~v9 共 8 个版本全失败。根因: MDP 工具 (SAC/HER/多步) 用于 Bandit 问题 + 0/1 reward.
详见 `docs/gc_sac_her_results.md`, `docs/mixture_policy_results.md`.

### 6.2 Route E (REINFORCE Bandit) — 打底方案, 已跑 (v10)

改用 Contextual Bandit 模型: REINFORCE + continuous reward + fixed σ.
结果: 训时 4/4, 推理 3/4 cell. avg_R 未改善, μ 未被推到位.
详见 `docs/v10_reinforce_bandit_results.md`.

### 6.3 Route F (Asymmetric Self-Play) — 当前主路线 (v11)

基于 Waabi 2025 "Learning to Drive via Asymmetric Self-Play":
Teacher (ego) vs Student (attacker K modes) 对抗学习.
Teacher 生成 Student's failure scenarios → 自动 curriculum.
详见 `docs/v11_asymmetric_selfplay_plan.md`.

### 6.4 Route G (Gigaflow PPO) — 已放弃

Scale gap 8 个数量级 (10^4 vs 10^12 transitions), 不可逾越.
详见 `docs/algorithm_decisions.md` §8.1.

### 6.5 Route H (DIAYN) — 备用路线

Mutual info I(s; z_skill) 替代 external reward.
需要 >20000 ep + batch decode. Skill ↔ cell 对应 post-hoc 不稳定.
详见 `docs/algorithm_decisions.md` §8.3.

## 7. 路线选择矩阵

| # | 路线 | 纯RL | 多解 | 单场景可行 | 跨场景泛化 | 论文支撑 | 状态 |
|---|------|:---:|:---:|:---:|:---:|------|------|
| A | GC-SAC+HER | ✅ | ✅ | ❌ (v3-v9全垮) | 未试 | HER 2017 | 已放弃 |
| B | PGA-MAP-Elites | ✅ | ✅ | ✅ (4/4 0.01m) | ❌ (死记硬背) | CMA-ME 2020 | 存档保留 |
| C | PPO + Diversity Bonus | ✅ | ✅ | 未实现 | 未试 | PPO 2017 | 未实现 |
| D | IL + Conditional Policy | ❌ | ✅ | ✅ | ✅ | — | 铲屎官禁止 |
| E | REINFORCE Bandit | ✅ | ⚠️ | ⚠️ (3/4 infer) | 未试 | Williams 1992 | 打底验证 |
| F | Asymmetric Self-Play standalone | ✅ | ✅ | 中 (collision窗口短) | 可扩展 | Waabi 2025 | 已重新评估 |
| G | Gigaflow PPO | ✅ | ✅ | ❌ (scale gap) | ✅ | Gigaflow 2025 | 已放弃 |
| H | DIAYN | ✅ | ✅ | 中 (需scale) | ✅ | Eysenbach 2018 | 备用 |
| **I** | **Flow-VAE + Asymmetric SP** | ✅ | ✅ | **高 (预期)** | **可扩展** | **Waabi 2026 + Waabi 2024 + GOAT 2025** | **主路线** |
| J | Guided LDM | n/a | ✅ | ✅ | ⚠️ | Peng 2025 | 待评估 |
| K | GOAT-style Δz | ✅ | ✅ | 低 (latent coverage low) | ❌ | Chaudhary 2025 | 已放弃 |

---

## 8. 对抗轨迹生成领域论文分类 (2026-06-15 调研)

### 按方法分类

| 类别 | 论文 | 解决 0.5% 瓶颈? | 适合我们? |
|------|------|----------------|----------|
| **Gradient optimization** | STRIVE (Rempe 2022, CVPR) | ✅ (梯度直达) | ❌ (per-scene, 非 policy) |
| **Latent diffusion** | Guided LDM (Peng 2025) | ⚠️ (nuScenes 数据) | ⚠️ |
| **Flow matching** | Flow-VAE (Waabi ICRA 2026) | ❌ (需碰撞数据对) | ⚠️ (结构可用, 数据不行) |
| **RL editing** | Liu et al. 2023 | ⚠️ (VAE constraint) | ⚠️ |
| **Min-max game** | ADV-0 (Nie 2026) | ❌ (closed-loop) | ❌ |
| **Frozen VAE + RL** | GOAT (Chaudhary 2025) | ❌ (100% coverage) | ⚠️ (框架对, 前提不对) |
| **Asymmetric SP** | Waabi 2024 | ⚠️ (curriculum) | ✅ (数据生成) |
| **PPO at scale** | Gigaflow (ICML 2025) | ❌ (scale) | ❌ (scale gap) |

### 关键发现

**"RL policy directly outputs 32D latent z to generate adversarial trajectories" — 文献空白。**

所有现有论文要么:
1. 不需要面对 0.5% 瓶颈 (GOAT, Gigaflow, Flow-VAE)
2. 不是 RL 方法 (STRIVE, Guided LDM)
3. 需要特定数据 (Flow-VAE, Guided LDM)

**我们的 v11 = Flow-VAE + Asymmetric SP 是原创组合**, 用 Asymmetric SP 生成 (z_prior, z_collision) 数据对, 用 Flow-VAE 的结构学分布变换。这个组合在已有论文中没有先例, 但组成部分各自有论文支撑。
