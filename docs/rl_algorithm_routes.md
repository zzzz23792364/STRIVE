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
| L | QD Distillation (DCG-ME) | ✅ | ✅ | ✅ (archive→policy) | ✅ (descriptor-conditioned) | Faldor 2023 + Hegde 2023 | 待调研 |

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

### 附加发现: QD-RL Policy 架构调研

**问题: QD-RL 论文的 policy 如何支持多样性输出?**

| 论文 | Policy 架构 | 多样性机制 |
|------|-----------|-----------|
| DCG-MAP-Elites | `π(a\|s, bd)` — 单网络, bd descriptor 作输入 | conditioning, 不是 sampling |
| Policy Diffusion | Diffusion over policy params, cond. on bd | conditioning |
| PGA-MAP-Elites | `π(a\|s, bd)` — 单网络 | conditioning |
| PPGA | `π(a\|s, bd)` — PPO policy | conditioning |
| SV-QD-RL (2026) | 多 branch, structural mask | branch 隔离 |
| 我们的 v10 | Mixture K=4 MoG heads | stochastic sampling |
| 我们的 PGA-ME | `π(a\|s, bd_onehot)` — ConditionalGaussianPolicy | conditioning (与 QD-RL 一致) |

**没有任何 QD-RL 论文使用 MoG head。** 多样性来自 descriptor conditioning (输入空间), 不是 sampling (输出空间)。

---

## 9. 参考文献 (含 arXiv 链接)

### 对抗轨迹生成

| # | 标题 | 作者 | 会议/年份 | arXiv |
|---|------|------|----------|-------|
| 1 | Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior (STRIVE) | Rempe et al. | CVPR 2022 | [2112.05077](https://arxiv.org/abs/2112.05077) |
| 2 | Conditional Flow-VAE for Safety-Critical Traffic Scenario Generation | Gong et al. (Waabi) | ICRA 2026 | [2605.04366](https://arxiv.org/abs/2605.04366) |
| 3 | Safety-Critical Traffic Simulation with Guided Latent Diffusion Model | Peng et al. | 2025 | [2505.00515](https://arxiv.org/abs/2505.00515) |
| 4 | Safety-Critical Scenario Generation via Reinforcement Learning Based Editing | Liu et al. | 2023 | [2306.14131](https://arxiv.org/abs/2306.14131) |
| 5 | ADV-0: Closed-Loop Min-Max Adversarial Training for Long-Tail Robustness | Nie et al. | 2026 | [2603.15221](https://arxiv.org/abs/2603.15221) |
| 6 | STRELGen: Guiding Neuro-Symbolic Scenario Generation with Spatio-Temporal Logic | Bonin et al. | 2026 | [2605.19038](https://arxiv.org/abs/2605.19038) |
| 7 | KG-ASG: Collision-Knowledge-Guided Closed-Loop Adversarial Scenario Generation | Wang et al. | 2026 | [2605.18895](https://arxiv.org/abs/2605.18895) |
| 8 | Controllable Risk Scenario Generation from Human Crash Data (CRAG) | Lu et al. | 2025 | [2512.07874](https://arxiv.org/abs/2512.07874) |
| 9 | Dynasto: Validity-Aware Dynamic-Static Parameter Optimization for AD Testing | Humeniuk et al. | 2026 | [2603.21427](https://arxiv.org/abs/2603.21427) |
| 10 | Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation | Rowe et al. | CVPR 2025 | [2503.22496](https://arxiv.org/abs/2503.22496) |

### Self-Play / Multi-Agent / RL

| # | 标题 | 作者 | 会议/年份 | arXiv |
|---|------|------|----------|-------|
| 11 | Learning to Drive via Asymmetric Self-Play | Zhang et al. (Waabi) | 2024 | [2409.18218](https://arxiv.org/abs/2409.18218) |
| 12 | Robust Autonomy Emerges from Self-Play (Gigaflow) | Cusumano-Towner et al. | ICML 2025 | [2502.03349](https://arxiv.org/abs/2502.03349) |
| 13 | Improving Human-AI Coordination through Online Adversarial Training (GOAT) | Chaudhary et al. | 2025 | [2504.15457](https://arxiv.org/abs/2504.15457) |
| 14 | Diversity is All You Need: Learning Skills without a Reward Function (DIAYN) | Eysenbach et al. | ICLR 2019 | [1802.06070](https://arxiv.org/abs/1802.06070) |
| 15 | Heterogeneous Adversarial Play in Interactive Environments (HAP) | Xu et al. | NeurIPS 2025 | [2510.18407](https://arxiv.org/abs/2510.18407) |

### 驾驶仿真 / 数据增强

| # | 标题 | 作者 | 会议/年份 | arXiv |
|---|------|------|----------|-------|
| 16 | SimScale: Learning to Drive via Real-World Simulation at Scale | Tian et al. | CVPR 2026 Oral | [2511.23369](https://arxiv.org/abs/2511.23369) |
| 17 | SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model | Tan et al. | CVPR 2025 | [2506.21976](https://arxiv.org/abs/2506.21976) |
| 18 | Reinforced Refinement with Self-Aware Expansion (R2SE) | Liu et al. | 2025 | [2506.09800](https://arxiv.org/abs/2506.09800) |

### RL 基础方法

| # | 标题 | 作者 | 会议/年份 | 链接 |
|---|------|------|----------|------|
| 19 | Simple Statistical Gradient-Following Algorithms for Connectionist RL (REINFORCE) | Williams | 1992 | — |
| 20 | Mixture Density Networks (MDN) | Bishop | 1994 | — |
| 21 | Asynchronous Methods for Deep Reinforcement Learning (A3C) | Mnih et al. | ICML 2016 | [1602.01783](https://arxiv.org/abs/1602.01783) |
| 22 | Reinforcement Learning: An Introduction | Sutton & Barto | 1998 | — |
| 23 | Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space (CMA-ME) | Fontaine et al. | GECCO 2020 | [1912.02400](https://arxiv.org/abs/1912.02400) |

### QD-RL (Quality Diversity + RL)

| # | 标题 | 作者 | 会议/年份 | arXiv |
|---|------|------|----------|-------|
| 24 | MAP-Elites with Descriptor-Conditioned Gradients and Archive Distillation into a Single Policy (DCG-ME) | Faldor et al. | 2023 | [2303.03832](https://arxiv.org/abs/2303.03832) |
| 25 | Generating Behaviorally Diverse Policies with Latent Diffusion Models | Hegde et al. | 2023 | [2305.18738](https://arxiv.org/abs/2305.18738) |
| 26 | Proximal Policy Gradient Arborescence for QD-RL (PPGA) | Batra et al. | ICLR 2024 Spotlight | [2305.13795](https://arxiv.org/abs/2305.13795) |
| 27 | AutoQD: Automatic Discovery of Diverse Behaviors with QD Optimization | Hedayatian et al. | ICLR 2026 | [2506.05634](https://arxiv.org/abs/2506.05634) |
| 28 | Structure-Conditioned Actor-Critic Branches for QD-RL (SV-QD-RL) | Zuo et al. | 2026 | [2606.08735](https://arxiv.org/abs/2606.08735) |
