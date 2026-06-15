# 算法选型决策记录

> 日期: 2026-06-14
> 关联: `docs/rl_algorithm_routes.md`, `docs/exp1_prior_perturbation_plan.md`, `docs/bd_design_spec.md`

## 1. 任务定义

**目标**: 在 val subseq 60（NA=2 单场景）通过 RL 找到多个不同的碰撞解（多解）。

**子目标链**:
- 阶段 1.5: 单场景多解（sub60 填 ≥4 BD cell）
- 阶段 2.5: 多场景数据生成
- 阶段 3: 跨场景泛化 policy

## 2. 强约束

1. **必须 Policy 网络** — 铲屎官明确要求，为阶段 3 跨场景泛化预留接口
2. **复用 STRIVE 现有代码** — 不破坏原项目
3. **sub60 物理特性**: NA=2, 6s 时间窗, atk 需 ~6.5m 横移才能撞

## 3. 候选方案与讨论

### 3.1 候选方案

| # | 方案 | 满足 Policy? | 阶段 1.5 成功率 | 实施成本 |
|---|------|:---:|:---:|:---:|
| A | CMA-ES + Archive | ❌ | 80% | ~150 行 |
| B | PGA-MAP-Elites | ✅ | 50% | ~500 行 |
| C | GC-SAC + HER | ✅ | 30% | ~800 行 |
| D | IL + 条件式 Policy | ✅ | 90% | ~400 行 |

### 3.2 淘汰过程

**Step 1: 淘汰 CMA-ES**
- 原因: 无 Policy 网络, 不能直接支持跨场景泛化
- 用途: 可作为阶段 1.5 的数据生成器, 但不直接产 policy

**Step 2: GC-SAC+HER 风险**
- 单步问题（无时序）→ SAC 退化为 V-function
- 16 cell 物理相关性弱 → HER 重标记合理性差
- 冷启动: 初始无碰撞样本 → HER 死锁
- 评估: 风险高

**Step 3: PGA-MAP-Elites 评估**
- 单场景 Critic 难训
- GA crossover 在 32 维 z 空间语义不清
- 但 QD 框架成熟, 与论文思路最对齐
- 评估: 中等风险

**Step 4: IL + 条件式 Policy**
- 阶段 1.5 用任一优化器产 (scene, z*) 数据
- 训条件式 policy: π(scene, bd_onehot) → z
- 训练稳定（监督学习）
- 阶段 3 直接复用
- 评估: 路径最稳

## 4. 最终决策

**先做实验, 再选方案** — 实验已完成 (`docs/exp1_prior_perturbation_results.md`)

### 4.1 实验关键数据 (sub60)
- 碰撞率峰值: 7.8% (σ=5×, 128 sample)
- **实际可填 BD cell = 4** (cell 2, 6, 10, 14, heading 固定 180° 朝后)
- min_dist 范围 0.8-1.7m (梯度基线 0.97m)
- GRPO 在 sub60 信号太稀疏 (K=32 group 内 < 3 个撞到)

### 4.2 排除 GRPO
- 8% 碰撞率 → K=32 group reward variance 极小
- 16 cell 12 个空 → archive 浪费
- 推荐路径: IL + Conditional Policy

### 4.3 暂定优先路径

```
阶段 1.5: Prior Perturbation 实验 (done)
   ↓ 拿 σ 扰动碰撞率/BD 分布数据
阶段 1.5 决策 (当前):
   └─ 选 IL + Conditional Policy (低风险, 适配 4 cell sub60)
   └─ 数据生成器: Gradient+Restart × 16 次 (复用现有 adv_gen_optim.py)
阶段 2.5: Gradient+Restart 在 5-10 个其他 subseq 累计数据
阶段 3:   Conditional Policy 训练 → zero-shot 跨场景泛化
```

## 5. 待办

- [x] 跑 prior perturbation 实验
- [x] 基于实验结果决定具体算法
- [x] v3~v9 GC-SAC+HER 实验 (全失败)
- [x] v10 REINFORCE Bandit 实验 (部分成功: 推理 3/4)
- [x] Gigaflow, Asymmetric SP, DIAYN 论文调研
- [x] 方向修正: 主路线 Asymmetric Self-Play
- [ ] 实施 v11 Asymmetric Self-Play
- [ ] 多场景扩展 (阶段 2.5)
- [ ] 跨场景泛化验证 (阶段 3)

---

> **以下为 2026-06-15 补充 — v3~v10 实验 + 论文调研 + 方向修正**

## 6. GC-SAC+HER 路线实验总结 (v3~v9)

> 详情见: `docs/gc_sac_her_results.md`, `docs/mixture_policy_results.md`, `docs/v9_multistep_sac_results.md`

**结论: 全失败。** 8 个版本 (v3~v9 + 变体) 中, 最佳结果训练时 4/16 cell, 推理时退化到 1/4。

### 6.1 失败根因

| 版本 | 算法 | 失败原因 |
|------|------|---------|
| v1-v2 | SAC | 0/1 reward → Q=nan, pi=-3e16 爆炸 |
| v3 | REINFORCE 0/1 | 信号太稀, 0/16 cell |
| v4 | mean-proximity | 全局 proximity 与 goal cell 不对齐 |
| v5 | KL penalty | KL 锁住 μ, 退化到 prior |
| v6 | endpoint-proximity | 训时探索/推理 greedy 分布不一致 |
| v7-v8 | Mixture REINFORCE | Per-mode baseline 不够, mode 仍 collapse |
| v9 | Mixture SAC+HER | hit 太稀 (0.2%), critic 无信号可学 |

**统一根因**: 0/1 reward + 极稀疏 hit (0.5%) + 共享 backbone → miss 噪声淹没 hit 梯度。

---

## 7. REINFORCE Bandit 路线 (v10)

> 详情见: `docs/v10_reinforce_bandit_results.md`

### 7.1 核心改动 (vs v3~v9)

| 组件 | v3~v9 | v10 |
|------|-------|-----|
| 问题建模 | MDP (SAC/HER/多步) | **Contextual Bandit (单步)** |
| Reward | 0/1 或 proximity | **Continuous (-min_dist)** |
| Critic | 有 (SAC 双 Q + target) | **无 (REINFORCE baseline)** |
| σ | 可训练, 自然收缩 | **固定 0.5** |
| Mode 隔离 | 共享 backbone | **Per-mode 独立更新** |
| 论文支撑 | — | **Williams 1992 + Bishop 1994 + Mnih 2016** |

### 7.2 实验结果

| Cell | 训练 (2000ep) | 推理 (200 sample/goal) |
|------|--------------|------------------------|
| 2 | 1 hit, md=1.818m | **md=1.338m** ✓ |
| 6 | 9 hits, md=0.739m | **md=1.227m** ✓ |
| 10 | 1 hit, md=0.534m | ✗ miss |
| 14 | 0 hit | **md=1.690m** ✓ (hit by luck) |

**avg_R 全程未涨**: ~20 → ~18 → ~21。Policy 的 μ 未被推到位, 推理命中本质仍靠 σ=0.5 的随机探索。

### 7.3 3 重矛盾分析

**矛盾 A (miss 噪声淹没)**:
- 500 miss samples × |advantage|≈0 的 `backward()` 累积 → backbone 被噪声污染
- 1 hit 梯度无法对抗 500 次 miss 的累积

**矛盾 B (高维梯度方向抵消)**:
- hit 样本的 z 在 32D 空间中方向不一致 → 两次 hit 的梯度相互抵消
- REINFORCE 用连续 weighting, CMA-ES 用 binary discard

**矛盾 C (binary gate 更新频率低)**:
- 只保留 hit 样本 → 0.5% 更新频率, 不够训神经网络
- 保留全部样本 → 矛盾 A
- CMA-ES 绕过神经网络 (O(z_dim) 更新) → 没有这个矛盾

---

## 8. 论文调研

### 8.1 对抗轨迹生成领域论文分类

> 调研日期: 2026-06-15 | 目标: 找到"32D latent space + 0.5% hit rate + 多解 + 无碰撞数据" 的方法论

| 类别 | 代表论文 | 核心方法 | 如何解决 0.5% 瓶颈? |
|------|---------|---------|---------------------|
| **Gradient opt** | STRIVE (Rempe et al., CVPR 2022) [[2112.05077](https://arxiv.org/abs/2112.05077)] | ∇_z 优化 | 梯度直达, 不需要 RL |
| **Latent diffusion** | Guided LDM (Peng et al., 2025) [[2505.00515](https://arxiv.org/abs/2505.00515)] | Graph VAE + diffusion + guidance | 用 nuScenes 开放数据 |
| **Flow matching** | Flow-VAE (Gong et al., Waabi, ICRA 2026) [[2605.04366](https://arxiv.org/abs/2605.04366)] | Flow transform z_nominal→z_critical | **不需要, 但有 500+10K 碰撞数据** |
| **RL editing** | Liu et al., 2023 [[2306.14131](https://arxiv.org/abs/2306.14131)] | RL 序列编辑场景 | 用 VAE 约束 realism |
| **Min-max** | ADV-0 (Nie et al., 2026) [[2603.15221](https://arxiv.org/abs/2603.15221)] | Zero-sum game, Nash eq | closed-loop sim |
| **GOAT** | Chaudhary et al., 2025 [[2504.15457](https://arxiv.org/abs/2504.15457)] | Frozen VAE + RL latent search | **不需要, 100% latent coverage** |
| **Asymmetric SP** | Zhang et al., Waabi, 2024 [[2409.18218](https://arxiv.org/abs/2409.18218)] | Teacher-Student scenario generation | 自对弈 curriculum |
| **Gigaflow PPO** | Cusumano-Towner et al., ICML 2025 [[2502.03349](https://arxiv.org/abs/2502.03349)] | PPO + 10^12 transitions + dense reward | Scale + dense reward |

### 8.2 关键发现: 文献中没有直接答案

**每篇论文都避开了"32D latent space + 0.5% hit rate + 无碰撞数据" 这个组合:**

- **Flow-VAE**: 有 500+10K 碰撞场景数据 → 不需要面对 0.5%
- **GOAT**: VAE latent space 覆盖 100% 行为 → 不需要面对 0.5%
- **Gigaflow**: 9 dense reward 分量 + 10^12 transitions → 不需要面对 0.5%
- **Guided LDM**: Graph VAE + diffusion 在 nuScenes 上训 → 不需要面对 0.5%

**没有任何一篇论文做 "RL policy directly outputs 32D latent z to generate adversarial trajectories"。** 我们的 v3-v10 走在地图上没有标注的区域。

### 8.3 GOAT (Chaudhary et al., 2025)

**"Improving Human-AI Coordination through GOAT"** [[2504.15457](https://arxiv.org/abs/2504.15457)]

- **核心**: Frozen VAE + RL adversary 搜索 latent space
- **Adversary πA(z) → z'**: 在 latent 空间搜索让 Cooperator regret 最大的 partner
- **Frozen VAE**: 保证所有生成的 partner 都是 realistic 的
- **Regret = SP return - XP return**: 自然 curriculum

**与我们 v10 的同构性**:
```
GOAT:  πA(z) → z' → frozen VAE → partner → regret
v10:   π(obs, g) → z → frozen decoder → trajectory → collision reward
```
结构完全同构。**但 GOAT 的 VAE latent 空间 100% 覆盖有效行为, 我们没有这个条件。**

**启示**: GOAT 验证了 "RL + frozen generative model + latent search" 框架的有效性, 但前提是生成模型已有足够 coverage。对我们的映射:
- 保持 STRIVE decoder **冻结** → 自动保证 trajectory realism
- RL policy 搜 latent space → 但不能搜全部 32D, 改为搜 **Δz ∈ bounded ball** (缩小探索)
- 不需要额外 realism loss (decoder 冻结 = realism constraint)

### 8.4 Flow-VAE (Gong et al., Waabi, ICRA 2026)

**"Conditional Flow-VAE for Safety-Critical Traffic Scenario Generation"** [[2605.04366](https://arxiv.org/abs/2605.04366)]

- **两阶段**: (1) CVAE 在 mixed data 上训 → (2) Flow model 学 z_nominal → z_critical
- **数据**: 500 real + 10K sim paired scenarios (in-house Waabi fleet)
- **不开源**: 无代码, 无开放数据
- **与 STRIVE 的关系**: 直接引用 STRIVE [15], 使用相同 CVAE 架构

**Flow-VAE 证明了两个重要命题**:
1. Latent space 有方向性: z 空间存在 "safe → dangerous" 的 flow direction
2. Flow matching 可以学这个方向, 不需要 RL 探索

**但前提是成对碰撞数据。** 没有这些数据, flow model 不知道往哪个方向推 z。

### 8.5 QD-RL 方法论调研 (2026-06-15)

> 问题: QD-RL 领域有哪些新方法论? policy 如何设计以支持跨场景泛化和多解?

#### 8.5.1 核心论文

**DCG-MAP-Elites** (Faldor et al., 2023) [[2303.03832](https://arxiv.org/abs/2303.03832)]
- Descriptor-conditioned critic → 在整个 BD 空间上优化, 避免 PGA-ME 的 descriptor collapse
- **Archive distillation into single policy**: actor-critic 训练过程同时蒸馏 archive → policy(bd) 可执行全部行为
- 算法: TD3 + descriptor-conditioned actor + critic
- 相对 PGA-ME 提升 82%

**Policy Diffusion** (Hegde et al., 2023) [[2305.18738](https://arxiv.org/abs/2305.18738)]
- 用 VAE + latent diffusion 压缩 QD archive → 单生成模型
- 压缩比 13x, 98% reward recovery, 89% coverage
- Conditioning: 行为 descriptor 或自然语言

**PPGA** (Batra et al., ICLR 2024 Spotlight) [[2305.13795](https://arxiv.org/abs/2305.13795)]
- PPO 适配 DQD 框架, 4x improvement over baselines
- 第一个 on-policy QD-RL 方法

**AutoQD** (Hedayatian et al., ICLR 2026) [[2506.05634](https://arxiv.org/abs/2506.05634)]
- 自动生成行为描述子 (MMD between occupancy measures), 不需要手工设计
- 开源: github.com/conflictednerd/autoqd-code

**SV-QD-RL** (Zuo et al., 2026) [[2606.08735](https://arxiv.org/abs/2606.08735)]
- Structure-conditioned branches: 每个 candidate 有独立的 structural mask + critic
- Branch-aware QD archive

#### 8.5.2 Policy 架构设计调研

**问题: QD-RL 论文的 policy 如何支持多样性输出? 是否使用 MoG head?**

| 论文 | Policy 架构 | 多样性机制 |
|------|-----------|-----------|
| **DCG-MAP-Elites** | `π(a\|s, bd)` — 单网络, bd 作为输入 | 不同 bd → 不同输出 (deterministic conditioning) |
| **Policy Diffusion** | Diffusion over policy params, conditioned on bd | Conditioning |
| **PGA-MAP-Elites** | `π(a\|s, bd)` — 单网络 | 不同 bd → 不同输出 |
| **PPGA** | `π(a\|s, bd)` — PPO policy | 不同 bd → 不同输出 |
| **SV-QD-RL** | 多 branch, 各带 structural mask | Branch 间结构隔离 |
| **我们的 v10** | Mixture K=4 MoG heads | Sample mode k (stochastic) |
| **我们的 PGA-ME** | `π(a\|s, bd_onehot)` — ConditionalGaussianPolicy | 不同 bd_onehot → 不同 μ,σ |

**关键发现**:
1. **没有任何 QD-RL 论文使用 MoG head** — 全部用 bd-descriptor conditioning
2. 多样性来自**输入 space** (descriptor), 不是**输出 space** (sampling)
3. 我们 PGA-ME 里已有的 `ConditionalGaussianPolicy(obs, bd_onehot) → (μ, σ)` 和 QD-RL 标准范式一致
4. v10 的 MoG 设计偏离了 QD-RL 的主流方法论

#### 8.5.3 对我们路线的潜在影响

**独立路线 K (QD Distillation)**: 与 Flow-VAE + Asymmetric SP 并行考虑

```
PGA-ME (已有) + DCG 扩展:
  → 多 scene 上训 better QD archive (DCG's descriptor-conditioned critic)
  → 蒸馏 archive → π(obs, bd_cell_g) → z
  → 新 scene 推理: π(obs_new, g=2) → z → 攻击轨迹

与 Flow-VAE + Asymmetric SP 的关系:
  共同点: 都需要 per-scene 的数据 (z solutions)
  区别: QD 路线用 archive 蒸馏, Flow-VAE 路线用 flow matching
  可组合: QD generate data → Flow model learn distribution
```

---

## 9. 方向修正 (2026-06-15, 更新于同日深夜)

### 9.1 废除的路线

| 路线 | 原因 |
|------|------|
| GC-SAC+HER (v3~v9) | MDP 工具用于 Bandit, 本质错误 |
| REINFORCE Bandit (v10) | 3 重矛盾在单场景下无解 |
| Gigaflow 级别纯 RL | Scale gap 8 数量级, 不可逾越 |
| IL + Conditional Policy | 铲屎官禁止 IL |
| Pure RL in full 32D z-space | 探索空间太大, 0.5% hit rate 是硬上限 |
| GOAT direct mapping | GOAT 的 latent 空间 100% 覆盖, 我们只有 ~8% |

### 9.2 当前主路线: Flow-VAE + Asymmetric SP 组合

> 核心 idea: **Asymmetric SP 生成训练数据 → Flow Model 学分布变换**

```
Phase 1 (Asymmetric SP):
  Teacher (ego): 学习躲避 (约束: 不能离 GT 太远)
  Student (attacker, K=4): z_atk = z_prior + Δz, ||Δz|| ≤ 2.0
  每次 attacker 碰撞成功 → 记录 (z_prior, cell_g, z_collision)
  → 自动生成训练数据对

Phase 2 (Flow Model):
  输入: (z_prior, cell_g)
  输出: z ≈ z_collision_g (学 Δz distribution)
  方法: Rectified Flow Matching in 32D z-space

Phase 3 (Inference):
  新 scene: z_prior_new + g=2 → Flow → z_collision_2 → decoder → 攻击轨迹
```

### 9.3 为什么这个组合可能 work

1. **Flow-VAE 证明了 z 空间有方向性** → flow model 可以学这个方向
2. **Asymmetric SP 生成训练数据** → 不需要预先收集碰撞数据
3. **Δz bounded** (||Δz|| ≤ 2.0) → 探索空间从 R^32 缩小到 32D 球 → 碰撞率可能提升 10-50x
4. **Decoder 冻结** (GOAT 验证的有效模式) → 自动 realism constraint
5. **Flow 学的是分布** (不是 point)→ 天然 diverse, 支持多解

### 9.4 待验证的风险

| 风险 | 问题 | 缓解 |
|------|------|------|
| **Teacher-Student 碰撞窗口短** | ego 可能 100ep 学会躲, 碰撞窗口太短 | Realism constraint 限制 ego 躲避能力 |
| **Flow 需要多少数据对** | 32D Rectified Flow 的最少样本量未知 | 预计 ≥50 对 per cell |
| **Asymmetric SP 能否持续产出** | 如果 collision 率太低, 无法积累数据 | 多场景并行 + batch decode 提速 |
| **跨场景泛化** | Flow model 能否泛化到新 scene | Flow 在 z_prior 空间学变换, 新 scene 推理需验证 |

### 9.5 路线对比总览

| 路线 | 纯RL | 多解 | 单场景可行 | 无碰撞数据 | 论文支撑 |
|------|:---:|:---:|:---:|:---:|:---:|
| **Flow-VAE + Asymmetric SP** | ✅ | ✅ | 高(预期) | ✅ | Waabi 2026 + Waabi 2024 + GOAT 2025 |
| v10 REINFORCE Bandit | ✅ | ⚠️ | 中(3/4) | ✅ | Williams 1992 |
| Pure Flow-VAE | n/a | ✅ | ✅ | ❌(需碰撞数据) | Waabi 2026 |
| GOAT direct | ✅ | ✅ | 低 | ✅ | Chaudhary 2025 |
| Guided LDM | n/a | ✅ | ✅ | ⚠️(需nuScenes) | Peng 2025 |

---

## 参考文献

### 对抗轨迹生成

1. **STRIVE**: David Rempe, Jonah Philion, Leonidas J. Guibas, Sanja Fidler, Or Litany. "Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior." CVPR 2022.
   - arXiv: [2112.05077](https://arxiv.org/abs/2112.05077)

2. **Flow-VAE**: Zimu Gong, Brian Zhaoning Zhang, Chris Zhang, Kelvin Wong, Raquel Urtasun. "Conditional Flow-VAE for Safety-Critical Traffic Scenario Generation." ICRA 2026.
   - arXiv: [2605.04366](https://arxiv.org/abs/2605.04366)

3. **Guided LDM**: Mingxing Peng, Ruoyu Yao, Xusen Guo, Yuting Xie, Xianda Chen, Jun Ma. "Safety-Critical Traffic Simulation with Guided Latent Diffusion Model." 2025.
   - arXiv: [2505.00515](https://arxiv.org/abs/2505.00515)

4. **RL-based Editing**: Haolan Liu, Liangjun Zhang, Siva Kumar Sastry Hari, Jishen Zhao. "Safety-Critical Scenario Generation via Reinforcement Learning Based Editing." 2023.
   - arXiv: [2306.14131](https://arxiv.org/abs/2306.14131)

5. **ADV-0**: Tong Nie, Yihong Tang, Junlin He, Yuewen Mei, Jie Sun, Lijun Sun, Wei Ma, Jian Sun. "ADV-0: Closed-Loop Min-Max Adversarial Training for Long-Tail Robustness in Autonomous Driving." 2026.
   - arXiv: [2603.15221](https://arxiv.org/abs/2603.15221)

6. **STRELGen**: Lorenzo Bonin, Francesco Giacomarra, Luca Bortolussi, Jyotirmoy V. Deshmukh, Francesca Cairoli. "Guiding Neuro-Symbolic Scenario Generation with Spatio-Temporal Logic." 2026.
   - arXiv: [2605.19038](https://arxiv.org/abs/2605.19038)

7. **KG-ASG**: Cheng Wang, Chen Xiong, Ziwen Wang, Yuchen Zhou, Qiang Liu. "KG-ASG: Collision-Knowledge-Guided Closed-Loop Adversarial Scenario Generation With Primary-Support Attribution." 2026.
   - arXiv: [2605.18895](https://arxiv.org/abs/2605.18895)

8. **CRAG**: Qiujing Lu, Xuanhan Wang, Runze Yuan, Wei Lu, Xinyi Gong, Shuo Feng. "Controllable Risk Scenario Generation from Human Crash Data for Autonomous Vehicle Testing." 2025.
   - arXiv: [2512.07874](https://arxiv.org/abs/2512.07874)

9. **Dynasto**: Dmytro Humeniuk, Mohammad Hamdaqa, Houssem Ben Braiek, Amel Bennaceur, Foutse Khomh. "Dynasto: Validity-Aware Dynamic-Static Parameter Optimization for Autonomous Driving Testing." 2026.
   - arXiv: [2603.21427](https://arxiv.org/abs/2603.21427)

10. **Scenario Dreamer**: Luke Rowe, Roger Girgis, Anthony Gosselin, Liam Paull, Christopher Pal, Felix Heide. "Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments." CVPR 2025.
    - arXiv: [2503.22496](https://arxiv.org/abs/2503.22496)

### Self-Play / Multi-Agent / RL

11. **Asymmetric Self-Play**: Chris Zhang, Sourav Biswas, Kelvin Wong, Kion Fallah, Lunjun Zhang, Dian Chen, Sergio Casas, Raquel Urtasun. "Learning to Drive via Asymmetric Self-Play." Waabi 2024.
    - arXiv: [2409.18218](https://arxiv.org/abs/2409.18218)

12. **Gigaflow**: Marco Cusumano-Towner, David Hafner, Alex Hertzberg, Brody Huval, Aleksei Petrenko, Eugene Vinitsky, Erik Wijmans, Taylor Killian, Stuart Bowers, Ozan Sener, Philipp Krähenbühl, Vladlen Koltun. "Robust Autonomy Emerges from Self-Play." ICML 2025.
    - arXiv: [2502.03349](https://arxiv.org/abs/2502.03349)

13. **GOAT**: Paresh Chaudhary, Yancheng Liang, Daphne Chen, Simon S. Du, Natasha Jaques. "Improving Human-AI Coordination through Online Adversarial Training and Generative Models." 2025.
    - arXiv: [2504.15457](https://arxiv.org/abs/2504.15457)

14. **DIAYN**: Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, Sergey Levine. "Diversity is All You Need: Learning Skills without a Reward Function." ICLR 2019.
    - arXiv: [1802.06070](https://arxiv.org/abs/1802.06070)

15. **HAP**: Manjie Xu, Xinyi Yang, Jiayu Zhan, Wei Liang, Chi Zhang, Yixin Zhu. "Heterogeneous Adversarial Play in Interactive Environments." NeurIPS 2025.
    - arXiv: [2510.18407](https://arxiv.org/abs/2510.18407)

### 驾驶仿真 / 数据增强

16. **SimScale**: Haochen Tian, Tianyu Li, Haochen Liu, Jiazhi Yang, et al. "SimScale: Learning to Drive via Real-World Simulation at Scale." CVPR 2026 Oral.
    - arXiv: [2511.23369](https://arxiv.org/abs/2511.23369)

17. **SceneDiffuser++**: Shuhan Tan, John Lambert, Hong Jeon, et al. "SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model." CVPR 2025.
    - arXiv: [2506.21976](https://arxiv.org/abs/2506.21976)

18. **R2SE**: Haochen Liu, Tianyu Li, Haohan Yang, et al. "Reinforced Refinement with Self-Aware Expansion for End-to-End Autonomous Driving." 2025.
    - arXiv: [2506.09800](https://arxiv.org/abs/2506.09800)

### QD-RL (Quality Diversity Reinforcement Learning)

24. **DCG-MAP-Elites**: Maxence Faldor, Félix Chalumeau, Manon Flageat, Antoine Cully. "MAP-Elites with Descriptor-Conditioned Gradients and Archive Distillation into a Single Policy." 2023.
    - arXiv: [2303.03832](https://arxiv.org/abs/2303.03832)

25. **Policy Diffusion**: Shashank Hegde, Sumeet Batra, K. R. Zentner, Gaurav S. Sukhatme. "Generating Behaviorally Diverse Policies with Latent Diffusion Models." 2023.
    - arXiv: [2305.18738](https://arxiv.org/abs/2305.18738)

26. **PPGA**: Sumeet Batra, Bryon Tjanaka, Matthew C. Fontaine, Aleksei Petrenko, Stefanos Nikolaidis, Gaurav Sukhatme. "Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning." ICLR 2024 Spotlight.
    - arXiv: [2305.13795](https://arxiv.org/abs/2305.13795)

27. **AutoQD**: Saeed Hedayatian, Stefanos Nikolaidis. "AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization." ICLR 2026.
    - arXiv: [2506.05634](https://arxiv.org/abs/2506.05634)

28. **SV-QD-RL**: Lianrong Zuo, Peilan Xu, Yong Liu, Wenjian Luo. "Structure-Conditioned Actor-Critic Branches for Quality-Diversity Reinforcement Learning." 2026.
    - arXiv: [2606.08735](https://arxiv.org/abs/2606.08735)

### RL 基础方法

29. **REINFORCE**: Ronald J. Williams. "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." Machine Learning, 1992.

30. **MDN**: Christopher M. Bishop. "Mixture Density Networks." 1994.

31. **A3C**: Volodymyr Mnih et al. "Asynchronous Methods for Deep Reinforcement Learning." ICML 2016.
    - arXiv: [1602.01783](https://arxiv.org/abs/1602.01783)

32. **Sutton & Barto**: "Reinforcement Learning: An Introduction." 1998.

33. **PGA-ME / CMA-ME**: Matthew C. Fontaine, Stefanos Nikolaidis. "Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space." GECCO 2020.
    - arXiv: [1912.02400](https://arxiv.org/abs/1912.02400)
