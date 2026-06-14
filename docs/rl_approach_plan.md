# RL 替换 Phase 2 单场景梯度方案 — 实施计划

## 1. 背景与动机

当前 STRIVE Phase 2 使用 **gradient-based optimization in latent space** 来生成对抗场景：

- 优化变量：攻击者的 CVAE latent `z ∈ R³²`
- 损失函数：`AdvGenLoss` = crash_dist + prior_NLL + vehicle_coll + env_coll
- 优化器：Adam，300 步迭代
- 特点：每场景独立优化，不跨场景共享知识

**用 RL 替代的动机**：
- 跨场景泛化：训练一个 policy，推理时 zero-shot 对新场景输出碰撞解
- 多解能力：一个场景输出多个不同的碰撞解
- 推理速度快：1× decoder forward vs 300× forward+backward

## 2. 分阶段计划

```
Phase 0: 基础设施 ──→ Phase 1: 单场景单解 ──→ Phase 2: 单场景多解
                                                     ↓
Phase 4: 多场景多解 ←───── Phase 3: 多场景单解
```

### Phase 0: 基础设施
- 新建 `src/rl/` 目录
- Policy 网络（MLP: obs(192) → 256 → 256 → 32(μ_z) + 32(log_σ)）
- Reward 封装层（复用 `AdvGenLoss`）
- REINFORCE + PPO 实现

### Phase 1: 单场景单解（当前目标）
- 固定一个场景，验证 RL 能找到碰撞解
- 对比 REINFORCE vs PPO vs 梯度基线
- 成功标准：success rate ≥ 90%

### Phase 2: 单场景多解
- diversity bonus / ensemble / conditioned policy
- 找到 ≥K 个不同的碰撞解

### Phase 3: 多场景泛化单解
- 在多场景上训练通用 policy
- 推理时 zero-shot 对新场景输出解

### Phase 4: 多场景泛化多解
- Phase 2 + Phase 3 的组合

## 3. RL 架构设计

```
Scene Graph + Map
       │
       ▼
┌───────────────┐     ┌──────────────────────┐
│ TrafficModel  │     │  RL Policy (MLP)      │
│  (embed)      │────→│  π(a|obs)             │
│  prior,feats  │     │  → μ_z, σ_z          │
└───────────────┘     └──────────┬───────────┘
                                 │ sample z ~ N(μ, σ)
                                 ▼
┌──────────────────────────────────────────────┐
│ TrafficModel (decode, frozen)                 │
│ decoder(z, map_feat, past_feat) → trajectory  │
└────────────────────────┬─────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────┐
│ Reward (AdvGenLoss + VehCollLoss + ...)       │
│ R = -[crash_dist + prior_NLL + penalties]     │
└──────────────────────────────────────────────┘
```

**Obs**（来自 model.embed()）：
```
obs = concat([map_feat (64), past_feat (64),
              prior_mu (32), prior_var (32),
              attacker_lw (2), attacker_sem (Nclass)])
→ ~200 dim
```

**Action**：z ∈ R³²（连续），Policy 输出多元高斯参数

**Reward**：`-AdvGenLoss(...)` 即负的 crash distance + prior NLL + 各项碰撞 penalty

## 4. 算法对比

| 维度 | REINFORCE | PPO |
|------|-----------|-----|
| 实现复杂度 | ~50 行 | ~150 行 |
| 样本效率 | 低 | 中 |
| 适用阶段 | Phase 1 快速验证 | Phase 1-3 主力 |
| off-policy | ❌ | ❌ |
| 优势 | 极简，快速跑通 | 稳定，clip 防止崩坏 |

建议：Phase 1 同时实现两者对比。

## 5. 硬件评估

### 远程服务器 (region-41.seetacloud.com)

| 硬件 | 规格 | 评估 |
|------|------|------|
| GPU | NVIDIA RTX 3090, 24GB VRAM | ✅ 远超需求 |
| CPU | Intel Xeon Platinum 8255C, 96 核 | ✅ 远超需求 |
| RAM | 375GB | ✅ 远超需求 |
| 磁盘 | 50GB NVMe (`autodl-tmp`) + 5.7TB 网络盘 (`autodl-pub`) | ✅ 充足 |

**结论**：远程服务器 RTX 3090 24GB 比本地 3060 Ti 8GB 更适合跑 RL 实验。

## 6. 数据与环境 (远程服务器)

- \u2713 nuScenes trainval: `data/nuscenes/trainval/` (v1.0-trainval + maps)
- \u2713 模型 checkpoint: `model_ckpt/traffic_model.pth`, `model_ckpt/traffic_model_all_cats.pth`
- \u2713 环境: base conda (torch 1.9.0+cu111, 所有依赖已安装)
- \u2713 推理验证通过: `test_traffic.py` 成功跑通
- \u274c 预生成场景: `data/strive_scenarios/` 暂未使用

> 详细部署记录见 `docs/remote_deployment.md`

## 7. 下一步

- [x] 讨论并记录方案
- [x] 远程环境部署完成
- [x] 数据 + 模型 checkpoints 就绪
- [x] 推理管线验证通过
- [ ] **Phase 0: 基础设施 (src/rl/)**
