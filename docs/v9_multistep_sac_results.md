# GC-SAC + Mixture Policy v9 训练结果 — 单场景 sub60

> 日期: 2026-06-14
> 状态: ⚠️ **部分成功** — 训时 2/16 cell, 推理 1/4 cell (退化)
> 关联: `docs/algorithm_decisions.md`, `docs/mixture_policy_results.md` (v7-v8)

## 1. v9 的动机

铲屎官敏锐诊断: **"v8 那是 REINFORCE 不是 SAC, 没有 critic"**。
v8 用 `reinforce_loss = -log_prob * advantage`, 没有 Q 函数、target net、bootstrap。
理论缺陷:
- 无 critic → variance 极高, 只能靠稀疏 reward
- 无 bootstrap → 多步 episode 优势浪费
- 无 target net → Q 会自我发散

v9 重写为完整 GC-SAC+HER+Mixture+Critic。

## 2. 架构

```
obs (10) + goal_emb (16) + mode_emb (4)
       ↓ shared MLP(256)
   (μ, σ, π)  ← K=4 mode heads
       ↓ sample z
Q1, Q2(obs+goal+z+mode)  ← per-(goal, mode) critic
       ↓ bootstrap
   target Q (Polyak 0.995)
```

## 3. 训练曲线

- **n_episodes**: 1000
- **n_steps_per_ep**: 5 (multi-step)
- **n_modes**: 4
- **lr**: 5e-5
- **target clip**: Q ∈ [-1, 2]
- **grad_clip**: 0.1

| Metric | v8 | v9 |
|--------|----|----|
| Pi loss 终值 | -1e15 (爆) | -2.1e9 (10 个数量级改善, 仍恶化) |
| Q loss 终值 | n/a | 0.6 (稳定) |
| Avg reward 终值 | 0.2-0.5 | 0.5-0.9 |
| Cells hit (训时) | 4/16 | 2/16 |
| Cells hit (推理) | 1/4 (cell 14) | 1/4 (cell 14) |

## 4. 撞 cell 分布

```
mode_hits = {
    "14_3": 1,   # mode 3 撞 cell 14
    "2_3": 1,    # mode 3 也撞 cell 2
}
```

**问题**: 1000 轮里只 2 次 hit, 而且是同一个 mode 3 同时承担两个 cell — **mode 没分化**。

## 5. 关键发现 / Bug 修复

1. **v9a 爆炸**: 初版 Q=175000, pi=-3.9e12
   - **修复**: target clip Q ∈ [-1, 2] + 缩小 lr 至 5e-5
2. **Pi loss 持续下降**: 即使加了 clip, π loss 仍从 -3.0 → -2.1e9
   - 表明 policy entropy 还在涨, 探索仍是 prior 状态
3. **HER ratio 实际 ~0.4-0.6** (标准 0.8): 多数 episode 没成功 trajectory 可重标

## 6. 推理结果 (192 trials/cell)

| Cell | Best dist | Hit |
|------|-----------|-----|
| 2 | ∞ | ✗ |
| 6 | ∞ | ✗ |
| 10 | ∞ | ✗ |
| 14 | 0.571m | ✓ |

**单 cell 推理退化**: 比 v8 (训时 4/16) 还退一步 (推理 1/4 vs 训时 1/4)

## 7. 与 v8 / 梯度基线对比

| 方法 | 训时 cells | 推理 cells | min_dist | 泛化性 |
|------|-----------|-----------|----------|--------|
| Gradient baseline (4 iter) | n/a | 4/4 | 0.97m | ✅ |
| Prior Perturbation (768) | n/a | 4/4 | 0.7-3.1m | ✅ |
| PGA-MAP-Elites (200) | n/a | 4/4 | 0.006-0.1m | ❌ 死记硬背 |
| Mixture v8 (REINFORCE) | 4/16 | 1/4 (cell 14) | 0.571m | ❌ |
| **Mixture v9 (GC-SAC)** | **2/16** | **1/4 (cell 14)** | **0.571m** | **❌** |

## 8. 根因分析

**单场景纯 RL 探索多解有根本性 sample efficiency 上限**:
- 1000 episode × 5 step = 5000 样本
- HER 标记后 ~3000 有效样本
- 撞墙正例极稀: ~2 次/1000 ep
- 远不够 4 mode 分化学习

**v9 没有比 v8 更好的根本原因**:
- Critic 在 hit rate = 0.2% 时学不到东西
- Multi-step bootstrap 在 hit 极少时同样无效
- 加了 critic 反而引入 boot target 噪声

**PGA-ME 才是这个问题的正确解法**:
- QD 优化不需要"撞到 4 cell 才给 reward"
- 它直接把 4 cell 都当 fitness, 用 novelty/quality 选择
- 所以 200 iter 4/4 0.006-0.1m
- 但代价: **archive 是 scene-specific z, 不跨场景**

## 9. 结论

**单场景混合多解策略**: GC-SAC+HER+Mixture 在 sub60 训时最多 4/16, 推理 1/4, **未达成目标**。
- v8 (REINFORCE) 和 v9 (GC-SAC) 推理均只 1/4 cell
- 多解学习所需信号远高于当前样本量
- 推荐路径: **跨场景蒸馏** (PGA-ME 训 200+ subseq → IL 蒸馏通用 policy)

## 10. 关联文件

- `scripts/train_gc_sac_her.py`: v9 训练脚本
- `out/gc_sac_her/`: 训练输出 (policy.pt 770KB, q1/q2.pt 569KB, 4 cell viz)
- `src/rl/mixture_policy.py`: Mixture 架构
- `src/rl/conditional_policy.py`: QNetwork / ConditionalPolicy
