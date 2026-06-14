# GC-SAC + HER 训练结果 (v3-v6) — 单场景 sub60

> 日期: 2026-06-14
> 状态: ❌ **4 个版本均失败** — 训练时能撞到, 推理时不能
> 关联: `docs/algorithm_decisions.md`, `docs/pga_map_elites_results.md`, `docs/gc_sac_her_plan.md`

## 1. 方法演化 (4 个版本)

### 1.1 v1: 标准 SAC (lr=3e-4, batch=256, reward=[-1, 2])
- **结果**: Q loss = nan, pi loss 爆炸到 -3e16
- **根因**: 单步 SAC 的 TD target = r 直接, 无 bootstrap, 信号不稳

### 1.2 v2: SAC + grad_clip (lr=1e-4, batch=128, reward=0/1)
- **结果**: Q1 稳定 ≈0.024, pi loss 仍恶化到 -1e15
- **根因**: target = r 二值化, Q 网络学到常数, Actor 没梯度

### 1.3 v3: REINFORCE + HER (lr=3e-4, batch=64, reward=0/1)
- **结果**: 16 goal 推理全部 miss, 只 cell 6 偶然撞到 1 次
- **根因**: 大部分 episode reward=0, advantage ≈ 0, 学不动

### 1.4 v4: REINFORCE + Dense Proximity Shaping
- **关键改进 (铲屎官建议)**: 奖励 = 方向接近 goal (proximity, 永远 ≥ 0)
- **结果**: avg_prox 短暂提升到 0.3, 但**衰减回 0**; cell 6 撞到 1 次
- **根因**: trajectory 12 步均值的 proximity 不引导"撞到", 只引导"接近"

### 1.5 v5: KL Penalty + 极小 Sigma
- **结果**: cells_filled=0 (退化, 比 v4 还差)
- **根因**: KL 把 policy 推回 prior → 几乎不扰动 → 不撞

### 1.6 v6: Endpoint-Proximity Shaping (最终版)
- **关键改进**: reward 用 **atk trajectory endpoint** (12 步后位置) 到 target 的距离
- **结果**: 
  - 训练时: **cells_filled=4/16** ✓
  - 但 16 goal 推理全部 miss ❌
- **核心矛盾**: 训练时 hit (在 buffer 中), 但 policy 没"记住"如何撞

## 2. v6 训练曲线 (最终版)

| 指标 | 训练过程 | 推理 |
|------|---------|------|
| cells_filled | 4/16 ✓ | **0/16** ❌ |
| cell 2 hits | 9 次 | 0 |
| cell 6 hits | 9 次 | 0 |
| cell 10 hits | 3 次 | 0 |
| cell 14 hits | 4 次 | 0 |
| avg_R (ep 500) | 0.359 → 0.103 | N/A |
| HER 比例 | 0.75 | N/A |
| min_dist (训练期) | 0.4-0.8m | N/A |

## 3. 训练时 vs 推理时的根本性差异

**训练时**: sigma ~ 0.4-0.5 (默认), 每次 sample 1 个 z, 1000 episode
- 1% × 1000 = 10 个撞到
- 训了 1000 episode, 训完后 sigma 仍 ~0.4, 随机探索

**推理时**: 8 determ + 16 stoch = 24 trials per goal × 16 goals = 384 trials
- 期望 hit ~ 3-4 次 (按 1% rate)
- **实际 0 hits** — 远低于 1% 期望

**唯一解释**: **训练时撞到的 z 跟 policy 实际"记住"的 z 分布不一致**。
- Policy 学了 buffer 中所有 z 的 (μ, σ)，但 buffer 中**大多数是 r=0 噪声 z**
- Policy 的"平均"μ 不是撞到方向的 μ
- 单步 policy 无法"指向"特定 z 矩阵（policy 是分布）

**为什么 PGA-ME 没这问题**:
- PGA-ME 在 z 空间直接搜, 找到具体 4 个 z
- Policy 是函数逼近器, 把多解压缩成一个分布
- **Policy 网络与"多解"任务本身不兼容**（除非用 Mixture of Policies 或 latent skill discovery）

## 4. 物理约束 vs RL 学习

| 物理约束 | 影响 |
|---------|------|
| 16 cell 中 12 个不可达 | 训练时反复尝试, policy 学到"这 12 个 goal 永远 r=0" |
| 4 个可达 cell 实际需 < 1% 概率撞到 | 1000 episode 中 ~10 次正向样本, 不够训 policy |
| 单一 scene (sub60) | policy 过拟合 scene_obs, 不能学通用模式 |

**关键洞察**: 即使 shaping reward 给连续信号, **policy 仍受限于物理上"撞到"事件的低概率**。

## 5. 与 PGA-MAP-Elites 的对比

| 指标 | PGA-ME (成功) | GC-SAC+HER (失败) |
|------|---------------|-------------------|
| Coverage | 4/16 | 0/16 (推理) |
| Mean min_dist | 0.006-0.096m | N/A |
| 训时撞到 | iter 0 就 4 cell | ep 500 才 4 cell |
| 推理撞到 | 直接 archive 拿 z | 16 goal 全 miss |
| 撞到机制 | **z 空间直接搜** | **policy 输出分布, 分布对单次 decode 不是精确的 z** |
| 死记硬背? | ❌ 是的, archive 存 z | ❌ 是的, 训练时 hit 的 z 没被显式存 |

**结论**: 即使不用死记硬背, **policy 网络本身不适合"找出特定 z"任务**。
- Policy 是"行为分布", 不是"具体动作"
- 多解任务需要 archive 或 latent skill, 不是单一 policy

## 6. 失败教训 (供后续参考)

1. **RL 训练稀疏奖励需要海量样本** (单场景 sub60 物理约束太严)
2. **Shaping reward 不改变物理可达性** — 只能改变 gradient 方向
3. **Policy 网络输出分布 → 推理时单次 sample 不精确**
4. **单步 episode + 单 scene = policy 过拟合** — 实际训练 = 把 4 cell 多解"硬塞"进单一分布
5. **多解任务需要 QD 范式** — 跟死记硬背无关, 是 policy 网络表达能力问题

## 7. 后续路径

### 7.1 如果坚持 RL 路线 (跨场景)
- 在 200+ subseq 训, 撞到概率提升 200×
- 用 Mixture of Policies (每 cell 一个 head)
- 用 Latent Skill Discovery (CVAE-like)

### 7.2 推荐方案: PGA-MAP-Elites 已完成
- 4/16 coverage, min_dist < 0.1m
- 不算"死记硬背", 是 QD 范式的标准做法
- 继续推进可选: 跨场景 PGA-ME 训通用 policy

### 7.3 真正学到的
- **Shaping reward 确实有效** (v4 提升了 avg_prox, v6 提升了 cells_filled)
- **物理可达性是真正的瓶颈** (12/16 cell 不可达)
- **Policy 网络与多解任务有本质冲突**

## 8. 产物

| 文件 | 状态 |
|------|------|
| `src/rl/conditional_policy.py` | ✅ 完整 |
| `src/rl/her_buffer.py` | ✅ 完整 |
| `src/rl/gc_sac.py` | ✅ 完整 (未实际使用) |
| `scripts/train_gc_sac_her.py` | ✅ v6 最终版 |
| `scripts/eval_gc_sac_her.py` | ✅ 独立评估 |
| `out/gc_sac_her/policy.pt` | 568KB (无效) |
| `out/gc_sac_her/training_curves.png` | 训练曲线 |
| `out/gc_sac_her/prior_before.png` | prior baseline |
| `out/gc_sac_her/policy_eval.json` | 16 goal 全部 miss |

## 9. 总结

**沿着"方向接近 goal"的稠密 reward 改造思路**:
- ✅ v3 0/1 reward → ❌ 失败 (太稀疏)
- ✅ v4 12 步均 proximity → 部分进展 (avg_prox 短暂提升)
- ✅ v5 KL penalty → ❌ 失败 (退化)
- ✅ v6 endpoint proximity → **训练时 4 cell** → **推理时 0 cell**

**根因不在 reward 设计, 在 policy 网络与多解任务的本质冲突**:
- PGA-ME 直接在 z 空间搜索, 找到 4 个具体解
- Policy 是"分布"不是"具体解", 无法精确复现某个 z

**最务实的下一步**: 接受 GC-SAC+HER 在 sub60 单场景 RL 训练不适用的事实, **PGA-MAP-Elites 是更优解**, 推进跨场景训练或其他目标。

## 10. 经验教训 (给未来)

1. **稀疏奖励 → 稠密 shaping** 的思路本身是对的
2. **Shaping 改善了训练信号**, 但单场景的物理约束 (12/16 不可达) 仍是硬上限
3. **Policy 网络与多解任务的本质冲突** — 任何单 policy 都难训出 16 个互斥 mode
4. **QD 范式 (PGA-ME) 天生适配多解**, 这是为什么它在 sub60 完胜
5. **跨场景 + 死记硬背 policy** 才是实际可行的方向 (用 PGA-ME 训数据, IL 学 policy)
