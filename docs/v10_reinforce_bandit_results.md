# v10: REINFORCE Bandit + Continuous Reward 实验结果

> 日期: 2026-06-15
> 状态: **部分成功** — 训时 3/4 cell, 推理 3/4 cell, avg_R 未改善
> 关联: `docs/algorithm_decisions.md`, `docs/mixture_policy_results.md` (v7-v8), `scripts/train_v10_reinforce_bandit.py`

## 1. 动机

8 版 RL 失败后重新从第一性原理出发:

- **问题本质**: Contextual Bandit (单步决策), 不是 MDP
- **正确工具**: REINFORCE with baseline (Williams 1992), 不是 SAC/HER/多步
- **正确 reward**: Continuous distance-based (Sutton & Barto Ch.3), 不是 0/1
- **正确策略**: Mixture of K Gaussians (Bishop 1994 - MDN), per-mode 独立

**论文支撑**: Williams 1992 + Bishop 1994 + Mnih 2016 (A3C entropy)

## 2. 架构

```
obs (192) + goal_onehot (16)
       ↓ shared MLP(256)
   (μ, σ, π)  ← K=4 mode heads
       ↓ sample from mode k (fixed mapping)
z ~ N(μ_k, σ_fixed=0.5)
       ↓ decoder → trajectory → collision check
R = -min_dist (collision target cell)
  = -min_dist (no collision)
  = -min_dist*0.5-1 (collision wrong cell)

Per-mode baseline (EMA of rewards)
REINFORCE update: L = -log_prob · (R - baseline[k])
Entropy bonus: -0.05 · H (keep exploration)
```

### 关键设计

- **Fixed σ = 0.5**: 不训练 σ, 防止探索消失
- **Per-mode baseline**: 4 个 mode 各有独立的 running mean
- **Continuous reward**: 每一步都有标量信号
- **No critic**: MC return = ground truth (bandit 不需要 bootstrap)
- **No HER**: bandit 没有序列, HER 无意义

## 3. 训练曲线

- **n_episodes**: 2000
- **每 episode**: 1 次 forward pass + decode (~0.24s)
- **总时间**: ~8 分钟 (1 GPU)

### 3.1 avg_R 曲线 — 未改善

```
ep 100: avg_R=-16.576, hit_rate=0.010, cells_filled=1/4
ep 200: avg_R=-20.263, hit_rate=0.005, cells_filled=1/4
ep 700: avg_R=-20.633, hit_rate=0.003, cells_filled=2/4
ep 1000: avg_R=-19.911, hit_rate=0.002, cells_filled=2/4
ep 1700: avg_R=-20.057, hit_rate=0.004, cells_filled=4/4  ← 首次 4/4!
ep 2000: avg_R=-21.178, hit_rate=0.003, cells_filled=4/4
```

**avg_R 全程在 ~-20 波动, 无上升趋势。** Policy 的 μ 未被推到位。hit rate 维持在 0.3-0.5% (与 prior perturbation 一致)。

### 3.2 调参历程

| 版本 | 改动 | 训练 cells_filled | 推理 cells_filled |
|------|------|------------------|------------------|
| v10a | clipped reward [-6,0] | 0/4 | 0/4 |
| v10b | unclipped reward + σ floor | 4/4 | 3/4* |
| v10c | fixed σ=0.5 + entropy 0.05 | 3/4 | 3/4 |

*v10b 推理时 σ 使用网络原始输出 (太小), 后修复为 fixed σ=0.5

## 4. 最终训练结果

| Cell | 命中次数 | min_dist | Baseline |
|------|---------|----------|----------|
| 2 (rear front) | 2 | 0.612m | ~-20 |
| 6 (right rear) | 1 | 1.854m | ~-20 |
| 10 (left rear) | 1 | 1.779m | ~-20 |
| 14 (rear left) | 2 | 1.617m | ~-20 |

**4/4 cells 在训时有 hit**, 但每个 cell 仅 1-2 次命中 (2000ep = 500ep/mode/cell, hit rate ~0.5%)。

## 5. 推理结果 (200 samples/goal)

| Goal | Cell | Mode | Min Dist | 状态 |
|------|------|------|----------|------|
| 2 | rear front | 0 | 1.338m | ✓ |
| 6 | right rear | 1 | 1.227m | ✓ |
| 10 | left rear | 2 | ∞ | ✗ miss |
| 14 | rear left | 3 | 1.690m | ✓ (训时 0 hit!) |

### 推理分析

- Goal 14 (cell 14): 训时 0 次命中,**推理时却命中** — 说明命中靠 200 次采样的偶然值, 不是 μ 学到了正确区域
- Goal 2, 6: 命中 min_dist ~1.2-1.3m, 勉强可接受
- Goal 10: 200 次采样中 0 次命中 — mode 2 的 μ 完全不在解区域

## 6. 与 v3~v9 对比

| 版本 | 算法 | 训时 cells | 推理 cells | min_dist 推理 |
|------|------|-----------|-----------|--------------|
| v3-v6 | SAC/REINFORCE | 0-4/16 | 0-1/4 | ∞ |
| v7-v8 | Mixture REINFORCE | 4/16 | 1/4 | 0.57m |
| v9 | Mixture SAC+HER | 2/16 | 1/4 | 0.57m |
| **v10** | **REINFORCE Bandit** | **4/4** | **3/4** | **1.23-1.69m** |

v10 在推理 cells_filled 上有本质提升 (1/4 → 3/4), 但这主要是因为 **fixed σ 防止了推理退化**, 不是 μ 被学到位了。

## 7. 3 重矛盾分析 (铲屎官诊断)

### 矛盾 A: miss 噪声淹没 hit 梯度
```
500 miss samples × |advantage| ≈ 0 → backward() 仍发生
1 hit sample  × |advantage| ≈ +15 → backward() 朝正确方向
共享 backbone: 500 次微小噪声累积 + 1 次强 signal = 噪声取胜
```

### 矛盾 B: 高维梯度方向不一致
```
hit_1 的 z 在 32D 中朝方向 d1 → gradient ∝ (z_1 - μ) 
hit_2 的 z 在 32D 中朝方向 d2 → gradient ∝ (z_2 - μ)
d1 ⟂ d2 (32D 空间中方向几乎正交)
两次 gradient 相互抵消 → μ 原地踏步
```

CMA-ES 解决矛盾 B 的机制: 取 top-k z 的均值 (而非单个 z 的 gradient); mean 方向 > gradient 方向。

### 矛盾 C: Binary gate 更新频率低
```
如果只保留 hit 样本 (binary gate):
  0.5% × 2000ep × 1 mode = ~2.5 hits per mode
  2.5 次 backward() 不够训共享 backbone
如果保留全部样本:
  回到矛盾 A
```

## 8. 结论

**v10 是纯 RL 在 sub60 上最好的结果** (推理 3/4 cells), 但 **avg_R 未改善 + 推理靠 200 次采样碰运气** 说明核心问题未解决: **μ 未被推到位。**

**根本原因**: REINFORCE 的 gradient-based update 在高维连续空间 + 极稀疏信号下, 梯度方向不一致 + 噪声淹没。不是 RL 算法不够好, 是单场景 Bandit 在 32D z 空间有物理极限。

**下一步**: Asymmetric Self-Play (v11) — 引入 Teacher-Student 对抗, 自动 curriculum, 打破 0.5% 固定碰撞率的僵局。
