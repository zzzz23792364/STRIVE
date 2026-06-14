# Experiment 1: Prior Perturbation Probe — Results

> 日期: 2026-06-14
> 状态: **已完成**
> 关联: `docs/exp1_prior_perturbation_plan.md`, `out/prior_perturbation/*`

## 1. 实验配置

- 场景: val subseq 60 (NA=2, init_d=6.42m, map_idx=3)
- 模型: `model_ckpt/traffic_model.pth`
- 设备: RTX 3090 (远程)
- 种子: 42
- σ 档位: [1, 3, 5, 7, 9, 11]
- sample/档: 128 (Q2=B)
- 总 decode: 768
- 碰撞检测: 向量化中心距离 < (avg_lw / 2) ≈ 1.0m

## 2. 结果总表

| sigma | coll_rate | md_median | md_q25 | md_q75 | bd_cov | avg_t |
|:-----:|:---------:|:---------:|:------:|:------:|:------:|:-----:|
| 1×    |    3.9%   |   1.218   |  0.955 |  1.463 |  3/16  | 10.6  |
| 3×    |    4.7%   |   1.527   |  1.387 |  1.684 |  4/16  |  9.3  |
| **5×**|  **7.8%** |   1.393   |  0.925 |  1.575 |  4/16  |  8.4  |
| 7×    |    5.5%   |   1.479   |  1.134 |  1.526 |  4/16  |  8.1  |
| 9×    |    3.1%   |   1.065   |  0.771 |  1.278 |  2/16  |  8.5  |
| 11×   |    3.1%   |   1.214   |  1.071 |  1.317 |  3/16  |  8.5  |

**碰撞率峰值 σ=5× (7.8%)**, 1σ/3σ 偏紧、9σ/11σ 偏松（飞出去）。

## 3. 关键发现

### 3.1 BD coverage 极不均匀

**所有命中点都聚集在 cell {2, 6, 10, 14}** (heading_bin=2, 180° 朝后):
- cell 2: pos_bin=0, heading_bin=2
- cell 6: pos_bin=1, heading_bin=2
- cell 10: pos_bin=2, heading_bin=2
- cell 14: pos_bin=3, heading_bin=2

heatmap 显示 cell 2, 6, 10, 14 总计 36 个命中, 其他 12 个 cell **零命中**。

### 3.2 物理原因

**val subseq 60 初始布局决定了一切**:
- atk_idx=1 位于 ego 斜后方 (init_d=6.42m)
- ego 走直线 (prior mean 平滑轨迹)
- atk 必须"调头追赶"才能撞 ego
- 调头意味着 atk 朝向必然 = ego 局部坐标的 180° (朝后)
- 撞到时的 4 种 pos_bin = atk 撞到 ego 4 个不同面 (前/右/后/左)
- 其他 12 cell (heading ≠ 180°) 物理上不可达

**结论**: 16 网格对 sub60 **过度细分**。实际可填 cell 数 ≤ 4。

### 3.3 min_dist 分布特征

| σ | 分布特征 |
|---|---------|
| 1× | 紧凑 (0-5m) — 不足以撞 (撞的 5 个都是边界 1-1.5m) |
| 3-5× | 双峰 — 5m 和 30m (部分飞出去) |
| 9-11× | 35-40m 峰显著 — 严重飞离 |

碰撞率峰值 7.8% (< 10%) — 即使是 σ=5× 也不算"容易撞"。

### 3.4 碰撞时间

集中在 t=7-11 (未来 3.5-5.5s) — 6s 时间窗下需要 1-2s 调头, 然后 3-4s 直线冲刺。

## 4. 对算法选择的指导意义

### 4.1 排除 GRPO 的根据

| 现象 | 对 GRPO 的影响 |
|------|---------------|
| 碰撞率 7.8% (峰值) | K=32 group 内 < 3 个撞到 → reward variance 极小 |
| 16 cell 实际只填 4 个 | archive 12/16 永远空 → "novelty" 信号弱 |
| 0.9-1.7m min_dist 距 0 较远 | policy 输出需大幅"放大"才能撞, 需强梯度 |

**GRPO 在 sub60 单场景很难学到东西** — 单 episode (scene) 反馈信号太稀疏。

### 4.2 RL 路径的可行性

| 路径 | 适配度 | 关键风险 |
|------|--------|---------|
| **PGA-MAP-Elites** | 🟡 中 | 16 cell 12 个空, archive 浪费; Critic 在单 scene 训不动 |
| **GC-SAC+HER** | 🔴 低 | 16 goal 12 个不可达, HER 重标记无效; 单步 SAC 退化 |
| **IL + 条件式 Policy** | 🟢 高 | 不依赖 RL 反馈, 直接学 (scene, bd) → z 的映射; 训练数据可由优化器产 |

### 4.3 子场景评估

**实际能填的 BD cell = 4 个** (cell 2, 6, 10, 14):
- (0, 2): atk 在前, 朝后 (实际位置=前)
- (1, 2): atk 在右, 朝后
- (2, 2): atk 在后, 朝后 (典型追尾)
- (3, 2): atk 在左, 朝后

由于 heading 固定, **BD 实际只有 4 个有效 cell** (pos_bin 一维)。

## 5. 实验局限性

1. **碰撞检测是近似**: 用中心距离 < 1.0m 代替 shapely IoU; 可能漏判边缘 case
2. **单种子**: 128 sample 在 σ=1× 时只 5 个撞, 统计噪声较大
3. **单场景**: 只在 sub60 跑, 其他 subseq 可能不同

## 6. 下一步建议

### 6.1 短期 (基于本次数据)

1. **重新审视 BD 维度**:
   - sub60 实际可填 = 4 cell → **改用 4 网格 (1×4, 仅 pos_bin)** 或 **4 cell (heading 固定, 仅 pos)**
   - 或者 **接受 16 网格但承认 sub60 只能填 4 cell** → 算法目标改为"填满 4 个有效 cell"

2. **算法路径调整**:
   - **不推荐 GRPO** (信号太稀疏)
   - **推荐**: 阶段 1.5 用 **Gradient+Restart × 16 次** 拿到 sub60 多解 baseline (和现有 adv_scenario_gen.py 一致, 改 z_init 即可)
   - 阶段 3: 训 **Conditional Policy** π(scene, bd_idx) → z

### 6.2 中期 (阶段 1.5 → 阶段 2.5)

1. 把 Gradient+Restart 在 5-10 个其他 subseq 也跑一遍
2. 累计 (scene, z*) 数据集
3. 训 Conditional Policy → zero-shot 跨场景泛化

### 6.3 待讨论决策点 (铲屎官确认)

1. **BD 网格重设**: 是用 1×4 (4 cell) 还是 4×4 (16 cell, 接受 12 个空)?
2. **算法路线**: IL + Conditional Policy vs PGA-MAP-Elites vs 其他?
3. **数据生成器**: Gradient+Restart vs 现有 adv_scenario_gen.py vs 新写?

## 7. 产物清单

```
docs/exp1_prior_perturbation_results.md  # 本文档
docs/exp1_prior_perturbation_plan.md     # 实验规划
docs/bd_design_spec.md                   # BD 设计规范
docs/algorithm_decisions.md              # 算法选型决策记录
src/rl/prior_perturbation.py             # 核心探测函数
scripts/exp1_prior_perturbation.py       # 实验入口
out/prior_perturbation/
├── sigma_stats.json                     # 6 档统计
├── all_samples.json                     # 768 个完整 sample
├── bd_scatter.png                       # BD 散点图
├── min_dist_dist.png                    # min_dist 直方图
├── coll_time_dist.png                   # 碰撞时刻分布
└── sigma_cell_heatmap.png               # sigma × cell 热力图
```

## 8. 关键数字摘要 (给决策者)

- **碰撞率峰值 7.8%** (σ=5×, 128 sample) — 单 episode 撞到的概率 ~8%
- **实际可填 BD cell = 4** (heading 固定 180° 朝后)
- **min_dist 范围 0.8-1.7m** — 远未触底 (vs 梯度基线 0.97m 已经很接近)
- **平均碰撞时刻 t=8-10** (3.5-5s 未来)
- **GRPO 不适合** (信号稀疏), **IL 路径最稳**
