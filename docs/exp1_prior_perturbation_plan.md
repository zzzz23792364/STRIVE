# Experiment 1: Prior Perturbation Probe (PPP)

> 日期: 2026-06-14
> 状态: **待执行**
> 关联: `docs/algorithm_decisions.md`, `docs/bd_design_spec.md`

## 1. 实验目标

回答 4 个关键问题:

| # | 问题 | 用途 |
|---|------|------|
| Q1 | σ 几倍能撞到? | 给 GRPO/CMA-ES 的 sigma_init 标定 |
| Q2 | 每 σ 倍数能填多少 BD cell? | 给 16 网格多解的天花板标定 |
| Q3 | min_dist 分布? | 给 RL reward scale 标定 |
| Q4 | prior 物理可塑性? | 给"是否需要 RL 训练"提供证据 |

## 2. 实验设计

### 2.1 场景
- val subseq 60 (NA=2, init_d=6.42m, map_idx=3, boston-seaport)

### 2.2 数据集配置（与 adv_scenario_gen.py 一致）
```python
NuScenesDataset(data_path, map_env, version="trainval", split="val",
    categories=["car", "truck"], npast=4, nfuture=12,
    reduce_cats=False, seq_interval=10,
    randomize_val=True, val_size=400)
```

### 2.3 变量

```
sigma_multiplier ∈ {1, 3, 5, 7, 9, 11}    # 6 档
samples_per_sigma = 128                    # Q2 答案
random_seed = 42                            # Q3 答案
n_total_decodes = 6 × 128 = 768
```

### 2.4 核心公式

```python
# 取 prior
prior_mu, prior_var = ei['prior_out']    # (NA, 32)
sigma = torch.sqrt(prior_var)             # (NA, 32)

# 扰动 (Q4: 只做 prior)
eps = torch.randn_like(prior_mu) * sigma_multiplier
z = prior_mu + sigma * eps                # (NA, 32)

# Decode
dec = model.decode_embedding(z, embed_info, sg, mi, map_env)
fut = model.get_normalizer().unnormalize(dec['future_pred'])  # (NA, FT, 4)

# 碰撞检测
ego_replay = model.get_normalizer().unnormalize(sg.future_gt[ego_mask])  # (1, FT, 4)
decoded_atk = fut[atk_idx, :, :4]  # (FT, 4)
veh_coll, coll_time = check_single_veh_coll(
    ego_replay[0, :, :4], ego_lw,
    decoded_atk.unsqueeze(0), atk_lw.unsqueeze(0)
)
```

## 3. 实验输出

### 3.1 统计指标 (per sigma)

| 指标 | 含义 |
|------|------|
| collision_rate | 128 sample 中撞到的比例 |
| min_dist_median | 撞到的 sample 的 min_dist 中位数 |
| min_dist_q25, q75 | min_dist 四分位数 |
| bd_coverage | 实际填充的 BD cell 数 (0-16) |
| avg_coll_time | 撞到的 sample 第一次碰撞时刻 (T index) |

### 3.2 可视化 (Q5: ABCD 全要)

- A. BD 散点图: `bd_scatter.png`
  - x = pos_bin (0-3), y = heading_bin (0-3)
  - color = sigma multiplier
  - 标记: 6 档叠加, 只画撞到的
- B. min_dist 直方图: `min_dist_dist.png`
  - 6 个子图 (per sigma)
  - 撞到的 sample 蓝色, 未撞到的灰色
- C. 碰撞时刻分布: `coll_time_dist.png`
  - 6 个子图
  - x = collision time step (0-11), y = count
- D. sigma-cell 热力图: `sigma_cell_heatmap.png`
  - x = sigma, y = BD cell index (0-15)
  - color = 撞到的 sample 数

### 3.3 文件清单

```
out/prior_perturbation/
├── sigma_stats.json          # 6 档统计
├── all_samples.json          # 768 个 sample 完整记录
├── bd_scatter.png            # 可视化 A
├── min_dist_dist.png         # 可视化 B
├── coll_time_dist.png        # 可视化 C
└── sigma_cell_heatmap.png    # 可视化 D

docs/exp1_prior_perturbation_results.md   # 结论文档
```

## 4. 代码实现

### 4.1 新模块

- `src/rl/prior_perturbation.py`: 核心探测函数
- `scripts/exp1_prior_perturbation.py`: 实验入口

### 4.2 复用代码

| 任务 | 复用 |
|---|---|
| 加载 subseq 60 | `scripts/adv_opt_sub60_fast.py:1-49` |
| `embed()` 取 prior | `src/models/traffic_model.py:372-403` |
| Reparameterize | `src/models/traffic_model.py:706-712` |
| Decode | `src/models/traffic_model.py:405-414` |
| 碰撞检测 | `src/losses/adv_gen_nusc.py:517-565` |
| Ego logger replay | `sg.future_gt[ego_mask]` |
| 局部坐标系 (BD 计算) | `src/utils/transforms.py:78` |

## 5. 实施步骤

```
Step 1: 写本规划文档 (docs/exp1_prior_perturbation_plan.md) - done
Step 2: 实现 src/rl/prior_perturbation.py
Step 3: 实现 scripts/exp1_prior_perturbation.py
Step 4: 上传到远程服务器
Step 5: 运行实验, 记录输出
Step 6: 写 docs/exp1_prior_perturbation_results.md
Step 7: 决策算法 (基于实验数据, 不基于预设阈值)
```

## 6. 决策映射 (实验后讨论)

> **注**: Q6 用户要求"现有实验结论之后再讨论决策阈值", 不预设阈值。
> 决策点将在 `exp1_prior_perturbation_results.md` 中基于实际数据讨论。

占位待填:
- collision_rate 曲线形状 → sigma scale 选择
- BD coverage 分布 → 16 网格是否合理
- min_dist 分布 → reward scale 标定
- 单 cell 集中度 → 是否需要 QD 框架
