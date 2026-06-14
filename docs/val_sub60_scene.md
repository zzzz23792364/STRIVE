# 验证场景: val subseq 60

## 基本信息

| 项目 | 值 |
|------|------|
| 数据集 | nuScenes trainval |
| Split | val |
| Subseq 索引 | 60 |
| NA（agent 数量） | 2（1 ego + 1 other） |
| 类别 | car + truck |
| Map | singapore-onenorth (map_idx=3) |
| 初始最小距离 | 6.42m |

## 数据加载参数

必须与 `adv_scenario_gen.py` 保持一致：

```python
NuScenesDataset(data_path, map_env,
    version="trainval",
    split="val",
    categories=["car", "truck"],
    npast=4, nfuture=12,
    seq_interval=10,
    randomize_val=True,
    val_size=400)
```

## 优化结果

| 方法 | 初始 min_dist | 最终 min_dist | 成功 |
|------|-------------|--------------|------|
| 梯度基线 (adv_scenario_gen) | 6.42m | **0.97m** | ✅ True |
| GRPO + 残差 | 待验证 | 待验证 | 待验证 |

## 可视化

```
out/viz_adv_sub60/before.png  ← 优化前（后验轨迹）
out/viz_adv_sub60/after.png   ← 优化后（对抗轨迹）
```

## 使用方式

### 原版梯度优化（已确认可解）

```bash
cd /root/autodl-tmp/STRIVE
python scripts/adv_opt_sub60_fast.py
```

### GRPO + 残差测试

基于 `scripts/grpo_mini.py` 修改，替换场景加载参数为上述参数+subseq=60。

## 说明

- 这是目前唯一确认梯度基线可解的 NA=2 场景
- 后续 RL 实验（REINFORCE / PPO / GRPO）统一使用此场景进行对比
- 可视化路径：`out/viz_adv_sub60/`
