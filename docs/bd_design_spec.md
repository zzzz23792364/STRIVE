# 碰撞行为描述子 (BD) 设计规范

> 日期: 2026-06-14
> 关联: 论文 STRIVE §4.1, `src/cluster_scenarios.py:96-121` (compute_coll_feat)

## 1. 设计决策

**采用 16 网格离散 BD** (4×4 = pos_bin × heading_bin)

**bin 划分方案 B**: 中心在 0°/90°/180°/270° (边在 ±45°)

```python
def angle_to_bin(angle_deg: float) -> int:
    """[-180°, 180°) → {0,1,2,3}, 中心 0°/90°/180°/270°"""
    a = (angle_deg + 45.0) % 360.0
    return int(a // 90.0)
```

| bin | 角度范围 | 中心 | 语义 (ego 局部坐标系) |
|-----|---------|------|---------------------|
| 0 | [-45°, 45°) | 0° | 前/后 (沿纵轴) |
| 1 | [45°, 135°) | 90° | 右/左 (横轴正) |
| 2 | [135°, 225°) | 180° | 后/前 (纵轴反向) |
| 3 | [225°, 315°) | 270° | 左/右 (横轴负) |

## 2. 特征计算

### 2.1 pos_angle
- 在 collision moment, atk 相对 ego 的位置向量
- 转到 ego 局部坐标系 (用 `transform2frame`)
- `pos_angle = atan2(local_y, local_x)` ∈ [-180°, 180°)

### 2.2 heading_angle
- atk 相对 ego 的朝向
- 同样转到 ego 局部坐标系
- `heading_angle = atan2(local_hy, local_hx)` ∈ [-180°, 180°)

### 2.3 BD 索引
```python
bd_idx = pos_bin * 4 + heading_bin  # 0-15
```

## 3. 16 cell 语义映射

| (pos, h) | pos 中心 | h 中心 | 论文对应类型 | 语义 |
|---|---|---|---|---|
| (0, 0) | 0° 前 | 0° 朝前 | Cut-in from front | atk 在前、朝前开 |
| (0, 1) | 0° 前 | 90° 朝右 | Cut-in from front-right | atk 在前、横着开 |
| (0, 2) | 0° 前 | 180° 朝后 | Head On | atk 在前、朝 ego 反向 (正面对撞) |
| (0, 3) | 0° 前 | 270° 朝左 | Cut-in from front-left | atk 在前、横着开 |
| (1, 0) | 90° 右 | 0° 朝前 | T-bone from right | atk 在右、朝前 (侧面撞) |
| (1, 1) | 90° 右 | 90° 朝右 | Sideswipe from right | atk 在右、同向侧擦 |
| (1, 2) | 90° 右 | 180° 朝后 | Sideswipe opposite right | atk 在右、反向侧擦 |
| (1, 3) | 90° 右 | 270° 朝左 | T-bone right crossing | atk 在右、横切 |
| (2, 0) | 180° 后 | 0° 朝前 | Rear-end (Behind) | atk 在后、同向追尾 |
| (2, 1) | 180° 后 | 90° 朝右 | Sideswipe from back-right | atk 在后、横开 |
| (2, 2) | 180° 后 | 180° 朝后 | Head-on from behind | atk 在后、朝反方向 (罕见) |
| (2, 3) | 180° 后 | 270° 朝左 | Sideswipe from back-left | atk 在后、横开 |
| (3, 0) | 270° 左 | 0° 朝前 | T-bone from left | atk 在左、朝前 |
| (3, 1) | 270° 左 | 90° 朝右 | T-bone left crossing | atk 在左、横切 |
| (3, 2) | 270° 左 | 180° 朝后 | Sideswipe opposite left | atk 在左、反向侧擦 |
| (3, 3) | 270° 左 | 270° 朝左 | Sideswipe from left | atk 在左、同向侧擦 |

## 4. 与论文方法的对比

| 维度 | 论文 (4D 连续 + K-means 10) | 我们 (16 网格离散) |
|---|---|---|
| 特征空间 | 4D 连续 | 2D 离散 (4+4 bin) |
| 类别数 | 10 (自适应) | 16 (硬编码) |
| 角度对称性 | 连续 | 4 重对称 (90°) |
| 复用 cluster_scenarios.py | ✅ | ⚠️ 需重写 compute_coll_feat 输出 bin |
| 物理可撞性 | 自适应 | 假设所有 16 cell 可撞 (待实验验证) |
| Policy 输入 | N/A | bd_idx one-hot 16 dim |

## 5. 在 RL 中的应用

### 5.1 作为 novelty 信号 (备选)
```python
bd_novelty = -manhattan_dist(current_bd, nearest_archive_bd)
reward = -min_dist + w_bd * bd_novelty
```

### 5.2 作为 goal condition (备选)
```python
policy = ConditionalPolicy(obs_dim + 16_onehot, z_dim)
# 推理时遍历 16 个 goal → 16 个解
```

### 5.3 作为 QD archive key (备选)
```python
archive = dict[bd_idx] -> best_z
# 16 cells 自然对应 16 archive slots
```

## 6. 注意事项

- **小场景浪费**: NA=2 实际可撞的 BD cell 数可能 < 16 (待实验验证)
- **边界 bin 量化误差**: pos_angle=42° vs 48° 一个 cell 差, 但语义相同
- **物理不可达 cell**: 部分 cell 在当前 ego 轨迹下物理上不可达 (如 atk 在 ego 后方 0.5m 且朝反方向)

## 7. 决策状态

- [x] 16 网格方案 B 确认
- [ ] 是否需要降为 8 网格 (待实验覆盖度数据)
- [ ] 是否需要回到连续 BD (待实验结果)
