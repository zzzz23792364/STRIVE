# 部署方式

## 1. 本地 (Windows WSL2)

### 路径

```
代码: /mnt/e/wsl-22/STRIVE/
数据: /mnt/e/wsl-22/STRIVE/data/nuscenes/
```

### 环境要求

- GPU: RTX 3060 Ti, 8GB VRAM
- CUDA 11.1+
- Python 3.8
- 依赖: `pip install -r requirements.txt`

### 本地运行

```bash
cd /mnt/e/wsl-22/STRIVE
python src/test_traffic.py --config ./configs/test_traffic.cfg --test_on_val
```

> 本地 GPU 显存较小 (8GB)，仅适合小 batch 测试，完整实验建议用云服务器。

---

## 2. 云服务器 (AutoDL)

### 连接信息

| 项目 | 值 |
|------|------|
| 地址 | `region-41.seetacloud.com:31198` |
| 用户 | `root` |
| 密码 | `IzX9/D3RvFKX` |
| GPU | NVIDIA RTX 3090, 24GB VRAM |
| CPU | Intel Xeon Platinum 8255C, 96 核 |
| RAM | 375GB |
| OS | Ubuntu 18.04 (Docker) |

### SSH 连接

```bash
sshpass -p 'IzX9/D3RvFKX' ssh -p 31198 root@region-41.seetacloud.com
```

> 需要先安装 `sshpass`: `sudo apt install sshpass`

### 代码位置

```
项目: /root/autodl-tmp/STRIVE/
```

### 环境

- Python: base conda env (miniconda3 at `/root/miniconda3`)
- PyTorch: 1.9.0+cu111 (已安装)
- 激活: `export PATH="/root/miniconda3/bin:$PATH"`

---

## 3. 代码同步（通过 Git）

### 仓库

主仓库: `https://github.com/zzzz23792364/STRIVE`

### 工作流

```bash
# 本地开发
cd /mnt/e/wsl-22/STRIVE
# ...改代码...
git add <files>
git commit -m "description"
git push personal main

# 云服务器拉取
# SSH 登录后:
cd /root/autodl-tmp/STRIVE
git pull
```

### 推送到个人仓库

创建个人 fork:

```bash
cd /mnt/e/wsl-22/STRIVE
git remote add personal https://github.com/zzzz23792364/STRIVE
git push personal main
```

### 云服务器首次部署（从头搭建）

```bash
# 1. SSH 连接
sshpass -p 'IzX9/D3RvFKX' ssh -p 31198 root@region-41.seetacloud.com

# 2. 环境变量
export PATH="/root/miniconda3/bin:$PATH"

# 3. 验证 GPU
nvidia-smi

# 4. 克隆代码
cd /root/autodl-tmp
git clone https://github.com/zzzz23792364/STRIVE

# 5. 数据
# nuScenes 数据已包含在仓库中
# 如果需重新解压:
cd /root/autodl-tmp/STRIVE
unzip -o ../model_ckpt.zip -d model_ckpt/   # 模型权重
mkdir -p data/nuscenes/trainval
tar -xzf ../v1.0-trainval_meta.tgz -C data/nuscenes/trainval/
unzip -o ../nuScenes-map-expansion-v1.3.zip -d data/nuscenes/trainval/

# 6. 修复目录结构
cd data/nuscenes/trainval
for d in basemap expansion prediction; do
  [ -d "$d" ] && mv "$d" maps/
done

# 7. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
pip install "shapely<2"  # 兼容 nuscenes-devkit
```

---

## 4. 快速命令

### 推理测试

```bash
# 本地 / 云服务器均可
cd /root/autodl-tmp/STRIVE
python src/test_traffic.py --config ./configs/test_traffic.cfg --test_on_val
```

### Phase 1 RL 训练

```bash
cd /root/autodl-tmp/STRIVE
python src/rl/train_phase1.py \
  --config ./configs/phase1_rl.cfg \
  --out ./out/phase1_result \
  --rl_algo reinforce \
  --warmup_steps 50 \
  --num_episodes 1000
```

### 梯度基线对比

```bash
cd /root/autodl-tmp/STRIVE
python src/rl/train_phase1.py \
  --config ./configs/phase1_rl.cfg \
  --out ./out/phase1_with_baseline \
  --rl_algo ppo \
  --num_episodes 1000 \
  --compare_baseline true
```

### 场景索引

```bash
# 修改场景: 添加 --scene_idx N (0 为第一个场景)
python src/rl/train_phase1.py --config ./configs/phase1_rl.cfg --scene_idx 5 ...
```

---

## 5. 目录结构

```
STRIVE/
├── src/
│   ├── rl/                 # RL 基础设施 (Phase 0)
│   │   ├── policy.py       #   策略网络
│   │   ├── reward.py       #   奖励函数
│   │   ├── reinforce.py    #   REINFORCE 算法
│   │   ├── ppo.py          #   PPO 算法
│   │   └── train_phase1.py #   Phase 1 训练入口
│   ├── models/             # 原始 TrafficModel (不动)
│   ├── losses/             # 原始损失函数 (不动)
│   ├── datasets/           # 原始数据集 (不动)
│   └── utils/              # 原始工具函数 (不动)
├── configs/
│   └── phase1_rl.cfg       # RL 训练配置
├── data/nuscenes/trainval/ # nuScenes 数据
├── model_ckpt/             # 预训练权重
├── docs/                   # 文档
└── requirements.txt        # 依赖
```
