# 远程服务器部署记录

## 服务器信息

| 项目 | 值 |
|------|------|
| 地址 | `region-41.seetacloud.com:31198` |
| 用户 | `root` |
| OS | Ubuntu 18.04.5 LTS (Docker 容器) |
| GPU | NVIDIA GeForce RTX 3090, 24GB VRAM |
| CPU | Intel Xeon Platinum 8255C, 96 核 |
| RAM | 375GB |
| 工作盘 | `/root/autodl-tmp/` 50GB NVMe |
| 存储盘 | `/root/autodl-pub/` 10TB (网络挂载, 5.7TB 可用) |

## 环境

- Python: 3.8.10 (miniconda3 base env at `/root/miniconda3`)
- PyTorch: 1.9.0+cu111
- CUDA Driver: 580.76.05 (CUDA 13.0)
- CUDA Toolkit: 11.1 (nvcc)

### 已安装依赖

| 包 | 版本 |
|------|------|
| torch | 1.9.0+cu111 |
| torchvision | 0.10.0+cu111 |
| numpy | 1.21.2 |
| matplotlib | 3.4.3 |
| PyYAML | 5.4.1 |
| ConfigArgParse | 1.5 |
| wandb | 0.10.32 |
| torch-scatter | 2.0.7 |
| torch-sparse | 0.6.10 |
| torch-geometric | 1.7.1 |
| nuscenes-devkit | 1.1.5 |
| Shapely | 1.8.5.post1 (降级, 2.x 与 nuscenes-devkit 不兼容) |

## 部署步骤

### 1. 传输压缩包

从本地 `E:\wsl-22\STRIVE\` 上传：

```
model_ckpt.zip              (26MB)
v1.0-trainval_meta.tgz      (441MB)
v1.0-test_meta.tgz          (68MB)
nuScenes-map-expansion-v1.3.zip (381MB)
```

### 2. 传输代码

由于服务器 GitHub 连接慢，改用 `tar` 管道传输本地代码：

```bash
tar cz --exclude={.git,data,model_ckpt*,*.zip,*.tgz,*.pth,__pycache__,...} \
  -C /mnt/e/wsl-22/STRIVE -f - . | ssh ... tar xzf - -C /root/autodl-tmp/STRIVE/
```

### 3. 解压数据

```bash
cd /root/autodl-tmp/STRIVE
unzip -o ../model_ckpt.zip -d model_ckpt/
mkdir -p data/nuscenes/trainval
tar -xzf ../v1.0-trainval_meta.tgz -C data/nuscenes/trainval/
tar -xzf ../v1.0-test_meta.tgz -C data/nuscenes/trainval/
unzip -o ../nuScenes-map-expansion-v1.3.zip -d data/nuscenes/trainval/
```

### 4. 目录结构

```
/root/autodl-tmp/STRIVE/
├── src/                    # 源代码
├── configs/                # 配置文件
├── model_ckpt/
│   ├── traffic_model.pth   # 预训练权重 (car+truck)
│   └── traffic_model_all_cats.pth  # 预训练权重 (全类别)
├── data/nuscenes/trainval/
│   ├── v1.0-trainval/      # nuScenes metadata JSONs
│   ├── v1.0-test/          # test metadata JSONs
│   └── maps/
│       ├── basemap/        # 底图 PNG
│       ├── expansion/      # 地图扩展 JSON
│       └── prediction/     # 预测场景 JSON
└── requirements.txt
```

## 推理验证

### 命令

```bash
cd /root/autodl-tmp/STRIVE
python src/test_traffic.py --config ./configs/test_traffic.cfg --test_on_val
```

### 结果

- **CUDA**: 可用 (`cuda:0`), RTX 3090
- **数据集**: 200 scenes, 4739 subsequences (val 分片)
- **模型**: `traffic_model.pth` 加载成功, 130 万参数
- **每步指标**: `recon_loss`, `kl_loss`, `pos_err`, `ang_err`, `pos_minADE`, `pos_minFDE`, `sample_coll_freq_map`, `sample_coll_freq_veh`
- **可视化**: 保存到 `out/test_traffic_out/viz_sample_multi/`

### 注意

- 完整跑完 4739 条约需数小时 (每步 2-5s)
- 当前只跑了 val 分片 (200 scenes)
- 如果需要完整 evaluation，去掉 `--test_on_val` 或调整 `use_challenge_splits`

## RL 实验准备

参考 `docs/rl_approach_plan.md`:

### 当前状态
- [x] 远程服务器部署完成
- [x] 数据准备就绪
- [x] 依赖安装完成
- [x] 推理验证通过
- [ ] Phase 0: 基础设施 (src/rl/ 目录, Policy 网络, Reward 封装)
- [ ] Phase 1: 单场景单解 RL
- [ ] Phase 2-4: 后续阶段

### SSH 连接

```bash
sshpass -p 'IzX9/D3RvFKX' ssh -p 31198 root@region-41.seetacloud.com
# 或手动输入密码: IzX9/D3RvFKX
```

激活环境后进入项目目录即可工作：

```bash
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/STRIVE
```
