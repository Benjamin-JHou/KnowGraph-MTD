# KPGT CPU版本使用指南

本文档说明如何在CPU上运行KPGT项目。


## 临时文档

```py
python finetune.py --config base --model_path ../pretrained/base/base.pth  --dataset bace --data_path ../datasets/ --dataset_type classification --metric cls-all --split scaffold-0 --weight_decay 0 --dropout 0 --lr 3e-5 --batch_size 10
```


## 环境配置

### 1. 创建CPU版本环境

```bash
# 使用CPU版本的环境文件
conda env create -f environment_cpu.yml

# 激活环境
conda activate KPGT_CPU
```

### 2. 安装transformers（如果需要微调）

```bash
pip install transformers
```

## 训练方式

### 1. 预训练（CPU版本）

```bash
cd scripts

# 方法1: 使用提供的脚本
chmod +x train_cpu.sh
./train_cpu.sh

# 方法2: 直接运行
python train_kpgt.py \
    --use_cpu \
    --save_path ../models/pretrained/base_cpu/ \
    --n_threads 4 \
    --config base \
    --n_steps 10000 \
    --data_path ../datasets/chembl29/ \
    --seed 42
```

### 2. 微调（CPU版本）

```bash
python finetune.py \
    --use_cpu \
    --model_path ../models/pretrained/base_cpu/model.pth \
    --dataset bace \
    --data_path ../datasets/ \
    --dataset_type classification \
    --metric rocauc \
    --split scaffold-0 \
    --n_epochs 50 \
    --batch_size 16 \
    --lr 1e-4
```

## 主要修改说明

1. **train_kpgt.py**:
   - 添加了 `--use_cpu` 参数
   - 自动检测是否有GPU可用
   - 禁用分布式训练（CPU模式）
   - 移除了CUDA特定的代码

2. **utils.py**:
   - 在 `set_random_seed()` 函数中添加了CUDA可用性检查
   - 只有在CUDA可用时才设置CUDA相关的随机种子

3. **finetune.py**:
   - 添加了 `--use_cpu` 参数
   - 强制使用CPU进行微调

4. **environment_cpu.yml**:
   - 移除了CUDA相关的包
   - 使用CPU版本的PyTorch和DGL

## 性能注意事项

- CPU训练会比GPU训练慢很多
- 建议减少batch_size以适应CPU内存限制
- 可以调整 `--n_threads` 参数来优化CPU使用
- 对于大规模训练，建议使用较少的训练步数或epoch

## 常见问题

1. **内存不足**: 减少batch_size
2. **训练太慢**: 减少训练步数，使用更多CPU线程
3. **依赖问题**: 确保使用了正确的environment_cpu.yml文件

## 示例配置

CPU训练推荐配置：
```bash
--batch_size 16        # 较小的batch size
--n_threads 4          # 根据CPU核心数调整
--n_steps 5000         # 较少的训练步数
```