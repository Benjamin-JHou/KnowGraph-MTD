#!/bin/bash

# CPU训练脚本示例
# 这个脚本展示如何在CPU上运行KPGT训练

echo "开始CPU训练..."

# 确保环境变量不包含CUDA设置
unset CUDA_VISIBLE_DEVICES
unset LOCAL_RANK

# 运行训练（使用--use_cpu标志）
python train_kpgt.py \
    --use_cpu \
    --save_path ../models/pretrained/base_cpu/ \
    --n_threads 4 \
    --config base \
    --n_steps 10000 \
    --data_path ../datasets/chembl29/ \
    --seed 42

echo "CPU训练完成！"