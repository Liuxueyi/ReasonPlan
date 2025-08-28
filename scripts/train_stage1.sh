#!/bin/bash

# 运行名称和输出目录配置
MID_RUN_NAME="stage1"
output_dir="checkpoints/${MID_RUN_NAME}/"
output_log="${output_dir}/training_output.log"

# 模型相关配置
pretrained_ckpt="llava-next-interleave-qwen-0.5b"
VISION_TOWER="google/siglip-so400m-patch14-384"

# 训练参数配置
train_stage="first_align"
batch_size=2
epoch=1
learning_rate=5e-5
nsp_enable=False
only_front=False
im_weight=1.0
reasoning_enable=False
port=21300
mm_patch_merge_type="flat"  # spatial_unpad

# 创建输出目录和日志文件
mkdir -p $output_dir
touch $output_log

# 打印配置信息
echo "train_stage: ${train_stage}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
echo "batch_size: ${batch_size}"
echo "epoch: ${epoch}"
echo "pretrained_ckpt: ${pretrained_ckpt}"
echo "learning_rate: ${learning_rate}"
echo "nsp_enable: ${nsp_enable}"
echo "only_front: ${only_front}"
echo "im_weight: ${im_weight}"
echo "reasoning_enable: ${reasoning_enable}"
echo "mm_patch_merge_type: ${mm_patch_merge_type}"
echo "vision_tower: ${VISION_TOWER}"

# 设置CUDA路径
CUDA_HOME=/usr/local/cuda-11.8

# 启动训练
deepspeed --master_port $port --include localhost:0,1,2,3 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $pretrained_ckpt \
    --model_max_length 32768 \
    --vision_tower $VISION_TOWER \
    --num_train_epochs $epoch \
    --mm_patch_merge_type $mm_patch_merge_type \
    --per_device_train_batch_size $batch_size \
    --dataloader_num_workers 2 \
    --learning_rate $learning_rate \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --warmup_ratio 0.03 \
    --freeze_vision_tower True \
    --run_name $MID_RUN_NAME \
    --output_dir $output_dir \
    --is_train True \
    --train_stage $train_stage \
    --nsp_enable $nsp_enable \
    --only_front $only_front \
    --im_weight $im_weight \
    --reasoning_enable $reasoning_enable \
    --text_data_path data/QA_data/train_final.pkl &> $output_log