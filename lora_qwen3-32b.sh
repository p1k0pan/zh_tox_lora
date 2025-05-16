#!/bin/bash

# 激活 Conda 环境
echo "🔄 正在切换到 Conda 环境 pjh_zhtox_swift..."
eval "$(conda shell.bash hook)"
conda activate pjh_zhtox_swift

# 检查 conda 环境是否激活成功
if [[ "$CONDA_DEFAULT_ENV" == "pjh_zhtox_swift" ]]; then
  echo "✅ Conda 环境 pjh_zhtox_swift 已成功激活！"
else
  echo "❌ Conda 环境激活失败！当前环境为：$CONDA_DEFAULT_ENV"
  exit 1
fi

WANDB_DIR=/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/wandb \
WANDB_API_KEY=1526cd13c8d1f8c8529ea57f23d553b20b03451c \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model Qwen/Qwen3-32B \
    --train_type lora \
    --dataset ./data/train_full_8148.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir /mnt/workspace/xintong/pjh/All_result/zh_tox_lora/lora_weights_full_data/ \
    --warmup_ratio 0.05 \
    --lora_dropout 0.05 \
    --deepspeed zero3 \
    --report_to wandb \
    --dataloader_num_workers 4


# # 如果训练成功才导出
# if [[ $? -eq 0 ]]; then
#   echo "✅ 训练完成，准备合并 LoRA 权重..."
#   swift export \
#     --ckpt_dir /mnt/workspace/xintong/pjh/All_result/zh_tox_lora/lora_weights/ \
#     --merge_lora true \
#     --output_dir /mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier/
# else
#   echo "❌ 训练失败，跳过导出。"
#   exit 1
# fi