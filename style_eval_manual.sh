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

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python style_classifier_model_batch.py --folder manual_eval