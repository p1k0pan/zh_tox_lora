# zh_tox_lora

## 日志
### 2025年5月18日
- 运行`bash lora_qwen3-32b_v3.sh`
- 权重保存在`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/lora_weights_3112_explain/{时间戳version}/{checkpoint最新的}`（需要确认最新的）
- 手动合并，修改`merge.sh`里面的`--adapters`后面的地址为权重地址，`--output_dir`应该是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier_v3`。然后运行`CUDA_VISIBLE_DEVICES=0 bash merge.sh`
- 输出的合并地址为：`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier_v3`
- 测试v3分类效果，运行：`CUDA_VISIBLE_DEVICES=0,1,2,3 python classifier_batch.py`，输出文件是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/Style-datasets-idx-classify_v3.json`

### 2025年5月17日
- 运行`bash lora_qwen3-32b_style.sh`
- 权重保存在`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/lora_weights_style_polarity/{时间戳version}/{checkpoint最新的}`（需要确认最新的）
- 手动合并，修改`merge.sh`里面的`--adapters`后面的地址为权重地址，`--output_dir`应该是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-style-polarity`。然后运行`CUDA_VISIBLE_DEVICES=0 bash merge.sh`
- 输出的合并地址为：`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-style-polarity`
- 测试情感极性分类效果，运行：`CUDA_VISIBLE_DEVICES=0,1,2,3 python style_classifier_batch.py`，输出文件是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/Style-datasets-idx-polarity.json`

### 2025年5月16日_v2
- 运行`bash lora_qwen3-32b_v2.sh`
- 权重保存在`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/lora_weights_3112/{时间戳version}/{checkpoint最新的}`（需要确认最新的）
- 手动合并，修改`merge.sh`里面的`--adapters`后面的地址为权重地址，`--output_dir`应该是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier_v2/`。然后运行`CUDA_VISIBLE_DEVICES=0 bash merge.sh`
- 输出的合并地址为：`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier_v2/`
- 测试分类效果，运行：`CUDA_VISIBLE_DEVICES=0,1,2,3 python classifier_batch.py`，输出文件是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/Style-datasets-idx-classify_v2.json`

### 2025年5月16日
- 应用全部数据进行训练
- 运行`bash lora_qwen3-32b.sh`
- 权重保存在`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/lora_weights_full_data/{时间戳}/checkpoint-{最新}`（需要确认最新的）
- 手动合并，修改`merge.sh`里面的`--adapters`后面的地址为权重地址，然后运行`CUDA_VISIBLE_DEVICES=0 bash merge.sh`
- 输出的合并地址为：`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier/`
- 测试分类效果，运行：`CUDA_VISIBLE_DEVICES=0,1,2,3 python classifier.py`，输出文件是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/Style-datasets-idx-classify.json`

### 2025年5月15日
- 使用ms-swift的lora微调qwen3-32b模型作为中文毒性分类器
- 步骤
    1. `conda env create -f environment.yml` 创建的环境名字叫`pjh_zhtox_swift`
    2. 运行8卡训练 `bash lora_qwen3-32b.sh` ，这个命令会进行训练然后进行合并权重
    3. 训练后权重合并之后保存的位置：`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier/`
