# zh_tox_lora

## 日志
### 2025年5月18日
#### 对所有结果进行风格分类
- Terminal 1运行：`bash style_eval_open.sh`，Terminal 2运行：`bash style_eval_close.sh`
- 结果保存在`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/open-source_style_results`以及`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/closed-source_style_results`

#### 对所有结果进行毒性分类
- Terminal 1运行：`bash tox_eval_open.sh`，Terminal 2运行：`bash tox_eval_close.sh`
- 结果保存在`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/open-source_results`以及`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/closed-source_results`

### 2025年5月18日
#### 重新运行第一版情感极性分类器效果
- 测试分类效果，运行：`CUDA_VISIBLE_DEVICES=0,1,2,3 python style_classifier_batch.py --model /mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-style-polarity --save_file Style-datasets-idx-polarity.json`，输出文件是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/Style-datasets-idx-polarity.json`
- 测试错误率，运行`python pol_eval.python --file /mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/Style-datasets-idx-polarity.json`，记录一下错误率。

#### Alter1: 如果上一个结果还是出现neutral错误率很高，对上一个版本继续训练
- 单独用neutral的数据继续微调3个epoch，lr: 5e-6。运行`bash lora_qwen3-32b_style_continue.sh`
- 权重保存在`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/lora_weights_style_polarity_continue/{时间戳version}/{checkpoint最新的}`（需要确认最新的）
- 手动合并，修改`merge.sh`里面的`--adapters`后面的地址为权重地址，`--output_dir`应该是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-style-polarity_continue`。然后运行`CUDA_VISIBLE_DEVICES=0 bash merge.sh`
- 测试分类效果，运行：`CUDA_VISIBLE_DEVICES=0,1,2,3 python style_classifier_batch.py --model /mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-style-polarity_continue --save_file Style-datasets-idx-polarity_continue.json`，输出文件是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/Style-datasets-idx-polarity_continue.json`
- 测试错误率，运行`python pol_eval.python --file /mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/Style-datasets-idx-polarity_continue.json`，记录一下错误率。

#### Alter2: 如果上一个结果还是出现neutral错误率很高，提高中性的数据配比
- 数据配比tox:1500, neutral:3000, polite:1500。运行`bash lora_qwen3-32b_style_ratio121.sh`
- 权重保存在`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/lora_weights_style_polarity_ratio121/{时间戳version}/{checkpoint最新的}`（需要确认最新的）
- 手动合并，修改`merge.sh`里面的`--adapters`后面的地址为权重地址，`--output_dir`应该是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-style-polarity_ratio121`。然后运行`CUDA_VISIBLE_DEVICES=0 bash merge.sh`
- 测试分类效果，运行：`CUDA_VISIBLE_DEVICES=0,1,2,3 python style_classifier_batch.py --model /mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-style-polarity_ratio121 --save_file Style-datasets-idx-polarity_ratio121.json`，输出文件是`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/Style-datasets-idx-polarity_ratio121.json`
- 测试错误率，运行`python pol_eval.python --file /mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/Style-datasets-idx-polarity_ratio121.json`，记录一下错误率。

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
