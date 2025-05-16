# zh_tox_lora

## 日志
### 2025年5月16日
- 应用全部数据进行训练
- 运行`bash lora_qwen3-32b.sh`
- 权重保存在`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/lora_weights_full_data/{时间戳}`（确认最新的）
- 手动合并，修改`merge.sh`里面的`--ckpt_dir`后面的地址为权重地址，然后运行`bash merge.sh`
- 输出的合并地址为：`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier/`
### 2025年5月15日
- 使用ms-swift的lora微调qwen3-32b模型作为中文毒性分类器
- 步骤
    1. `conda env create -f environment.yml` 创建的环境名字叫`pjh_zhtox_swift`
    2. 运行8卡训练 `bash lora_qwen3-32b.sh` ，这个命令会进行训练然后进行合并权重
    3. 训练后权重合并之后保存的位置：`/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier/`
