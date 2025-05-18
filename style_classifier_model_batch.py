from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
from pathlib import Path
import argparse

# Prepare the input to the model
system_prompt = """你是一名文本情感与毒性分析专家。阅读用户给出的文本，然后判断它属于下面三类中的哪一类。
请只输出类别编号，不要解释，也不要输出其它内容。

1. Toxic      — 句子含有脏话、不雅用词、粗俗表达等毒性词。
2. Neutral    — 句子没有毒性词汇，但仍保留原有的情感强度或语气（愤怒/讽刺等）。
3. Polite     — 无任何毒性词汇，整体语气礼貌、克制，情感强度明显降低或中和。

输出格式：
Toxic/Neutral/Polite"""


def flush(prompts, meta):
    """把累计的一批 prompts 送进 vLLM，并把结果写回 data。"""
    if not prompts:
        return
    outputs = llm.generate(prompts, sampling_params)
    preds   = [o.outputs[0].text.strip() for o in outputs]

    for (idx, k), p in zip(meta, preds):
        data[idx]["style_cls"] =p

def process_one_file():
    prompts   = []
    meta      = []

    BATCH_SZ = 128
    for idx, item in enumerate(tqdm(data, desc="batch infer")):
        # 为同一个样本的 3 句话都创建 prompt
        k = "rewritten"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": item[k]}
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        )
        meta.append((idx, k))

        # 满批就推理一次
        if len(prompts) >= BATCH_SZ:
            flush(prompts, meta)
            prompts, meta = [], []

    # 处理残余 < BATCH_SZ 的部分
    flush(prompts, meta)


    json.dump(data, open(output_path / (file.stem+"_cls.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    print("Classification completed and saved to", output_path / (file.stem+"_cls.json"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--folder',
        type=str,
        default="",
        help=f'Path to save the classification results. '
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    folder = args.folder

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

    # Configurae the sampling parameters (for thinking mode)
    sampling_params = SamplingParams (
        temperature=0,
        top_p=0.9,
        top_k=20,
        max_tokens=128
    )

    # Initialize the vLLM engine
    llm = LLM(
        model="/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-style-polarity_ratio121",
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        tensor_parallel_size=4
    )

    open_source = Path(f"./{folder}")
    output_path = Path(f"/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result/{folder}_style_results")
    output_path.mkdir(parents=True, exist_ok=True)
    for file in open_source.rglob("*.json"):
        data = json.load(open(file, "r", encoding="utf-8"))
        print(f"Processing {file}")
        process_one_file()