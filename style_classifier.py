from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
from pathlib import Path


# Prepare the input to the model
system_prompt = """你是一名文本情感与毒性分析专家。阅读用户给出的文本，然后判断它属于下面三类中的哪一类。
请只输出类别编号，不要解释，也不要输出其它内容。

1. Toxic      — 句子含有脏话、不雅用词、粗俗表达等毒性词。
2. Neutral    — 句子没有毒性词汇，但仍保留原有的情感强度或语气（愤怒/讽刺等）。
3. Polite     — 无任何毒性词汇，整体语气礼貌、克制，情感强度明显降低或中和。

输出格式：
Toxic/Neutral/Polite"""

def classify(sentence):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentence}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Generate outputs
    outputs = llm.generate([text], sampling_params)
    response = outputs[0].outputs[0].text
    return response

def classify_batch(sentences):
    """sentences: List[str]  ->  List[str]"""
    prompts = []
    for s in sentences:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": s}
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    # vLLM 接受 List[str]，顺序不变地返回同长度的结果
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]

if __name__ == "__main__":

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

    # Configurae the sampling parameters (for thinking mode)
    sampling_params = SamplingParams (
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=2048
    )

    # Initialize the vLLM engine
    llm = LLM(
        model="/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-style-polarity",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=8
    )

    data = json.load(open("./data/Style-datasets-idx.json", "r", encoding="utf-8"))
    keys = ["toxic", "neutral", "polite"]
    for item in tqdm(data):
        # save = {}
        # for key in keys:
        #     key_sentence = item[key]
        #     res = classify(key_sentence)
        #     save[key] = res
        # item["classify"] = save

        sents = [item[k] for k in keys]
        res_list = classify_batch(sents)
        item["classify"] = dict(zip(keys, res_list))

    output_path = Path("/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result")
    output_path.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(output_path / "Style-datasets-idx-polarity.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    print("Classification completed and saved to", output_path / "Style-datasets-idx-polarity.json")