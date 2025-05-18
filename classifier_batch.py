from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
from pathlib import Path
import sys


# Prepare the input to the model
system_prompt = """你是一个文明表达的审查员，你的任务是判断用户输入的中文表达是否含有毒性。判断依据如下：

毒性：
1. 含有轻微脏话、不雅用词、粗俗表达。
2. 使用变体脏话、拼音脏词、谐音粗话、委婉替代词（如"tm，特么，sb，woc，我超，鸟事"等），这些表达在语境中常用作粗俗表达。
3. 含有明显的歧视、仇恨、侮辱、性别攻击、动物比喻式人身攻击、侮辱性动物词或恶意中伤。

非毒性：
1. 不含脏话或不雅用词。

**注意**：
- 如果句子只是包含强烈的情感极性和表达意图，如愤怒、激动、讽刺、委屈、激昂等，只要不包含毒性表达，都不能算含有毒性。

输出格式：
有毒/无毒。
"""

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

def flush(prompts, meta):
    """把累计的一批 prompts 送进 vLLM，并把结果写回 data。"""
    if not prompts:
        return
    outputs = llm.generate(prompts, sampling_params)
    preds   = [o.outputs[0].text.strip() for o in outputs]

    for (idx, k), p in zip(meta, preds):
        data[idx].setdefault("classify", {})[k] = p

if __name__ == "__main__":

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
        model="/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier",
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        tensor_parallel_size=4
    )

    data = json.load(open("./data/Style-datasets-idx.json", "r", encoding="utf-8"))
    keys = ["toxic", "neutral", "polite"]

    prompts   = []
    meta      = []

    BATCH_SZ = 128
    for idx, item in enumerate(tqdm(data, desc="batch infer")):
        # 为同一个样本的 3 句话都创建 prompt
        for k in keys:
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


    output_path = Path("/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result")
    output_path.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(output_path / "Style-datasets-idx-classify_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    print("Classification completed and saved to", output_path / "Style-datasets-idx-classify_v3.json")