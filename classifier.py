from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
from pathlib import Path


# Prepare the input to the model
system_prompt = """你是一个文明表达的审查员，你的任务是判断用户输入的中文表达是否含有毒性。判断依据如下：

含有毒性：
1. 含有轻微脏话、不雅用词、粗俗表达。
2. 使用变体脏话、拼音脏词、谐音粗话、委婉替代词（如"tm，特么，sb，woc，我超，鸟事"等），这些表达在语境中常用作粗俗表达。
3. 含有明显的歧视、仇恨、侮辱、性别攻击、动物比喻式人身攻击、侮辱性动物词或恶意中伤。

不含毒性：
1. 不含脏话或不雅用词。

**注意**：
- 如果句子只是包含强烈的情感极性，如愤怒、激动、讽刺、委屈、激昂等，只要不包含毒性表达，都不能算含有毒性。

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

if __name__ == "__main__":

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

    # Configurae the sampling parameters (for thinking mode)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=512)

    # Initialize the vLLM engine
    llm = LLM(model="/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/merged_lora/qwen3-32b-tox-classifier", gpu_memory_utilization=0.95)

    data = json.load(open("./data/Style-datasets-idx.json", "r", encoding="utf-8"))
    keys = ["toxic", "neutral", "polite"]
    for item in tqdm(data):
        save = {}
        for key in keys:
            key_sentence = item[key]
            res = classify(key_sentence)
            save[key] = res
        item["classify"] = save

    output_path = Path("/mnt/workspace/xintong/pjh/All_result/zh_tox_lora/class_result")
    output_path.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(output_path / "Style-datasets-idx-classify.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    print("Classification completed and saved to", output_path / "Style-datasets-idx-classify.json")