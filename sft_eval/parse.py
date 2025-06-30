import json
import re

data = json.load(open("/mnt/data/users/liamding/data/sft_zh_tox/eval/qwen3_r1/test_result_v2.json", "r", encoding="utf-8"))

for item in data:
    output = item["output"]
    think_content = re.search(r"<think>(.*?)</think>", output, re.DOTALL)
    answer_content = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
    if think_content is None or answer_content is None:
        print(f"Error in item {item['idx']}: think or answer content not found")
        think_content = output
        answer_content = item["neutral"]
    else:
        think_content = think_content.group(1).strip()
        answer_content = answer_content.group(1).strip()
    item["output"] = answer_content
    item["reasoning"] = think_content

json.dump(data, open("/mnt/data/users/liamding/data/sft_zh_tox/eval/qwen3_r1/test_result_v2_parsed.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)