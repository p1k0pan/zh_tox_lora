import json

data = json.load(open("./Style-datasets-idx.json", "r", encoding="utf-8"))

prompt = "{sent}"
system_prompt = """你是一名文本情感与毒性分析专家。阅读用户给出的文本，然后判断它属于下面三类中的哪一类。
请只输出类别编号，不要解释，也不要输出其它内容。

1. Toxic      — 句子含有脏话、不雅用词、粗俗表达等毒性词。
2. Neutral    — 句子没有毒性词汇，但仍保留原有的情感强度或语气（愤怒/讽刺等）。
3. Polite     — 无任何毒性词汇，整体语气礼貌、克制，情感强度明显降低或中和。

输出格式：
Toxic/Neutral/Polite
"""

def build_dataset(data):
    ds = []
    i = 0
    # keys = [ "toxic", "neutral", "polite"]
    keys = [ "toxic", "neutral","neutral", "polite"]
    for item in data:
        for key in keys:
            if key == "toxic":
                label = "Toxic"
            elif key == "neutral":
                label = "Neutral"
            elif key == "polite":
                label = "Polite"
            payload = {
                "idx": i,
                "messages":[],
            }
            payload["messages"].append({
                "role": "system", 
                "content": system_prompt
            })
            payload["messages"].append({
                "role": "user", 
                "content": item[key]
            })
            payload["messages"].append({
                "role": "assistant", 
                "content": label
            })
            ds.append(payload)
            i += 1

    return ds, i

ds,i = build_dataset(data)
print(len(ds))

# json.dump(ds, open(f"train_pol_{len(ds)}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
json.dump(ds, open(f"train_pol_ratio_212.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)