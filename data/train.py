import json

data = json.load(open("./Style-datasets-idx.json", "r", encoding="utf-8"))

prompt = "{sent}"
system_prompt = """你是一个文明表达的审查员，你的任务是判断用户输入的中文表达是否含有毒性。判断依据如下：

含有毒性：
1. 含有轻微脏话、不雅用词、粗俗表达或比喻侮辱。
2. 使用变体脏话、拼音脏词、谐音粗话、委婉替代词（如"tm，特么，sb，woc，我超，鸟事"等），这些表达在语境中常用作粗俗表达。
3. 含有明显的歧视、仇恨、侮辱、性别攻击、动物比喻式人身攻击、侮辱性动物词或恶意中伤。

不含毒性：
1. 不含脏话或不雅用词。

**注意**：
- 如果句子只是包含强烈的情感极性，如愤怒、激动、讽刺、委屈、激昂等，只要不包含毒性表达，都不能算含有毒性。

输出格式：
有毒/无毒。
"""

def build_dataset(data):
    ds = []
    i = 0
    keys = [ "toxic", "neutral"]
    for item in data:
        for key in keys:
            if key == "toxic":
                label = "有毒"
            else:
                label = "无毒"
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

json.dump(ds, open(f"train_{len(ds)}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)