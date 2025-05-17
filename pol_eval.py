import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--file', 
    type=str, 
    default="", 
    help=f'Path to the JSON file to evaluate. '
)
args = parser.parse_args()

data = json.load(open(args.file))

tox_num = []
non_tox_num = []
error_non_tox_num = []
polite_num = []
error_polite_num = []
all_dataset = {}
dataset_cls = {}
for item in data:
    for k,v in item["classify"].items():
        v = v.strip()
        v = v.replace("。", "")
        v = v.replace("\n", "")
        if v != "Toxic" and v != "Neutral" and v != "Polite":
            print(item['idx'],v)
        if k == "toxic":
            if v == "Toxic":
                tox_num.append(item)
        elif k == "neutral":
            if v == "Neutral":
                non_tox_num.append(item)
            else:
                error_non_tox_num.append(item)
        elif k == "polite":
            if v == "Polite":
                polite_num.append(item)
            
print(len(tox_num), len(non_tox_num), len(polite_num))
# 计算错误率
tox_err = round(1 - len(tox_num) / len(data), 2)
non_tox_err = round(1 - len(non_tox_num) / len(data), 2)
polite_err = round(1 - len(polite_num) / len(data), 2)
print("tox错误率:", tox_err, "\nneutral错误率:", non_tox_err, "\npolite错误率:", polite_err)

tx = 0
po = 0
for item in error_non_tox_num:
    if item["classify"]["neutral"] == "Toxic":
        tx += 1
    elif item["classify"]["neutral"] == "Polite":
        po += 1

print("错误样本中，neutral为toxic的样本数:", tx, "\n错误样本中，neutral为polite的样本数:", po)
