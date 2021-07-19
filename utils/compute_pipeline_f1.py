# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: compute_pipeline_f1.py
@time: 2021/5/24 20:00
"""
from collections import defaultdict
import json
from tqdm import tqdm

title_relation_map = defaultdict()
with open("", "r", encoding="utf-8") as g:
    for line in g:
        title, category, keywords = line.strip().split("\t")
        combine_set = set()
        for words in json.loads(keywords):
            assert len(words) == 4
            key = words[0]
            combine_set.add((category, key))
        title_relation_map.setdefault(title, set()).union(combine_set)

title_raw_map = defaultdict()
with open("", "r", encoding="utf-8") as g:
    for line in g:
        data = json.loads(line.strip())
        title = data["title"]
        match = data["match"]
        match_set = set()
        for d in match:
            c = d["category"]
            k = d["keywords"]
            match_set.add((c, k))
        title_raw_map.setdefault(title, set()).union(match_set)

X, Y, Z = 1e-10, 1e-10, 1e-10
no_result_num = 0
for title in tqdm(title_raw_map.items()):
    if title not in title_relation_map:
        no_result_num += 1
    preds = title_relation_map[title]
    target = title_raw_map[title]
    X += len(preds & target)
    Y += len(preds)
    Z += len(target)

f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
print(f"no match num {no_result_num}")
print(f1, precision, recall)
