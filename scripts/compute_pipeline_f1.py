# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: compute_pipeline_f1.py
@time: 2021/5/27 10:52
"""
from collections import defaultdict
import json
from tqdm import tqdm
import re

pat = re.compile("(?<=商品标题中与).*(?=关联的主题关键词有哪些)")

title_relation_map = defaultdict()
with open("/home/nlpbigdata/net_disk_project/zhubin/gds_associate_project/data_repository/pre_keyword_extract.txt", "r",
          encoding="utf-8") as g:
    for line in g:
        d = json.loads(line)
        cs = re.findall(pat, d["query"])
        category = cs[0]
        combine_set = set()
        title = d["title"]
        for words in d["entities"]:
            key = "".join(words.split(" "))
            combine_set.add((category, key))
        title_relation_map.setdefault(title, set()).update(combine_set)

title_raw_map = defaultdict()
with open(
        "/home/nlpbigdata/net_disk_project/zhubin/gds_associate_project/data_repository/joint_extract_data/title_extract/test.txt",
        "r", encoding="utf-8") as g:
    for line in g:
        data = json.loads(line.strip())
        title = data["title"]
        match = data["match"]
        match_set = set()
        for d in match:
            c = d["category"]["text"]
            k = d["keyword"]["text"]
            match_set.add((c, k))
        title_raw_map.setdefault(title, set()).update(match_set)

X, Y, Z = 1e-10, 1e-10, 1e-10
no_result_num = 0
for title, target in tqdm(title_raw_map.items()):
    if title not in title_relation_map:
        no_result_num += 1
    else:
        preds = title_relation_map[title]
        X += len(preds & target)
        Y += len(preds)
        Z += len(target)

f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
print(f"no match num {no_result_num}")
print(f1, precision, recall)
