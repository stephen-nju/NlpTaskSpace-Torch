# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: predict_fast_text.py
@time: 2022/7/13 15:08
"""
import fasttext
import jieba
from tqdm import tqdm

if __name__ == '__main__':

    classifier = fasttext.load_model("model.bin")
    p = []
    with open("data/predict.txt", "r", encoding="utf-8") as g:
        for line in g:
            p.append(line.strip().split("\t"))

    output = []
    for t in tqdm(p):
        seg = jieba.lcut(" ".join(t))
        out = classifier.predict("\t".join(seg))
        output.append([out[0][0], out[1][0].astype(str), "\t".join(t)])
    with open("predict_output.txt", "w", encoding="utf-8") as g:
        for a in output:
            g.write("\t".join(a) + "\n")

