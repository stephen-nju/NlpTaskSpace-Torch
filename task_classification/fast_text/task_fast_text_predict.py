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

data=""

def filter_query(text):
    if isinstance(text,str) and text.isdigit():
        return False
    
    if isinstance(text,int) or isinstance(text,float):
        return False

    # 删除ticketID=和 tag=
    
    if "ticketID=" in text or "tag=" in text:
        return False

    return True



if __name__ == '__main__':

    classifier = fasttext.load_model("model.bin")
    p = []
    with open(data, "r", encoding="utf-8") as g:
        for line in g:
            if filter_query(line.strip())
                p.append(line.strip().split("\t"))





    output = []
    for t in tqdm(p):
        seg = jieba.lcut(" ".join(t))
        out = classifier.predict("\t".join(seg))
        output.append([out[0][0], out[1][0].astype(str), "\t".join(t)])
    with open("predict_output.txt", "w", encoding="utf-8") as g:
        for a in output:
            g.write("\t".join(a) + "\n")

