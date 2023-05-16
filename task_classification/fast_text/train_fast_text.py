# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: train_fast_text.py
@time: 2022/7/12 16:30
"""
import fasttext
import jieba
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def process_text():
    df = pd.read_csv("task_classification/fast_text/data/param_data_labeled_bingxi.txt", sep="\t", header=None)
    df = df[~(df[2].isna())]
    df[2] = df[2].astype(str)
    # sample_data = []
    # for name, group in df.groupby("参数类型"):
    #     if len(group) > 500:
    #         sample_data.append(group.loc[:, :500])
    #     else:
    #         sample_data.append(group)
    # sample_data = pd.concat(sample_data, axis=0)
    df_pos = df[df[0] == 1]
    df_neg = df[df[0] == -1]
    df_neu = df[df[0] == 0]
    # 上采样
    d_p = []
    d_n = []
    rate = int(len(df_neg) / len(df_pos))
    for i in range(rate):
        d_p.append(df_pos)

    rate2 = int(len(df_neg) / len(df_neu))

    for i in range(rate2):
        d_n.append(df_neu)

    df_neu = pd.concat(d_n, axis=0)
    df_pos = pd.concat(d_p, axis=0)
    df = pd.concat([df_pos, df_neg, df_neu], axis=0)

    train_data, test_data = train_test_split(df, test_size=0.3)
    with open("task_classification/fast_text/data/train_data.txt", "w", encoding="utf-8") as g:
        for index, row in train_data.iterrows():
            segs = jieba.lcut(" ".join([row[1], row[2]]))
            g.write(f"__label__{row[0]}" + " " + " ".join(segs) + "\n")

    with open("task_classification/fast_text/data/test_data.txt", "w", encoding="utf-8") as g:
        for index, row in train_data.iterrows():
            segs = jieba.lcut(" ".join([row[1], row[2]]))
            g.write(f"__label__{row[0]}" + " " + " ".join(segs) + "\n")


if __name__ == '__main__':
    process_text()
    classifier = fasttext.train_supervised('task_classification/fast_text/data/train_data.txt', label='__label__',
                                           wordNgrams=2, epoch=30, lr=0.1,
                                           dim=100)

    test_result = classifier.test('task_classification/fast_text/data/test_data.txt')
    print('test_precision:', test_result[1])
    print('test_recall:', test_result[2])
    print('Number of test examples:', test_result[0])
    classifier.save_model("task_classification/fast_text/model.bin")
