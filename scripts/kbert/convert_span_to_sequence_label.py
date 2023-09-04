# -----------coding:utf-8-----#
"""
将qa格式的训练文件转化成sequence label格式
"""

import argparse
import json
import os
from schema import Schema, And, Use, Optional
from typing import List, Tuple, Dict
import uuid
from tqdm import tqdm

qas_schema = Schema(
    {
        "context": And(str, len),
        "qas": [
            {
                "id": Optional(str, int),
                "question": And(str, len),
                "answers": [
                    {
                        "text": And(str, len),
                        "answer_start": int,
                    }
                ],
            }
        ],
    }
)

input_schema = Schema(
    {
        "name": And(str, len),
        "desc": And(str, len),
        "num": And(Use(int)),
        "data": [qas_schema],
    }
)

output_schema = Schema([str, str])


def read_input_data(path):
    with open(path, "r", encoding="utf-8") as g:
        input_data = json.loads(g.read())
        # 数据量太大，太耗时，取消验证
        # input_schema.validate(input_data)
    return input_data


def write_output_data(output_data, output_path):
    with open(output_path, "w", encoding="utf-8") as g:
        for a in output_data:
            output_data = output_schema.validate(a)
            g.write("\t".join(a) + "\n")


def ruler(context, labels):
    # 规定怎样进行转换，数据转换的核心逻辑
    output = []
    c = list(context)
    # 第一种情况 len(labels)==0,该条数据不包含实体
    if len(labels) == 0:
        l = ["O"] * len(c)

    else:
        # 如果存在实体，需要保证实体不能存在重叠，交叉等情况
        index_map = {index: "O" for index in range(len(c))}
        for label in labels:
            start = label[0]
            end = label[1]
            type_name = label[2]
            assert start < end
            # 判断start,end 区间是否已经有标签
            flag = True
            for i in range(start, end):
                if index_map[i] != "O":
                    flag = False
                    break
            # 重叠的实体直接移除
            if flag:
                index_map[start] = f"B-{type_name}"
                for j in range(start + 1, end):
                    index_map[j] = f"I-{type_name}"
            else:
                # 重叠实体直接跳过
                continue

        l = [index_map[k] for k in range(len(c))]

    return [" ".join(c), " ".join(l)]


def convert(input_data, convert_ruler):
    """
    转化为序列标注的形式，目前的方式是将英文按照char形式进行分割
    #TODO 考虑转化为子词形式
    """
    output_data = []
    data = input_data["data"]

    for d in tqdm(data):
        qas = d["qas"]
        context = d["context"]
        labels = []
        for qa in qas:
            question = qa["question"]
            answers = qa["answers"]
            if question == "标题中产品提及有哪些":
                type_name = "HC"
            elif question == "标题中品牌提及有哪些":
                type_name = "HP"
            elif question == "标题中系列型号提及有哪些":
                type_name = "XL"

            if len(answers) > 0:
                start = answers[0]["answer_start"]
                end = start + len(answers[0]["text"])
                labels.append([start, end, type_name, answers[0]["text"]])
        o = convert_ruler(context, labels)
        output_data.append(o)

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chat gpt requests")
    parser.add_argument(
        "--input_data_path", type=str, help="input data path,data format is json"
    )
    parser.add_argument(
        "--output_data_path", type=str, default=".", help="output data path"
    )

    args = parser.parse_args()
    i = read_input_data(args.input_data_path)
    
    o = convert(i, ruler)
    write_output_data(o, args.output_data_path)
