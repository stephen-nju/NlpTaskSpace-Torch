# ----conding:utf-8----------
"""
将qa形式的数据转化为huggingface格式的datasets，用于微调百川模型命名实体识别

"""

import jsonlines
import argparse
import json
import os
from schema import Schema, And, Use, Optional
from typing import List, Tuple, Dict
import uuid
from tqdm import tqdm
from collections import defaultdict

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

output_schema=Schema({
    "context":And(str,len),
    "ner":Schema({
        Optional("品类"):Schema([str]),
        Optional("品牌"):Schema([str]),
        Optional("系列型号"):Schema([str]),
        })
    })

def read_input_data(path):
    with open(path, "r", encoding="utf-8") as g:
        input_data = json.loads(g.read())
        # 数据量太大，太耗时，取消验证
        # input_schema.validate(input_data)
    return input_data


def write_output_data(output_data, output_path):
    with jsonlines.open(output_path, "w") as g:
        for a in output_data:
            output_data = output_schema.validate(a)
            g.write(output_data)

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
                type_name = "品类"
            elif question == "标题中品牌提及有哪些":
                type_name = "品牌"
            elif question == "标题中系列型号提及有哪些":
                type_name = "系列型号"
            if len(answers) > 0:
                labels.append([type_name, answers[0]["text"]])
        o = convert_ruler(context, labels)
        output_data.append(o)

    return output_data

def ruler(context,labels):
    output={}
    output["context"]=context
    type_map=defaultdict()
    for type_name,type_text in labels:
        type_map.setdefault(type_name,[]).append(type_text)
    output["ner"]=type_map
    return output


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
