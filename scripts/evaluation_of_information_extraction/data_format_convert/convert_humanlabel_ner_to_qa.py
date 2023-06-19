import argparse
import json
import os
from schema import Schema, And, Use, Optional
from typing import List, Tuple, Dict
import uuid
from tqdm import tqdm
InputFormat = List[Tuple[str, Dict[str, str]]]

qas_schema = Schema({
    "context": And(str, len),
    "qas": [{
        "id": Optional(str, int),
        "question": And(str, len),
        "answers": [{
            "text": And(str, len),
            "answer_start": int,
        }]
    }]
})

output_schema = Schema({
    "name": And(str, len),
    "desc": And(str,len),
    "num":And(Use(int)),
    "data": [qas_schema],
})


def read_input_data(path):
    input_data: InputFormat = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ss = line.strip().split("\t")
            input_data.append((ss[0], json.loads(ss[1])["output"]))
    return input_data


def write_output_data(output_data, output_path):
    output_data = output_schema.validate(output_data)
    with open(output_path, "w", encoding="utf-8") as g:
        g.write(json.dumps(output_data, indent=2, ensure_ascii=False) + "\n")


def ruler_v1(query, result: List):
    # 规定怎样进行转换，数据转换的核心逻辑
    output = {"context": query}
    qas = []
    
    already=set()
    for res in result:
        ans = []
        key = res["type"]
        if key == "HC":
            question = "标题中产品提及有哪些"
        elif key == "HP":
            question = "标题中品牌提及有哪些"
        elif key == "XL" or key=="XH":
            key="XL"
            question = "标题中系列型号提及有哪些"
        else:
            continue
        
        already.add(key)
        text = res["span"]
        start = res["start"]
        ans.append({"text": text, "answer_start": int(start)})

        qas.append({"id": uuid.uuid1().hex, "question": question, "answers": ans})

        # 没有答案，拒识的要加进去
    for key in ["HC", "HP", "XL"]:
        if key == "HC":
            question = "标题中产品提及有哪些"
        if key == "HP":
            question = "标题中品牌提及有哪些"
        if key == "XL":
            question = "标题中系列型号提及有哪些"
        if key not in already:
            qas.append({"id": uuid.uuid1().hex, "question": question, "answers": []})

    output["qas"] = qas
    return output


def convert(input_data: InputFormat, name, desc, convert_ruler):
    output_data = {}
    output_data["name"] = name
    output_data["desc"] = desc
    output_data["num"] = len(input_data)
    data = []
    for query, result in tqdm(input_data,total=len(input_data)):
        o = convert_ruler(query, result)
        data.append(o)

    output_data["data"] = data
    return output_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="chat gpt requests")
    parser.add_argument("--input_data_path", type=str, help="input data path,data format is json")
    parser.add_argument("--output_data_path", type=str, default=".", help="output data path")
    parser.add_argument("--name",type=str,default="eval_v1",help="dataset name")
    parser.add_argument("--desc",type=str,help="dataset description")
    args = parser.parse_args()
    i=read_input_data(args.input_data_path)
    o=   convert(i,name=args.name,desc=args.desc,convert_ruler=ruler_v1)
    write_output_data(o,args.output_data_path)

    
