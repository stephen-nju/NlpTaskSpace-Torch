import argparse
import json
import os
from schema import Schema, And, Use, Optional
from typing import List, Tuple, Dict
import uuid
from tqdm import tqdm

# 四列分别表示query,品牌,品类,系列
InputFormat = List[Tuple[str, str, str, str]]

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

output_schema = Schema(
    {
        "name": And(str, len),
        "desc": And(str, len),
        "num": And(Use(int)),
        "data": [qas_schema],
    }
)


def read_input_data(path):
    input_data: InputFormat = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ss = line.strip().split("\t")
            input_data.append(ss)
    return input_data


def write_output_data(output_data, output_path):
    output_data = output_schema.validate(output_data)
    with open(output_path, "w", encoding="utf-8") as g:
        g.write(json.dumps(output_data, indent=2, ensure_ascii=False) + "\n")


def ruler(inputs: List):
    # 规定怎样进行转换，数据转换的核心逻辑
    query = inputs[0]
    brand = json.loads(inputs[1])
    category = json.loads(inputs[2])
    xl = json.loads(inputs[3])
    output = {"context": query}
    results = {"HC": category, "HP": brand, "XL": xl}
    qas=[]
    def finding_ans(key,data)
        ans=[]
        if key == "HC":
            question = "标题中产品提及有哪些"
        elif key == "HP":
            question = "标题中品牌提及有哪些"
        elif key == "XL" or key == "XH":
            question = "标题中系列型号提及有哪些"
        else:
            raise ValueError("key error ")
        if len(data)>0:
            for p in data:
                raw = p.strip()
                p = re.escape(raw)
                pattern = re.compile(p)
                re_data = re.finditer(pattern, query)
                start_index = []
                for r in re_data:
                    start_index.append(r.start(0))
                if len(start_index)>0:
                    ans.append({"text": raw, "answer_start": start_index[0]})
                else:
                    raise ValueError(f"match error query={query},raw match={raw}")
                
            return {"id": uuid.uuid1().hex, "question":question, "answers": ans}
        else:
            return {"id": uuid.uuid1().hex, "question": question, "answers": ans}

    for key,value in results.items():
        out=finding_ans(key,value)
        qas.append(out)
    output["qas"] = qas
    return output


def convert(input_data: InputFormat, name, desc, convert_ruler):
    output_data = {}
    output_data["name"] = name
    output_data["desc"] = desc
    output_data["num"] = len(input_data)
    data = []
    for inputs in tqdm(input_data, total=len(input_data)):
        o = convert_ruler(inputs)
        if o is not None:
            data.append(o)
        else:
            print(inputs)
    output_data["data"] = data
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chat gpt requests")
    parser.add_argument(
        "--input_data_path", type=str, help="input data path,data format is json"
    )
    parser.add_argument(
        "--output_data_path", type=str, default=".", help="output data path"
    )
    parser.add_argument("--name", type=str, default="eval_v1", help="dataset name")
    parser.add_argument("--desc", type=str, help="dataset description")
    args = parser.parse_args()
    i = read_input_data(args.input_data_path)
    o = convert(i, name=args.name, desc=args.desc, convert_ruler=ruler_v1)
    write_output_data(o, args.output_data_path)
