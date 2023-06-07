# -*- coding:utf-8 -*-

import argparse
import json
import re
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.DEBUG)

def has_duplicate(tmp_list):
    """ has duplicate ?
    """
    if tmp_list == []:
        return False

    if type(tmp_list[0]) == str:
        if len(tmp_list) == len(set(tmp_list)):
            return False
        else:
            return True

    if type(tmp_list[0]) == list:
        tmp = []
        for t in tmp_list:
            if t not in tmp:
                tmp.append(t)
        if len(tmp_list) == len(tmp):
            return False
        else:
            return True


def get_correct_list_from_response_list(target_list, response_list):
    """
    target_list 和 response_list 均有可能包含重复的 item
    """
    # print(f"{target_list}=={response_list}")
    res = []
    if not has_duplicate(response_list):
        res = [item for item in response_list if item in target_list]
    else:
        if not has_duplicate(target_list):
            # 去重
            uni_response_list = []
            for item in response_list:
                if item not in uni_response_list:
                    uni_response_list.append(item)
            res = [item for item in uni_response_list if item in target_list]
        else:
            res = []
            processed_item_list = []
            for item in response_list:
                if item not in processed_item_list:
                    processed_item_list.append(item)

                    num_item = response_list.count(item)
                    if num_item == 1:  # not duplicate
                        if item in target_list:
                            res.append(item)
                    else:  # duplicate
                        if item in target_list:
                            num_item_in_target = target_list.count(item)
                            num_item_correct = min([num_item, num_item_in_target])
                            res += [item] * num_item_correct

    return res


def print_metrics(tp, fp, fn, task):
    p, r, f1 = 0.0, 0.0, 0.0

    if tp + fp != 0:
        p = 1.0 * tp / (tp + fp)
    if tp + fn != 0:
        r = 1.0 * tp / (tp + fn)
    if p + r != 0.0:
        f1 = 2.0 * p * r / (p + r)

    print("{}\n| p: {:.4f}, r: {:.4f}, f1: {:.4f} | tp: {:4d}, fp: {:4d}, fn: {:4d}, tp+fn: {:4d}\n".format(
        task,
        round(p, 4),
        round(r, 4),
        round(f1, 4),
        tp,
        fp,
        fn,
        tp + fn,
    ))
    return f1


def get_chatgpt_result(response):
    gpt_data_dict = json.loads(response)
    res = []
    
    try:
        gpt_content = gpt_data_dict["choices"][0]["message"]["content"]
    except:
        logging.error(f"get gpt content error")
        return res
    
    try:
        if isinstance(gpt_content,str):
            if "输出结果为：" in gpt_content:
                gpt_content=gpt_content.replace("输出结果为：","")
            elif "输出为：" in gpt_content:
                gpt_content=gpt_content.replace("输出为：","")

            elif "输出结果为:" in gpt_content:
                gpt_content=gpt_content.replace("输出结果为:","")
            elif "输出为:" in gpt_content:                
                gpt_content=gpt_content.replace("输出为:","")

        r = eval(gpt_content)
    except:
        logging.error(f"eval gpt content error")
    
        return res

    if isinstance(r,dict):
        for key, value in r.items():
            if key == "品牌":
                type_name = "HP"
            elif key == "品类":
                type_name = "HC"
            elif key == "商品系列型号":
                type_name = "XL"
            else:
                type_name = "Other"

            if isinstance(value, str):
                if value != "":
                    res.append({"type": type_name, "span": value, "start": None, "end": None})
            elif isinstance(value, list):
                for v in value:
                    if v != "":
                        res.append({"type": type_name, "span": v, "start": None, "end": None})
            elif isinstance(value, dict):
                continue
                # res.append({"type":type_name,"span":value,"start":None,"end":None})
            else:
                continue
    return res


def get_chatgpt_result_v3(response):
    gpt_data_dict = json.loads(response)
    res = []
    try:
        gpt_content = gpt_data_dict["choices"][0]["message"]["content"]
    except:
        logging.error(f"get gpt content error")
        return res
    try:
        if isinstance(gpt_content,str):
            gpt_content=gpt_content.strip()
            if gpt_content=="无":
                return res
            # 用字符串解析dict
            r={}
            out=re.findall("\{(.*)\}",gpt_content)[0]
            left,right=out.strip().split(":")
            left=re.sub('[\'"]','', left)
            right=re.sub('[\'"]','', right)
            if left=="品牌":
                r["品牌"]=right
            
    except:
        logging.error(f"eval gpt content error")
        return res
    
    if isinstance(r,dict):
        for key, value in r.items():
            if key == "品牌":
                type_name = "HP"
            else:
                type_name = "Other"
            if isinstance(value, str):
                value=value.strip()
                if value != "无":
                    res.append({"type": type_name, "span": value, "start": None, "end": None})
            elif isinstance(value, list):
                for v in value:
                    v=v.strip()
                    if v != "无":
                        res.append({"type": type_name, "span": v, "start": None, "end": None})
            elif isinstance(value, dict):
                continue
                # res.append({"type":type_name,"span":value,"start":None,"end":None})
            else:
                continue
    return res


## report overall metric
def report_metric(inputs):
    data = inputs["data"]
    total = inputs["total_data"]

    e_types_list = ["HP", "HC", "CX", "CP", "XL", "XH"]
    ## per type
    hard_boundaries = dict()
    for key in e_types_list:
        hard_boundaries[key] = {"tp": 0, "fp": 0, "fn": 0}

    num_entity = 0
    tp_ner_boundaries = 0
    fp_ner_boundaries = 0
    fn_ner_boundaries = 0
    tp_ner_strict = 0
    fp_ner_strict = 0
    fn_ner_strict = 0

    for example in tqdm(data, total=total):
        ## target
        strict_target_list = []
        boundaries_target_list = []

        ## predict
        strict_predict_list = []
        boundaries_predict_list = []
        ## per type target
        boundaries_target_list_dict = {}
        # per type predict
        boundaries_predict_list_dict = {}

        for key in e_types_list:
            boundaries_target_list_dict[key] = []
            boundaries_predict_list_dict[key] = []

        ground_truth = example["ground_truth"]
        for truth in ground_truth:
            # print(ground_truth)
            type_name = truth["type"]
            if type_name in e_types_list:
                start = truth["start"]
                end = truth["end"]
                span = truth["span"].lower()
                # 总类型
                # 统一转化为小写字母进行比较
                strict_target_list.append([type_name, span])
                boundaries_target_list.append(span)
                ## per type
                boundaries_target_list_dict[type_name].append(span)
                num_entity += 1

        chatgpt = example["chatgpt_text"]
        # 定制化解析chatgpt的接口
        gpt_res=get_chatgpt_result_v3(chatgpt)
        print(f"ground_truth={ground_truth} ***** gpt_res={gpt_res}")
        for ent in gpt_res:

            if len(ent) > 0:
                ent_name = ent["span"].lower()
                ent_type = ent["type"]
                if ent_type in e_types_list:
                    strict_predict_list.append([ent_type, ent_name])
                    boundaries_predict_list.append(ent_name)
                    # per type
                    boundaries_predict_list_dict[ent_type].append(ent_name)

        ## hard-match
        strict_correct_list = get_correct_list_from_response_list(strict_target_list, strict_predict_list)
        # boundaries_correct_list = get_correct_list_from_response_list(boundaries_target_list, boundaries_predict_list)

        tp_ner_strict += len(strict_correct_list)
        fp_ner_strict += len(strict_predict_list) - len(strict_correct_list)
        fn_ner_strict += len(strict_target_list) - len(strict_correct_list)

        # tp_ner_boundaries += len(boundaries_correct_list)
        # fp_ner_boundaries += len(boundaries_predict_list) - len(boundaries_correct_list)
        # fn_ner_boundaries += len(boundaries_target_list) - len(boundaries_correct_list)

        for key in e_types_list:
            cur_correct = get_correct_list_from_response_list(boundaries_target_list_dict[key],
                                                              boundaries_predict_list_dict[key])
            hard_boundaries[key]["tp"] += len(cur_correct)
            hard_boundaries[key]["fp"] += len(boundaries_predict_list_dict[key]) - len(cur_correct)
            hard_boundaries[key]["fn"] += len(boundaries_target_list_dict[key]) - len(cur_correct)

    print("#sentence: {}, #entity: {}".format(len(data), num_entity))

    print_metrics(tp_ner_strict, fp_ner_strict, fn_ner_strict, "NER-strict-hardMatch")

    # per type
    for key in e_types_list:
        print_metrics(hard_boundaries[key]["tp"], hard_boundaries[key]["fp"], hard_boundaries[key]["fn"],
                      f"Ner-strict-hardmatch-{key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chat gpt requests")
    parser.add_argument("--data", type=str,
                        default="./chatgpt_promt_test_v1_output.txt",
                        help="input data path,data format is json")
    parser.add_argument("--output_dir", type=str, default=".", help="output data path")
    args = parser.parse_args()
    input_dict = json.loads(open(args.data, "r", encoding="utf-8").read())

    # total = input_json["total_data"]
    # name = input_json["name"]
    # input_data = input_json["data"]

    report_metric(input_dict)
