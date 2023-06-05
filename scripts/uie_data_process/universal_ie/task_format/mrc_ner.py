#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
from collections import Counter, defaultdict
from typing import Dict, List

from torch.nn import InstanceNorm1d
from universal_ie.task_format.spannet import Spannet
from universal_ie.ie_format import Sentence


class MRCNER(Spannet):
    """ MRC NER format at https://github.com/ShannonAI/mrc-for-flat-nested-ner"""
    id_template = "%s#%s"

    def __init__(self, instance_json: Dict, language='en'):
        super().__init__(
            instance_json=instance_json,
            language=language
        )

    @ staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        counter = Counter()
        dataset = defaultdict(dict)
        with open(filename) as fin:
            for instance in json.load(fin):
                counter.update(['label sentence'])
                key, _ = instance['qas_id'].split('.')
                dataset[key]['tokens'] = instance['context'].split()
                if 'spans' not in dataset[key]:
                    dataset[key]['spans'] = list()
                for start, end in zip(instance['start_position'],
                                      instance['end_position']):
                    dataset[key]['spans'] += [{
                        'start': start,
                        'end': end,
                        'type': instance['entity_label']
                    }]
                    counter.update(['span'])

        sentence_list = list()
        for sentence_id, sentence in dataset.items():
            counter.update(['sentence'])
            mrc_instance = MRCNER(
                instance_json={
                    'tokens': sentence['tokens'],
                    'span_list': sentence['spans'],
                    'id': sentence_id
                },
                language=language
            )
            sentence_list += [mrc_instance.generate_instance()]

        print(filename, counter)

        return sentence_list
    

class QueryAlioutput(Spannet):
    
    id_template = "%s#%s"
    def __init__(self, instance_json: Dict, language='zh'):
        super().__init__(
            instance_json=instance_json,
            language=language
        )

    @ staticmethod
    def load_from_file(filename, language='zh') -> List[Sentence]:
        counter = Counter()
        dataset = defaultdict(dict)
        with open(filename,"r",encoding="utf-8") as fin:
            for index,line in enumerate(fin):
                index=str(index)
                ss=line.strip().split("\t")
                assert len(ss)==2
                query=ss[0]
                output=json.loads(ss[1])
                instances=output["output"]
                counter.update(['label sentence'])

                dataset[index]['tokens'] = list(query)
                if 'spans' not in dataset[index]:
                    dataset[index]['spans'] = list()
                for instance in instances:
                    type_name=instance["type"]
                    if type_name=="HP":
                        type_name="品牌"
                    elif type_name=="HC":
                        type_name="品类"
                    elif type_name=="XH":
                        type_name="型号"
                    else:
                        #其它类型直接跳过
                        continue

                    dataset[index]["spans"]+=[{
                        'start':instance["start"],
                        'end':instance['end']-1,
                        'type':type_name
                        }]
                    counter.update(['span'])

        sentence_list = list()
        for sentence_id, sentence in dataset.items():
            counter.update(['sentence'])
            mrc_instance = QueryAlioutput(
                instance_json={
                    'tokens': sentence['tokens'],
                    'span_list': sentence['spans'],
                    'id': sentence_id
                },
                language=language
            )
            sentence_list += [mrc_instance.generate_instance()]

        print(filename, counter)

        return sentence_list





