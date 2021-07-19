#!/usr/bin/env python
# -*- encoding=utf-8 -*-
"""
@author: 18073701
@email:  18073701@suning.com
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@software: pycharm
@file: task4ner.py
@time: 2020/3/7 12:47 下午
"""
import logging
from typing import List

from dataprocessor import NerDataGenerator
from dataprocessor import get_entities
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras_contrib.callbacks import F1Metrics
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

from dependence.bert4keras.backend import K
from dependence.bert4keras.layers import *
from dependence.bert4keras.tokenizers import Tokenizer

logging.basicConfig(filename='logger.log', level=logging.INFO)
logger = logging.getLogger(__name__)


# 主要用来构建模型和数据集
class Task4BertNer():

    def __init__(self, args):
        super(Task4BertNer, self).__init__(args)

    def load_data_set(self, data, tokenizer, label_dict, label_pad_token):
        # 根据不同的task_name 加载不同数据集

        return NerDataGenerator(data=data,
                                tokenizer=tokenizer,
                                label_dict=label_dict,
                                batch_size=self.args.batch_size,
                                label_pad_token=label_pad_token,
                                maxlen=self.args.maxlen,
                                )

    def build_model(self, word_dict, label_dict):
        from models import Bert4NerBaseModel
        model = Bert4NerBaseModel(self.args, word_dict, label_dict)
        return model.build_model()

    def train(self):
        # 准备训练数据

        train = self.load_data(self.args.train_data)
        dev = self.load_data(self.args.dev_data)
        # 处理数据，构建字典
        labels = ["O", "B-adj", "I-adj", "X"]
        tokenizer = self.build_vocab(user_dict=self.args.bert_dict_path)
        word_dict = tokenizer.token_dict
        label_dict = self.build_label_dict(built=True, labels=labels)
        label_dict_invert = {j: i for i, j in label_dict.items()}
        # 构建模型（模型依赖于字典）
        model = self.build_model(word_dict, label_dict)
        # callback 等回调函数

        dev_data = self.load_data_set(data=dev, tokenizer=tokenizer, label_dict=label_dict, label_pad_token="X")
        call_backs = self.build_callbacks(label_dict_invert, dev_data, label_pad_value=label_dict.get("X"))
        # 训练模型
        train_generator = self.load_data_set(train, tokenizer=tokenizer, label_dict=label_dict, label_pad_token="X")
        model.fit_generator(train_generator.forfit(),
                            steps_per_epoch=len(train_generator),
                            epochs=self.args.epochs,
                            callbacks=call_backs)
        model.save("bert_ner_model.h5")
        # self.save(model=model, model_name="bert_pretrain")

    def save(self, model, model_name):
        output_dir = self.args.output_root
        out_prefix = "output_"
        out_nodes = []
        for i in range(len(model.outputs)):
            out_nodes.append(out_prefix + str(i + 1))
            tf.identity(model.outputs[i], out_prefix + str(i + 1))
        sess = K.get_session()
        from tensorflow.python.framework import graph_util, graph_io
        init_graph = sess.graph.as_graph_def()
        main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
        graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    @staticmethod
    def convert_idx_to_name(y, id2label, array_indexes):
        y = [[id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    @staticmethod
    def build_callbacks(id2label, validation_data, label_pad_value):
        call_backs = []
        f1 = F1Metrics(id2label, validation_data=validation_data, pad_value=label_pad_value)
        call_backs.append(f1)
        early_stop = EarlyStopping('f1', patience=4, mode='max', verbose=2, restore_best_weights=True)
        call_backs.append(early_stop)

        return call_backs

    @staticmethod
    def build_vocab(user_dict=None, **kwargs):
        # todo 添加不自定义词典
        if user_dict is not None:
            processor = Tokenizer(user_dict, do_lower_case=True)

        else:
            data = kwargs.get("data")
            from dataprocessor import NerProcessor
            processor = NerProcessor()
            processor.fit(data)
        return processor

    @staticmethod
    def build_label_dict(user_dict=None, built=True, **kwargs):
        if user_dict is not None:
            return user_dict
        else:
            if built:
                labels = kwargs.get("labels")
                id2labels = dict(enumerate(labels))
                label_dict = {j: i for i, j in id2labels.items()}
                return label_dict
            else:
                # 从语料直接传入的数据,包含类别[O,PRE,ORG],默认BIO
                labels = kwargs.get("labels", None)
                if labels is None:
                    ValueError("Missing labels")
                labels = list(set(labels))
                new_labels = []
                for label in labels:
                    if label != "O":
                        new_labels.append("B-" + label)
                        new_labels.append("I-" + label)
                pad_token = kwargs.get("pad_token", None)
                if pad_token is None:
                    raise ValueError("Missing pad token")
                new_labels.append(pad_token)
                new_labels.append("O")
                id2labels = dict(enumerate(new_labels))
                label_dict = {j: i for i, j in id2labels.items()}
                return label_dict

    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):
                if not l:
                    continue
                d, last_flag = [], ''
                for c in l.split('\n'):
                    char, this_flag = c.split('\t')[:2]
                    if this_flag == 'O' and last_flag == 'O':
                        d[-1][0] += char
                    elif this_flag == 'O' and last_flag != 'O':
                        d.append([char, 'O'])
                    elif this_flag[:1] == 'B':
                        d.append([char, this_flag[2:]])
                    else:
                        d[-1][0] += char
                    last_flag = this_flag
                D.append(d)
        return D


class Task4BertNerWithPos():

    def __init__(self, args):
        super(Task4BertNerWithPos, self).__init__(args)

    def build_model(self, word_dict, label_dict, pos_dict):
        from models import Ber4NerWithPos
        model = Ber4NerWithPos(self.args, word_dict, label_dict, pos_dict)
        return model.build_model()

    def train(self):
        # 准备训练数据

        train = self.load_data(self.args.train_data)
        dev = self.load_data(self.args.dev_data)
        dev_v2 = self.load_data(self.args.dev_data_v2)
        # 处理数据，构建字典
        labels = ["O", "B-adj", "I-adj", "X"]
        tokenizer = self.build_vocab(user_dict=self.args.bert_dict_path)
        word_dict = tokenizer.token_dict
        label_dict = self.build_label_dict(built=True, labels=labels)
        label_dict_invert = {j: i for i, j in label_dict.items()}
        # 构建模型（模型依赖于字典）
        pos_dict = self.build_pos_dict([x[2] for x in train])

        train_generator = self.load_data_set(train,
                                             tokenizer=tokenizer,
                                             label_dict=label_dict,
                                             label_pad_token="X",
                                             pos_dict=pos_dict,
                                             pos_pad_token="POS_PAD"
                                             )
        model = self.build_model(word_dict, label_dict, pos_dict)
        model.summary()
        # callback 等回调函数
        dev_data = self.load_data_set(data=dev,
                                      tokenizer=tokenizer,
                                      label_dict=label_dict,
                                      label_pad_token="X",
                                      pos_dict=pos_dict,
                                      pos_pad_token="POS_PAD"
                                      )
        dev_data_v2 = self.load_data_set(data=dev_v2,
                                         tokenizer=tokenizer,
                                         label_dict=label_dict,
                                         label_pad_token="X",
                                         pos_dict=pos_dict,
                                         pos_pad_token="POS_PAD"
                                         )

        call_backs = self.build_callbacks(id2label=label_dict_invert,
                                          validation_data=dev_data,
                                          validation_data_v2=dev_data_v2,
                                          label_pad_value=label_dict.get("X")
                                          )
        # 训练模型
        model.fit_generator(train_generator.forfit(),
                            steps_per_epoch=len(train_generator),
                            epochs=self.args.epochs,
                            callbacks=call_backs)
        # model.save("bert_ner_model.h5")
        # self.save(model=model, model_name="bert_pretrain")

    def predict(self):
        custom_objects = {'CRF': CRF,
                          'crf_loss': crf_loss,
                          'crf_accuracy': crf_accuracy}
        pre_model = load_model("bert_ner_model.h5",
                               custom_objects=custom_objects)
        test = self.load_data(self.args.dev_data)
        labels = ["O", "B-obj", "I-obj", "X"]
        tokenizer = self.build_vocab(user_dict=self.args.bert_dict_path)
        label_dict = self.build_label_dict(labels=labels)
        label_dict_invert = {j: i for i, j in label_dict.items()}

        test_generation = self.load_data_set(data=test, tokenizer=tokenizer, label_dict=label_dict, label_pad_token="X")
        for data in test_generation:
            # print(data)
            pre = pre_model.predic(data[0])
            # mask掉pad的标签
            pre = np.argmax(pre, axis=-1)
            non_pad_indexes = [np.nonzero(y_true_row != label_dict["X"])[0] for y_true_row in pre]
            y_pred = self.convert_idx_to_name(pre, label_dict_invert, non_pad_indexes)
            entities = get_entities(y_pred)
            # todo 添加获取实体内容的方法
            print(entities)

    def load_data_set(self, data, tokenizer, label_dict, label_pad_token, pos_dict, pos_pad_token):
        from dataprocessor import NerDataGeneratorWithPos
        return NerDataGeneratorWithPos(data=data,
                                       tokenizer=tokenizer,
                                       label_dict=label_dict,
                                       batch_size=self.args.batch_size,
                                       label_pad_token=label_pad_token,
                                       pos_dict=pos_dict,
                                       pos_pad_token=pos_pad_token,
                                       maxlen=self.args.maxlen,
                                       )

    @staticmethod
    def build_callbacks(id2label, validation_data, validation_data_v2, label_pad_value):
        call_backs = []
        f1 = F1Metrics(id2label, validation_data=validation_data, name="f1", pad_value=label_pad_value)
        call_backs.append(f1)
        f1_v2 = F1Metrics(id2label, validation_data=validation_data_v2, name="f1_v2", pad_value=label_pad_value)
        call_backs.append(f1_v2)
        early_stop = EarlyStopping('f1_v2', patience=4, mode='max', verbose=2, restore_best_weights=True)
        call_backs.append(early_stop)
        return call_backs

    @staticmethod
    def load_data(filename: str) -> List:
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):
                if not l:
                    continue
                s, f, p, m = [], [], [], []
                for c in l.split('\n'):
                    words = c.split("\t")
                    if len(words) == 4:
                        char, this_flag, pos, mask_c = words[:]
                        s.append(char)
                        f.append(this_flag)
                        p.append(pos)
                        m.append(int(mask_c))
                D.append((s, f, p, m))
        return D

    @staticmethod
    def build_vocab(user_dict=None, **kwargs):
        # todo 添加不自定义词典
        if user_dict is not None:
            processor = Tokenizer(user_dict, do_lower_case=True)

        else:
            data = kwargs.get("data")
            from dataprocessor import NerProcessor
            processor = NerProcessor()
            processor.fit(data)
        return processor

    @staticmethod
    def build_pos_dict(X, pos_pad_token="POS_PAD"):
        from bert4keras.tokenizers import PosTokenizer
        pos_tokenizer = PosTokenizer()
        pos_tokenizer.fit(X)
        # pos添加pad字符
        pos_tokenizer.add_token(pos_pad_token)
        pos_dict = pos_tokenizer.token_dict
        return pos_dict

    @staticmethod
    def build_label_dict(user_dict=None, built=True, **kwargs):
        if user_dict is not None:
            return user_dict
        else:
            if built:
                labels = kwargs.get("labels")
                id2labels = dict(enumerate(labels))
                label_dict = {j: i for i, j in id2labels.items()}
                return label_dict
            else:
                # 从语料直接传入的数据,包含类别[O,PRE,ORG],默认BIO
                labels = kwargs.get("labels", None)
                if labels is None:
                    ValueError("Missing labels")
                labels = list(set(labels))
                new_labels = []
                for label in labels:
                    if label != "O":
                        new_labels.append("B-" + label)
                        new_labels.append("I-" + label)
                pad_token = kwargs.get("pad_token", None)
                if pad_token is None:
                    raise ValueError("Missing pad token")
                new_labels.append(pad_token)
                new_labels.append("O")
                id2labels = dict(enumerate(new_labels))
                label_dict = {j: i for i, j in id2labels.items()}
                return label_dict

    @staticmethod
    def convert_idx_to_name(y, id2label, array_indexes):
        y = [[id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

# 主要用来构建模型和数据集
# class Task4NerAppendCategory(Task4Ner):
#
#     def __init__(self, args):
#         super(Task4NerAppendCategory, self).__init__(args)
#
#     def build_model(self, word_dict, label_dict):
#         from models import Bert4NerAppendCategoryModel
#         model = Bert4NerAppendCategoryModel(self.args, word_dict, label_dict)
#         return model.build_model()
#
#     def train(self):
#         # 准备训练数据
#
#         data = self.load_data(self.args.train_data)
#         dev = self.load_data(self.args.dev_data)
#         dev_v2 = self.load_data(self.args.dev_data_v2)
#         # 处理数据，构建字典
#         labels = ["O", "B-adj", "I-adj", "X"]
#         tokenizer = self.build_vocab(user_dict=self.args.bert_dict_path)
#         word_dict = tokenizer.token_dict
#         label_dict = self.build_label_dict(built=True, labels=labels)
#         label_dict_invert = {j: i for i, j in label_dict.items()}
#         # 构建模型（模型依赖于字典）
#         model = self.build_model(word_dict, label_dict)
#         # callback 等回调函数
#         dev_data = self.load_data_set(data=dev, tokenizer=tokenizer, label_dict=label_dict, label_pad_token="X")
#         dev_data_v2 = self.load_data_set(data=dev_v2, tokenizer=tokenizer, label_dict=label_dict, label_pad_token="X")
#         # 添加未知品类词的测试
#         call_backs = self.build_callbacks(label_dict_invert, dev_data, dev_data_v2, label_pad_value=label_dict.get("X"))
#         # 训练模型
#         train_generator = self.load_data_set(data, tokenizer=tokenizer, label_dict=label_dict, label_pad_token="X")
#         model.fit_generator(train_generator.forfit(),
#                             steps_per_epoch=len(train_generator),
#                             epochs=self.args.epochs,
#                             callbacks=call_backs)
#         model.save("bert_ner_model.h5")
#         # self.save(model=model, model_name="bert_pretrain")
#
#     def write_log_file(self,
#                        log_file: str,
#                        log_dict: Dict) -> None:
#         with codecs.open(log_file, "w", encoding="utf-8") as l:
#             string = json.dumps(log_dict)
#             l.write(string)
#
#     def loader_log(self, log_dir: str) -> Dict[str, Dict]:
#         log_files = os.listdir(log_dir)
#         log_information_dict = {}
#         # 加载log信息
#         print(">>>>>>>>>>>>>>>>>>>>>>>>>>加载log信息>>>>>>>>>>>>>>>>>>>>>>>>>>")
#         for log_file in log_files:
#             detail_log = os.path.join(log_dir, log_file)
#             if os.path.isfile(detail_log):
#                 flag = log_file.split("_")[0]
#                 if flag == "log":
#                     with open(detail_log, "r", encoding="utf-8") as f:
#                         log_data = json.load(f)
#                         file_name = log_data["file"]
#                         log_information_dict[file_name] = log_data
#         return log_information_dict
#
#     def compute_output(self,
#                        batch_size: int or None,
#                        batch_token_id: List,
#                        batch_segment_id: List,
#                        batch_tokens: List,
#                        batch_source_text: List,
#                        label_dict: Dict[str, int],
#                        label_dict_invert: Dict[int, str]) -> List:
#         batch_token_ids = sequence_padding(batch_token_id, padding=0)
#         batch_segment_ids = sequence_padding(batch_segment_id)
#         if batch_size is None:
#             pre = self.pre_model.predict([batch_token_ids, batch_segment_ids])
#         else:
#             pre = self.pre_model.predict([batch_token_ids, batch_segment_ids], batch_size=batch_size)
#         # print(tokenizer.decode(data[0]))
#         # mask掉pad的标签
#         pre = np.argmax(pre, axis=-1)
#         non_pad_indexes = [np.nonzero(y_true_row != label_dict["X"])[0] for y_true_row
#                            in pre]
#         y_preds = self.convert_idx_to_name(pre, label_dict_invert, non_pad_indexes)
#         entity = []
#         for y_pred, tokens in zip(y_preds, batch_tokens):
#             entities = get_entities(y_pred, tokens)
#             entity.append("#".join([entity[0] for entity in entities]))
#         assert len(entity) == len(batch_source_text)
#         res = [line.strip() + "\t" + en + "\n" for line, en in zip(batch_source_text, entity)]
#         return res
#
#     def process_on_new_file(self,
#                             file_name: str,
#                             total_size: int,
#                             reader_file: str,
#                             writer_file: str,
#                             log_file: str,
#                             tokenizer,
#                             label_dict: Dict[str, int],
#                             label_dict_invert: Dict[int, str]
#                             ) -> None:
#         with tqdm(total=total_size) as bar:
#             bar.set_description("Prcessing new {}".format(file_name))
#             with open(writer_file, "w", encoding="utf-8") as g:
#                 log_dict = {"file": file_name, "finish": False, "already_line": 0}
#                 with open(log_file, "w", encoding="utf-8") as l:
#                     string = json.dumps(log_dict)
#                     l.write(string)
#                 file_finish = False
#                 # 每1000步写入log文件一次
#                 with open(reader_file, "r", encoding="utf-8") as f:
#                     batch_token_id = []
#                     batch_segment_id = []
#                     batch_tokens = []
#                     batch_source_text = []
#                     index = 0
#                     for line in f:
#                         index = index + 1
#                         # line_split=line.split("\t")
#                         log_dict["already_line"] = log_dict["already_line"] + 1
#                         if index % 1000 == 0:
#                             self.write_log_file(log_file=log_file, log_dict=log_dict)
#                         line_split = line.strip().split("\t")
#                         text = line_split[2]
#
#                         if len(line_split) == 4:
#                             category = line_split[-1]
#                             # 数据预处理
#                             category = "[SEP]" + category
#                         else:
#                             category = ""
#                             category = "[SEP]" + category
#
#                         text = re.sub(pattern="\s+", string=text, repl="|")
#                         text += category
#                         # 添加品类词
#                         tokens = tokenizer.tokenize(text, max_length=128)
#                         token_ids = tokenizer.tokens_to_ids(tokens)
#                         segment_ids = [0] * len(token_ids)
#                         bar.update(1)
#                         # 构建预测的batch
#                         batch_source_text.append(line)
#                         batch_token_id.append(token_ids)
#                         batch_segment_id.append(segment_ids)
#                         batch_tokens.append(tokens)
#                         if len(batch_segment_id) == len(batch_token_id) == 512:
#                             res = self.compute_output(batch_size=512,
#                                                       batch_token_id=batch_token_id,
#                                                       batch_segment_id=batch_segment_id,
#                                                       batch_tokens=batch_tokens,
#                                                       batch_source_text=batch_source_text,
#                                                       label_dict=label_dict,
#                                                       label_dict_invert=label_dict_invert)
#                             g.writelines(res)
#                             batch_source_text = []
#                             batch_token_id = []
#                             batch_segment_id = []
#                             batch_tokens = []
#                     else:
#                         # 结束时候运行
#                         res = self.compute_output(batch_size=None,
#                                                   batch_token_id=batch_token_id,
#                                                   batch_segment_id=batch_segment_id,
#                                                   batch_tokens=batch_tokens,
#                                                   batch_source_text=batch_source_text,
#                                                   label_dict=label_dict,
#                                                   label_dict_invert=label_dict_invert)
#                         g.writelines(res)
#                         bar.set_description("{} end".format(file_name))
#                         log_dict["file"] = file_name
#                         log_dict["finish"] = True
#                         log_dict["already_line"] = index
#                         self.write_log_file(log_file=log_file, log_dict=log_dict)
#
#     def process_on_old_file(self,
#                             file_name: str,
#                             total_size: int,
#                             reader_file: str,
#                             writer_file: str,
#                             log_file: str,
#                             tokenizer,
#                             label_dict: Dict[str, int],
#                             label_dict_invert: Dict[int, str],
#                             log_dict: Dict):
#         log_dict = log_dict
#         already_line = log_dict["already_line"]
#         with tqdm(total=total_size) as bar:
#             bar.set_description("Prcessing old {}".format(file_name))
#             write_file_len = self.file_len(writer_file)
#             if os.path.isfile(writer_file):
#                 already_line = write_file_len
#             with codecs.open(reader_file, "r", encoding="utf-8") as f:
#                 file_finish = False
#                 with codecs.open(writer_file, "a", encoding="utf-8") as g:
#                     batch_token_id = []
#                     batch_segment_id = []
#                     batch_tokens = []
#                     batch_source_text = []
#                     index = 0
#                     for line in f:
#                         index += 1
#                         if index < already_line:
#                             bar.update(1)
#                             continue
#                         log_dict["already_line"] = log_dict["already_line"] + 1
#                         if index % 1000 == 0:
#                             self.write_log_file(log_file=log_file, log_dict=log_dict)
#                         bar.update(1)
#                         line_split = line.strip().split("\t")
#                         if len(line_split) == 4:
#                             category = line_split[-1]
#                             # 数据预处理
#                             category = "[SEP]" + category
#                         else:
#                             category = ""
#                             category = "[SEP]" + category
#                         text = line_split[2]
#                         text = re.sub(pattern="\s+", string=text, repl="|")
#                         text += category
#                         # 添加品类词
#                         tokens = tokenizer.tokenize(text, max_length=128)
#                         token_ids = tokenizer.tokens_to_ids(tokens)
#                         segment_ids = [0] * len(token_ids)
#                         batch_source_text.append(line)
#                         batch_token_id.append(token_ids)
#                         batch_segment_id.append(segment_ids)
#                         batch_tokens.append(tokens)
#                         if len(batch_segment_id) == len(batch_token_id) == 512 or file_finish:
#                             res = self.compute_output(batch_size=512,
#                                                       batch_token_id=batch_token_id,
#                                                       batch_segment_id=batch_segment_id,
#                                                       batch_tokens=batch_tokens,
#                                                       batch_source_text=batch_source_text,
#                                                       label_dict=label_dict,
#                                                       label_dict_invert=label_dict_invert)
#                             g.writelines(res)
#
#                             batch_source_text = []
#                             batch_token_id = []
#                             batch_segment_id = []
#                             batch_tokens = []
#                     else:
#                         res = self.compute_output(batch_size=None,
#                                                   batch_token_id=batch_token_id,
#                                                   batch_segment_id=batch_segment_id,
#                                                   batch_tokens=batch_tokens,
#                                                   batch_source_text=batch_source_text,
#                                                   label_dict=label_dict,
#                                                   label_dict_invert=label_dict_invert)
#                         g.writelines(res)
#                         bar.set_description("{} end".format(file_name))
#                         log_dict["file"] = file_name
#                         log_dict["finish"] = True
#                         log_dict["already_line"] = index
#                         self.write_log_file(log_file=log_file, log_dict=log_dict)
#
#     def process_on_file(self,
#                         file_name: str,
#                         total_size: int,
#                         reader_file: str,
#                         writer_file: str,
#                         log_file: str,
#                         tokenizer,
#                         label_dict: Dict[str, int],
#                         label_dict_invert: Dict[int, str],
#                         log_information_dict: Dict[str, Dict] or None
#                         ) -> None:
#         if log_information_dict is not None:
#             if file_name in log_information_dict.keys():
#                 assert file_name == log_information_dict[file_name]["file"]
#                 finish_read = log_information_dict[file_name]["finish"]
#                 if finish_read:
#                     logger.info("{} process over".format(file_name))
#                     # already_line = log_information_dict[file].already_line
#
#                 else:
#                     self.process_on_old_file(file_name=file_name,
#                                              total_size=total_size,
#                                              reader_file=reader_file,
#                                              writer_file=writer_file,
#                                              log_file=log_file,
#                                              tokenizer=tokenizer,
#                                              label_dict=label_dict,
#                                              label_dict_invert=label_dict_invert,
#                                              log_dict=log_information_dict[file_name])
#             else:
#                 self.process_on_new_file(file_name=file_name,
#                                          total_size=total_size,
#                                          reader_file=reader_file,
#                                          writer_file=writer_file,
#                                          log_file=log_file,
#                                          tokenizer=tokenizer,
#                                          label_dict=label_dict,
#                                          label_dict_invert=label_dict_invert)
#
#     def predict(self):
#         custom_objects = {'CRF': CRF,
#                           'crf_loss': crf_loss,
#                           'crf_accuracy': crf_accuracy,
#                           'PositionEmbedding': PositionEmbedding,
#                           }
#         self.pre_model = load_model("bert_ner_model.h5",
#                                     custom_objects=custom_objects)
#         # self.pre_model = multi_gpu_model(self.pre_model, gpus=2)
#         # todo 预测时候的输入[token_ids,segment_ids]
#         # test = self.load_data(self.args.dev_data)
#         labels = ["O", "B-adj", "I-adj", "X"]
#         tokenizer = self.build_vocab(user_dict=self.args.bert_dict_path)
#         label_dict = self.build_label_dict(labels=labels)
#         label_dict_invert = {j: i for i, j in label_dict.items()}
#         # 配置数据的路径
#         # data_read_dir = "/home/nlpbigdata/net_disk_project/zhubin/topic_extract_project/data_repository/huimai_merge"
#         # data_write_dir="/home/nlpbigdata/local_disk/huimai_decoration"
#         # log_dir="/home/nlpbigdata/local_disk/huimai_log"
#
#         # data_read_dir="/home/nlpbigdata/local_disk/sql_full_category_data_new/gds_intelligent_list_commodity/title_pinlei"
#         # data_write_dir="/home/nlpbigdata/local_disk/sql_full_category_data_new/gds_intelligent_list_commodity/title_pinlei_res"
#         # log_dir="/home/nlpbigdata/local_disk/sql_full_category_data_new/gds_intelligent_list_commodity/title_pinlei_log"
#
#         # data_read_dir="/home/nlpbigdata/net_disk_project/zhujingtao/sql_full_category_data_new/gds_intelligent_list_commodity_new/title_pinlei"
#         data_read_dir = "/home/nlpbigdata/net_disk_project/zhujingtao/sql_full_category_data_new/gds_intelligent_list_commodity_new/department"
#         data_write_dir = "/home/nlpbigdata/local_disk/department"
#         log_dir = "/home/nlpbigdata/local_disk/department_log_new"
#
#         # file_lists=["title_pinlei_00045.txt","title_pinlei_00046.txt"]
#         # file_lists=[
#         #             "title_pinlei_00027.txt",
#         #             "title_pinlei_00029.txt",
#         #             "title_pinlei_00031.txt",
#         #             "title_pinlei_00048.txt"]
#         # file_lists=[
#         #             "title_pinlei_00051.txt",
#         #             "title_pinlei_00053.txt",
#         #             "title_pinlei_99999.txt"]
#         # file_lists=["title_depa_16_00",
#         #             "title_depa_16_02",
#         #             "title_depa_16_04"]
#
#         # file_lists=["title_depa_16_06",
#         #             "title_depa_16_08",
#         #             "title_depa_16_10"]
#
#         # file_lists= ["title_depa_16_12",
#         #             "title_depa_16_14",
#         #             "title_depa_16_01"]
#
#         # file_lists=["title_depa_16_03",
#         #             "title_depa_16_05",
#         #             "title_depa_16_07"]
#         file_lists= ["title_depa_16_09",
#                     "title_depa_16_11",
#                     "title_depa_16_13"]
#
#         # file_lists=["title_pinlei_00016.txt"]
#         start = True
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#
#         log_information_dict = self.loader_log(log_dir)
#
#         if len(log_information_dict) > 0:
#             start = False
#         # file_nums=len(os.listdir(data_read_dir))
#         file_nums = len(file_lists)
#
#         if start:
#             print(">>>>>>>>>>>>>>>>>>> start with no log >>>>>>>>>>>>>>>>>>>>")
#             with tqdm(total=file_nums) as bar:
#                 # for file in os.listdir(data_read_dir):
#                 for file in file_lists:
#                     if os.path.isfile(os.path.join(data_read_dir, file)):
#                         # 判断是否为文件
#                         file_start = file.split("_")[0]
#                         # file_name, file_suffix = os.path.splitext(file)
#                         if file_start == "title":
#                             # 构建读取的文件
#
#                             reader_file = os.path.join(data_read_dir, file)
#                             writer_file = os.path.join(data_write_dir, file)
#                             total_size = self.file_len(reader_file)
#                             log_file = os.path.join(log_dir, "log_" + file)
#                             self.process_on_new_file(file_name=file,
#                                                      total_size=total_size,
#                                                      reader_file=reader_file,
#                                                      writer_file=writer_file,
#                                                      log_file=log_file,
#                                                      tokenizer=tokenizer,
#                                                      label_dict=label_dict,
#                                                      label_dict_invert=label_dict_invert)
#                             bar.update(1)
#         else:
#             print(">>>>>>>>>>start from already>>>>>>>>>>>>>>>>>>>>>")
#             with tqdm(total=file_nums) as bar:
#                 # for file in os.listdir(data_read_dir):
#                 for file in file_lists:
#                     bar.set_description("Processing : {}".format(file))
#                     if os.path.isfile(os.path.join(data_read_dir, file)):
#                         file_start = file.split("_")[0]
#                         # 不同文件设置不同的
#                         if file_start == "title":
#                             # txt 文件才可以按行计数
#                             reader_file = os.path.join(data_read_dir, file)
#                             total = self.file_len(reader_file)
#                             writer_file = os.path.join(data_write_dir, file)
#                             log_file = os.path.join(log_dir, "log_" + file)
#                             self.process_on_file(file_name=file,
#                                                  total_size=total,
#                                                  reader_file=reader_file,
#                                                  writer_file=writer_file,
#                                                  log_file=log_file,
#                                                  tokenizer=tokenizer,
#                                                  label_dict=label_dict,
#                                                  label_dict_invert=label_dict_invert,
#                                                  log_information_dict=log_information_dict)
#                             bar.update(1)
#
#     #     def predict(self):
#     #         custom_objects = {'CRF': CRF,
#     #                           'crf_loss': crf_loss,
#     #                           'crf_accuracy': crf_accuracy,
#     #                           'PositionEmbedding':PositionEmbedding,
#     #         }
#     #         pre_model = load_model("bert_ner_model.h5",
#     #                               custom_objects=custom_objects)
#     #         pre_model = multi_gpu_model(pre_model, gpus=4)
#     #         # todo 预测时候的输入[token_ids,segment_ids]
#     #         # test = self.load_data(self.args.dev_data)
#     #         labels = ["O", "B-adj", "I-adj", "X"]
#     #         tokenizer = self.build_vocab(user_dict=self.args.bert_dict_path)
#     #         label_dict = self.build_label_dict(labels=labels)
#     #         label_dict_invert = {j: i for i, j in label_dict.items()}
#     #         total=56084487
#     #         index=0
#     # #         data_read_dir="/home/nlpbigdata/local_disk/gds_intelligent_list_commodity/title_pinlei"
#     # #         data_write_dir="/home/nlpbigdata/local_disk/gds_intelligent_list_commodity/title_pinlei_res"
#     #         data_read_dir="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_code_repository/bert4kerasApp/demo_data_with_brand_category.txt"
#     #         data_write_dir="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_code_repository/bert4kerasApp/demo_data_res.txt"
#     #         with tqdm(total=total) as bar:
#     #                 with codecs.open(data_write_dir,"w",encoding="utf-8") as g:
#     #                     with codecs.open(data_read_dir,"r",encoding="utf-8") as f:
#     #                         batch_token_id=[]
#     #                         batch_segment_id=[]
#     #                         batch_tokens=[]
#     #                         batch_source_text=[]
#     #                         finish=False
#     #                         for line in f:
#     #                             if line is None:
#     #                                 finish=True
#     #                             bar.update(1)
#     #                             index+=1
#     #                             line_split=line.strip().split("\t")
#     #                             text=line_split[5]
#     #                             category=line_split[-1]
#     #                             # 数据预处理
#     #                             category="[SEP]"+category
#     #                             text = re.sub(pattern="\s+", string=text, repl="|")
#     #                             text+=category
#     #                             # 添加品类词
#     #                             tokens = tokenizer.tokenize(text, max_length=128)
#     #                             token_ids = tokenizer.tokens_to_ids(tokens)
#     #                             segment_ids = [0] * len(token_ids)
#     #                             if len(batch_segment_id)==len(batch_token_id)==1536 or finish:
#     #                                 batch_token_ids= sequence_padding(batch_token_id, padding=0)
#     #                                 batch_segment_ids = sequence_padding(batch_segment_id)
#     #                                 pre = pre_model.predict([batch_token_ids,batch_segment_ids],batch_size=1536)
#     #                             # print(tokenizer.decode(data[0]))
#     #                             # mask掉pad的标签
#     #                                 pre = np.argmax(pre, axis=-1)
#     #                                 non_pad_indexes = [np.nonzero(y_true_row != label_dict["X"])[0] for y_true_row in pre]
#     #                                 y_preds = self.convert_idx_to_name(pre, label_dict_invert, non_pad_indexes)
#     #                                 entity=[]
#     #                                 for y_pred,tokens in zip(y_preds,batch_tokens):
#     #                                     entities = get_entities(y_pred, tokens)
#     #                                     entity.append("#".join([entity[0] for entity in entities]))
#     #                             # print(entities)
#     #                             # print(text+"\t"+"".join([entity[0] for entity in entities])+"\n")
#     #                                 # for line,en in zip(batch_source_text,entity):
#     #                                 #     g.write(line.strip()+"\t"+en +"\n" )
#     #                                 g.writelines([line.strip()+"\t"+en +"\n" for line,en in zip(batch_source_text,entity)])
#     #                                 batch_source_text=[]
#     #                                 batch_token_id=[]
#     #                                 batch_segment_id=[]
#     #                                 batch_tokens=[]
#     #                             else:
#     #                                 batch_source_text.append(line)
#     #                                 batch_token_id.append(token_ids)
#     #                                 batch_segment_id.append(segment_ids)
#     #                                 batch_tokens.append(tokens)
#
#     def load_data_set(self, data, tokenizer, label_dict, label_pad_token):
#
#         return NerDataGeneratorAppendCategory(data=data,
#                                               tokenizer=tokenizer,
#                                               label_dict=label_dict,
#                                               batch_size=self.args.batch_size,
#                                               label_pad_token=label_pad_token,
#                                               maxlen=self.args.maxlen,
#                                               )
#
#     @staticmethod
#     def build_callbacks(id2label, validation_data, validation_data_v2, label_pad_value):
#         call_backs = []
#         f1 = F1Metrics(id2label, validation_data=validation_data, name="f1", pad_value=label_pad_value)
#         call_backs.append(f1)
#         f1_v2 = F1Metrics(id2label, validation_data=validation_data_v2, name="f1_v2", pad_value=label_pad_value)
#         call_backs.append(f1_v2)
#         early_stop = EarlyStopping('f1_v2', patience=4, mode='max', verbose=2, restore_best_weights=True)
#         call_backs.append(early_stop)
#
#         return call_backs
#
#     @staticmethod
#     def load_data(filename):
#         D = []
#         with open(filename, encoding='utf-8') as f:
#             f = f.read()
#             for l in f.split('\n\n'):
#                 if not l:
#                     continue
#                 d, last_flag = [], ''
#                 for c in l.split('\n'):
#                     words = c.split("\t")
#                     if len(words) == 3:
#                         char, this_flag = words[:2]
#                         if this_flag == 'O' and last_flag == 'O':
#                             d[-1][0] += char
#                         elif this_flag == 'O' and last_flag != 'O':
#                             d.append([char, 'O'])
#                         elif this_flag[:1] == 'B':
#                             d.append([char, this_flag[2:]])
#                         else:
#                             d[-1][0] += char
#                         last_flag = this_flag
#                     if len(words) == 1:
#                         if last_flag == "category":
#                             d[-1][0] += words[0]
#                         else:
#                             d.append([words[0], "category"])
#                             last_flag = "category"
#                 D.append(d)
#         return D
#
#     @staticmethod
#     def file_len(file):
#         with open(file, 'r', encoding="utf-8") as f:
#             for index, line in enumerate(f):
#                 pass
#             return index
