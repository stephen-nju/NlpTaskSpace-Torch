# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: squad.py
@time: 2021/6/11 10:13
"""

import dataclasses
from functools import partial
from multiprocessing.dummy import Pool
from os import cpu_count
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding

MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class QuestionAnswerInputExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
            self,
            qas_id,
            question_text,
            context_text,
            answer_text,
            raw_start_position,
            title,
            answers=[],
            is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers
        self.raw_start_position = raw_start_position

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if self.raw_start_position is not None and not self.is_impossible:
            self.start_position = char_to_word_offset[self.raw_start_position]
            self.end_position = char_to_word_offset[
                min(self.raw_start_position + len(self.answer_text) - 1, len(char_to_word_offset) - 1)
            ]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (str(self.qas_id))
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class QuestionAnswerInputExampleFast:
    def __init__(
            self,
            example_index,
            question_text,
            context_text,
            title,
            answers,
            qas_id,
            is_impossible=False,
    ):
        self.example_index = example_index
        self.question_text = question_text
        self.context_text = context_text
        self.title = title
        self.qas_id=qas_id
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (str(self.example_index))
        s += ", question_text: %s" % (self.question_text)
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


# @dataclasses.dataclass
# class QAInputExample:
#     qas_id: Any
#     question_text: str
#     context_text: str
#     answer_text: str
#     raw_start_position: int
#     title: Any
#     is_impossible: bool = False
#     answers: list = dataclasses.field(default_factory=list)
#     start_position: int = dataclasses.field(default=0)
#     end_position: int = dataclasses.field(default=0)

#     def __post_init__(self):
#         doc_tokens = []
#         char_to_word_offset = []
#         prev_is_whitespace = True
#         # Split on whitespace so that different tokens may be attributed to their original position.
#         for c in self.context_text:
#             if _is_whitespace(c):
#                 prev_is_whitespace = True
#             else:
#                 if prev_is_whitespace:
#                     doc_tokens.append(c)
#                 else:
#                     doc_tokens[-1] += c
#                 prev_is_whitespace = False
#             char_to_word_offset.append(len(doc_tokens) - 1)

#         self.doc_tokens = doc_tokens
#         self.char_to_word_offset = char_to_word_offset

#         if self.raw_start_position is not None and not self.is_impossible:
#             self.start_position = char_to_word_offset[self.raw_start_position]
#             self.end_position = char_to_word_offset[
#                 min(self.raw_start_position + len(self.answer_text) - 1, len(char_to_word_offset) - 1)
#             ]


@dataclasses.dataclass
class QuestionAnswerInputFeatures:
    # 模型预测的输入
    input_ids: list
    attention_mask: list
    token_type_ids: list
    cls_index: list
    p_mask: list
    example_index: int
    unique_id: int
    paragraph_len: int
    token_is_max_context: bool
    tokens: list
    token_to_orig_map: Dict
    start_position: int
    end_position: int
    is_impossible: bool
    qas_id: int
    encoding: BatchEncoding = None


class QuestionAnswerInputFeaturesFast:
    """
    采用 transformers 中的SquadFeatures
      Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context:
            List of booleans identifying which tokens have their maximum context in this feature object. If a token
            does not have their maximum context in this feature object, it means that another feature object has more
            information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
        encoding: optionally store the BatchEncoding with the fast-tokenizer alignment methods.
    """

    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cls_index,
            p_mask,
            example_index,
            unique_id,  # feature的唯一id
            paragraph_len,
            token_is_max_context,
            tokens,
            offset_mapping,
            start_position,
            end_position,
            is_impossible,
            qas_id: str = None,
            encoding: BatchEncoding = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.offset_mapping = offset_mapping

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id

        self.encoding = encoding


@dataclasses.dataclass
class QuestionAnswerOutputResult:
    unique_id: int
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    cls_logits: torch.Tensor = None


def convert_train_example_to_features(example, tokenizer, max_seq_length, doc_stride,
                                      max_query_length, padding_strategy,
                                      is_training):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            print(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][
                                 : encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                        len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                    tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(
                truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
                "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implementation also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens:] = 0
        else:
            p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            QuestionAnswerInputFeatures(
                input_ids=span["input_ids"],
                attention_mask=span["attention_mask"],
                token_type_ids=span["token_type_ids"],
                cls_index=cls_index,
                p_mask=p_mask.tolist(),
                example_index=0,
                # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features


def convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        padding_strategy="max_length",
        threads=3,
        tqdm_enabled=True,
):
    threads = min(threads, cpu_count())
    with Pool(threads) as p:
        annotate_ = partial(
            convert_train_example_to_features,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert examples to features",
                disable=not tqdm_enabled,
            )
        )

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
            features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    return features


class QuestionAnswerDataset(Dataset):
    def __init__(self, features
                 ):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        feature = self.features[i]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        label_mask = [1] + feature.token_type_ids[1:]
        # TODO label mask right?
        start_position = torch.tensor(feature.start_position, dtype=torch.long)
        end_position = torch.tensor(feature.end_position, dtype=torch.long)
        label_mask = torch.tensor(label_mask, dtype=torch.long)
        inputs.update({"start_position": start_position,
                       "end_position": end_position,
                       "label_mask": label_mask,
                       "unique_id": feature.unique_id
                       })
        return inputs
