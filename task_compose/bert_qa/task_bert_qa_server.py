# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
import sys
import torch
import argparse
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers.models.bert.tokenization_bert import BertTokenizer
from collections import namedtuple
from torch import Tensor
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
import itertools
from typing import List, Any, Optional
import collections
import logging
import time

log = logging.getLogger()

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

import json
import math
import re
import string

from transformers.models.bert import BasicTokenizer

logger = logging.getLogger()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print(f"Missing prediction for {qas_id}")
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval[f"{prefix}_{k}"] = new_eval[k]


def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["has_ans_exact"] = has_ans_exact
    main_eval["has_ans_f1"] = has_ans_f1


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1 = get_raw_scores(examples, preds)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

    return evaluation


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for i, c in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(f"Unable to find text: '{pred_text}' in '{orig_text}'")
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(f"Length not equal after stripping spaces: '{orig_ns_text}' vs '{tok_ns_text}'")
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for i, tok_index in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position: (orig_end_position + 1)]
    return output_text, orig_start_position


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def compute_predictions_logits(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        verbose_logging,
        version_2_with_negative,
        null_score_diff_threshold,
        tokenizer,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file:
        logger.info(f"Writing predictions to: {output_prediction_file}")
    if output_nbest_file:
        logger.info(f"Writing nbest to: {output_nbest_file}")
    if output_null_log_odds_file and version_2_with_negative:
        logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for example_index, example in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for feature_index, feature in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                print(f"==doc start=={orig_doc_start}====doc end={orig_doc_end}")
                print(f"====featrue={feature}")
                print(f"====example={example}")

                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text, in_doc_start = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                # 获取在原始中start和end_position
                if orig_doc_start == 0:
                    s = in_doc_start
                else:
                    offset = sum([len(d) for d in example.doc_tokens[:orig_doc_start]]) + orig_doc_start
                    s = in_doc_start + offset

                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                s = 0
                seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit, start_index=s,
                                 end_index=s + len(final_text)))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit, start_index=0,
                                     end_index=0))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=0, end_index=0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=0, end_index=0))

        if len(nbest) < 1:
            raise ValueError("No valid predictions")

        total_scores = []
        best_non_null_entry = None
        print(f"====nbest={nbest}")

        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)
        nbest_json = []
        entry_probablity = {}
        for i, entry in enumerate(nbest):
            entry_probablity[entry] = probs[i]
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["start_index"] = entry.start_index
            output["end_index"] = entry.end_index
            nbest_json.append(output)

        if len(nbest_json) < 1:
            raise ValueError("No valid predictions")

        if not version_2_with_negative:
            all_predictions[example.qas_id] = [nbest_json[0]["text"], nbest_json[0]["probability"],
                                               nbest_json[0]["start_index"], nbest_json[0]["end_index"]]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ["", 0, 0, 0, 0]
            else:
                print(f"====={best_non_null_entry}")
                all_predictions[example.qas_id] = [best_non_null_entry.text, entry_probablity[best_non_null_entry],
                                                   best_non_null_entry.start_index, best_non_null_entry.end_index]
        all_nbest_json[example.qas_id] = nbest_json

    if output_prediction_file:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if output_nbest_file:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if output_null_log_odds_file and version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
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


@dataclasses.dataclass
class QAInputExample:
    qas_id: Any
    question_text: str
    context_text: str
    answer_text: str
    raw_start_position: int
    title: Any
    is_impossible: bool = False
    answers: list = dataclasses.field(default_factory=list)
    start_position: int = dataclasses.field(default=0)
    end_position: int = dataclasses.field(default=0)

    def __post_init__(self):
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

        if self.raw_start_position is not None and not self.is_impossible:
            self.start_position = char_to_word_offset[self.raw_start_position]
            self.end_position = char_to_word_offset[min(self.raw_start_position + len(self.answer_text) - 1,
                                                        len(char_to_word_offset) - 1)]


@dataclasses.dataclass
class QAInputFeatures:
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


@dataclasses.dataclass
class QAOutputResult:
    unique_id: int
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    cls_logits: torch.Tensor = None


def convert_train_example_to_features(example, tokenizer, max_seq_length, doc_stride, max_query_length,
                                      padding_strategy,
                                      is_training):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position:(end_position + 1)])
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

        (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens, tok_start_position,
                                                                      tok_end_position,
                                                                      tokenizer, example.answer_text)

    spans = []

    truncated_query = tokenizer.encode(example.question_text,
                                       add_special_tokens=False,
                                       truncation=True,
                                       max_length=max_query_length)

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (tokenizer.model_max_length - tokenizer.max_len_single_sentence +
                             1 if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET else tokenizer.model_max_length -
                                                                                         tokenizer.max_len_single_sentence)
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
                non_padded_ids = encoded_dict["input_ids"][:encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (len(encoded_dict["input_ids"]) - 1 -
                                            encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id))
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or ("overflowing_tokens" in encoded_dict and
                                                        len(encoded_dict["overflowing_tokens"]) == 0):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (j if tokenizer.padding_side == "left" else
                     spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j)
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
            p_mask[-len(span["tokens"]):-(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(tokenizer.get_special_tokens_mask(span["input_ids"],
                                                                             already_has_special_tokens=True)).nonzero()

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
            QAInputFeatures(
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
            ))
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
            ))

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features,
                                 total=len(features),
                                 desc="add example index and unique id",
                                 disable=not tqdm_enabled):
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


sys.path.insert(0, '/opt/files/QueryCorrect/src/search_query_correct/detector_modules')
sys.path.insert(0, '/opt/files/QueryCorrect/src/search_query_correct/detector_modules/tokenization')

DetectorInfo = namedtuple("DetectorInfo", ["phrase", "start", "end", "score", "type"])
# 模型, 配置文件, 资源文件
model_base_dir = '/opt/files/mxmodels'


# model_base_dir = '/home/nlpbigdata/net_disk_project/xufeifei/search_engine/model_x/query_phrase_extraction'


def truncated_normal_(tensor, mean=0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class BertForQAConfig(BertConfig):

    def __init__(self, **kwargs):
        super(BertForQAConfig, self).__init__(**kwargs)


class BertForQuestionAnswering(BertPreTrainedModel):
    """Finetuning Bert Model for the question answering task."""

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_classifier = nn.Linear(config.hidden_size, 2)
        self.qa_classifier.weight = truncated_normal_(self.qa_classifier.weight, mean=0, std=0.02)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor):
        """
        Args:
            input_ids: Bert input tokens, tensor of shape [batch, seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [batch, seq_len]
            attention_mask: attention mask, tensor of shape [batch, seq_len]
        Returns:
            start_logits: start/non-start logits of shape [batch, seq_len]
            end_logits: end/non-end logits of shape [batch, seq_len]
        """
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_heatmap = self.dropout(bert_outputs[0])  # [batch, seq_len, hidden]
        logits = self.qa_classifier(sequence_heatmap)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        #         print(start_logits)
        return start_logits, end_logits


class BertForQA(pl.LightningModule):

    def __init__(self):
        super().__init__()
        bert_config = BertForQAConfig.from_pretrained(
            "/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model")
        self.model = BertForQuestionAnswering.from_pretrained(
            "/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model",
            config=bert_config)
        self.tokenizer = BertTokenizer.from_pretrained(
            "/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model")

    def forward(self, input_ids, attention_mask, token_type_ids):
        """forward inputs to BERT models."""
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


class Detector(object):

    def __init__(self):
        ## main ##
        log.info("start:品类识别模型初始化")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bert_config = BertForQAConfig.from_pretrained(
            "/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model")

        #         self.model=BertForQA.load_from_checkpoint("./bert_model.ckpt")
        #         self.model.to(self.device)
        #         self.model.eval()
        #         print(self.model.state_dict())
        #         torch.save(self.model.model.state_dict(), "./output/pytorch_model.bin")

        #         self.model.model.save_pretrained("./output")
        self.model = BertForQuestionAnswering(bert_config).from_pretrained("./output", config=bert_config).to(
            self.device)
        #         print(self.model.state_dict())
        self.tokenizer = BertTokenizer.from_pretrained(
            "/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model")
        log.info("end:品类识别模型初始化完成")

    # 推理调用的主方法入口，支持字符串、JSON、二进制参数类型
    def detect(self, query) -> List[DetectorInfo]:
        for i in range(1):
            predict_examples = []
            predict_examples.append(
                QAInputExample(
                    qas_id=int(time.time()),
                    title=None,
                    question_text="标题中产品提及有哪些",
                    context_text=query,
                    answer_text=None,
                    raw_start_position=None,
                    answers=[]
                ))

            predict_features = convert_examples_to_features(examples=predict_examples,
                                                            tokenizer=self.tokenizer,
                                                            max_query_length=64,
                                                            max_seq_length=128,
                                                            doc_stride=128,
                                                            is_training=False
                                                            )

            log.info("start:开始query品类词抽取")

            predict_results, all_results = [], []
            # 批预测
            input_ids = []
            attention_mask = []
            token_type_ids = []
            unique_id = []

            for p_features in predict_features:
                input_ids.append(p_features.input_ids)
                attention_mask.append(p_features.attention_mask)
                token_type_ids.append(p_features.token_type_ids)
                unique_id.append(p_features.unique_id)

            a = time.time()
            start_logits, end_logits = self.model.forward(
                input_ids=torch.tensor(input_ids, dtype=torch.long).to(self.device),
                token_type_ids=torch.tensor(token_type_ids, dtype=torch.long).to(self.device),
                attention_mask=torch.tensor(attention_mask, dtype=torch.long).to(self.device)
            )

            print(f"===model==={time.time() - a}")
            single_start_logits = torch.split(start_logits, 1, dim=0)
            single_end_logits = torch.split(end_logits, 1, dim=0)
            for unique_id, start, end in zip(unique_id, single_start_logits, single_end_logits):
                predict_results.append(
                    QAOutputResult(unique_id=int(unique_id),
                                   start_logits=start.squeeze().detach().cpu().tolist(),
                                   end_logits=end.squeeze().detach().cpu().tolist(),
                                   cls_logits=None))

            predict_results = compute_predictions_logits(all_examples=predict_examples,
                                                         all_features=predict_features,
                                                         all_results=predict_results,
                                                         tokenizer=self.tokenizer,
                                                         n_best_size=5,
                                                         max_answer_length=10,
                                                         do_lower_case=True,
                                                         output_prediction_file=None,
                                                         output_nbest_file="a.txt",
                                                         output_null_log_odds_file=None,
                                                         verbose_logging=True,
                                                         version_2_with_negative=True,
                                                         null_score_diff_threshold=0,
                                                         )
            query_phrase_res = []
            for _, p_r in predict_results.items():
                # TODO start end 需要减去query的长度
                if p_r[0] != "empty" and p_r[0] != "":
                    query_phrase_res.append(DetectorInfo(p_r[0], p_r[2] - 12, p_r[3] - 11, p_r[1], "category"))
            log.info("end:结束query品类词抽取")
            print(f"===={query_phrase_res}")
            # query_phrase_res = []
            # phrase_result_dict = compute_phrase(feature_infos, prediction_results)
            # for passage_id in phrase_result_dict:
            #     passage = passage_dict[int(passage_id)]
            #     phrase_items = phrase_result_dict[passage_id]['phrase_items']
            #     phrase_list = []
            #     for phrase_item in phrase_items:
            #         start = phrase_item['start']
            #         end = phrase_item['end']
            #         start_prob = phrase_item['start_prob']
            #         end_prob = phrase_item['end_prob']
            #         phrase = passage[start:end + 1]
            #         phrase_score = min(start_prob, end_prob)
            #         phrase_list.append((phrase, start, end, str(phrase_score)))
            #         log.info('passage:[%s], phrase:[%s], start:[%d], end:[%d], score:[%lf]' %
            #                  (passage, phrase, start, end, phrase_score))
            #         query_phrase_res.append(DetectorInfo(phrase, start, end + 1, phrase_score, "category"))

        #         torch.save(self.model.model.state_dict(), "./output/pytorch_model.bin")

        return predict_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--s", type=str, help="bert config dir")

    args = parser.parse_args()

    query = args.s

    detector = Detector()
    s = time.time()
    res = detector.detect(query)
    print(f"====spantine={time.time() - s}")
    print("res: ", res)
