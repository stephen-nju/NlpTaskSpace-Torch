import collections
import json
import math
import re
import string
import torch
from typing import List
from torchmetrics import Metric
from transformers import BasicTokenizer
from transformers.utils import logging
from datahelper.bert_qa.bert_qa_dataset import QuestionAnswerInputExample, QuestionAnswerInputExampleFast, QuestionAnswerInputFeatures, QuestionAnswerInputFeaturesFast, QuestionAnswerOutputResult

from tqdm import tqdm

logger = logging.get_logger(__name__)


# 修改 transformers 的squad metrics，适用于中午的qa数据集验证
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
    
        # __import__('pdb').set_trace()

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
        return collections.OrderedDict([
            ("exact", 100.0 * sum(exact_scores.values()) / total),
            ("f1", 100.0 * sum(f1_scores.values()) / total),
            ("total", total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ("total", total),
        ])


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

    exact_threshold = apply_no_ans_threshold(exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh_v2(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

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
        for (i, c) in enumerate(text):
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
    for (i, tok_index) in tok_ns_to_s_map.items():
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

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


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
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
    tokenizer,
):
    # null_score_diff_threshold 每个结果有一个拒识的logits,有一个最佳span的logits 
    # 如果拒识logits减去最佳span的logits >null_score_diff_threshold 就拒识，否则保留
    # null_score_diff_threshold 一般为0，如果null_score_diff_threshold <0,相当于降低拒识的条件，越容易拒识

    
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        if isinstance(example, QuestionAnswerInputExampleFast):
            example_index = example.example_index

        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
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

                    if isinstance(example, QuestionAnswerInputExample) and isinstance(feature, QuestionAnswerInputFeatures):
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

                    elif isinstance(example, QuestionAnswerInputExampleFast) and isinstance(
                            feature, QuestionAnswerInputFeaturesFast):

                        offset_mapping = feature.offset_mapping

                        # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context 
                        token_is_max_context = feature.token_is_max_context if (
                            feature.token_is_max_context and isinstance(feature.token_is_max_context, dict)) else None
                        # 超出范围的答案

                        if (start_index >= len(offset_mapping) or end_index >= len(offset_mapping) or
                                offset_mapping[start_index] is None or len(offset_mapping[start_index]) < 2 or
                                offset_mapping[end_index] is None or len(offset_mapping[end_index]) < 2):
                            continue
                        #答案长度限制
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        #最大上下文
                        if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                            continue

                    else:
                        raise ValueError(
                            f"""example argument needs to be of type of (QuestionAnswerInputExample,QuestionAnswerInputExampleFast) and 
                            feature argument needs to be of type of (QuestionAnswerInputFeatures,QuestionAnswerInputFeaturesFast)"""
                        )
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        ))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                ))
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                if isinstance(example, QuestionAnswerInputExampleFast) and isinstance(feature, QuestionAnswerInputFeaturesFast):
                    offset_mapping = feature.offset_mapping
                    offsets = offset_mapping[pred.start_index][0], offset_mapping[pred.end_index][1]
                    context = example.context_text
                    final_text = context[offsets[0]:offsets[1]]
                elif isinstance(example, QuestionAnswerInputExample) and isinstance(feature, QuestionAnswerInputFeatures):
                    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]

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

                    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                else:
                    raise ValueError(
                        f"""example argument needs to be of type of (QuestionAnswerInputExample,QuestionAnswerInputExampleFast) and 
                        feature argument needs to be of type of (QuestionAnswerInputFeatures,QuestionAnswerInputFeaturesFast)"""
                    )

                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1, "No valid predictions"

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1, "No valid predictions"

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions


def compute_predictions_log_probs(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    start_n_top,
    end_n_top,
    version_2_with_negative,
    tokenizer,
    verbose_logging,
):
    """
    XLNet write prediction logic (more complex than Bert's). Write final predictions to the json file and log-odds of
    null if needed.

    Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_log_prob", "end_log_prob"])

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

    logger.info(f"Writing predictions to: {output_prediction_file}")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_logits[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_logits[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
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
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob,
                        ))

        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_log_prob + x.end_log_prob), reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            #
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            if hasattr(tokenizer, "do_lower_case"):
                do_lower_case = tokenizer.do_lower_case
            else:
                do_lower_case = tokenizer.do_lowercase_and_remove_accent

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_log_prob=pred.start_log_prob, end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="", start_log_prob=-1e6, end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1, "No valid predictions"
        assert best_non_null_entry is not None, "No valid predictions"

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

# ner 测评方案
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
    return p,r,f1

## report overall metric
def report_metric(input_data,pred_data,convert):
    inputs=convert(input_data,pred_data)
    data = inputs["data"]
    total = inputs["total_data"]
    e_types_list = ["HP", "HC", "XL"]
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

        predict= example["predict"]
        for ent in predict:
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
    
    p,r,f1=print_metrics(tp_ner_strict, fp_ner_strict, fn_ner_strict, "NER-strict-hardMatch")

    # per type
    for key in e_types_list:
        print_metrics(hard_boundaries[key]["tp"], hard_boundaries[key]["fp"], hard_boundaries[key]["fn"],
        f"Ner-strict-hardmatch-{key}")
    
    return f1

def convert_fast_example_pred(input_data:List[QuestionAnswerInputExampleFast],predict_data:collections.OrderedDict):
    output_data={"name":"predict","total_data":len(input_data)}
    data=[]

    # 先转化example数据
    query_map={}
    for example in input_data:
        query=example.context_text
        question=example.question_text
        ids=example.qas_id
        answers=example.answers
        if query not in query_map:
            output=[]
        else:
            output=query_map[query]
            
        if question=="标题中产品提及有哪些":
            type_name="HC"
        elif question=="标题中品牌提及有哪些":
            type_name="HP"
        elif question=="标题中系列型号提及有哪些":
            type_name="XL"
        else:
            raise ValueError("error value in question")
        if len(answers)==0:
          #无答案的数据不参与计算
            query_map[query]=output
            # 直接设置为空
            continue

        answer=answers[0]
        span=answer["text"]
        start=answer["answer_start"]
        output.append({"type":type_name,"start":start,"end":start+len(span),"span":span,"id":ids})
        query_map[query]=output
        
    # print(query_map)
    
    # __import__('ipdb').set_trace()
     # 处理pred数据
    for q,gts in query_map.items():
        ground_truth=[]
        o={"query":q}
        pred=[]
        for gt in gts:
            # 不评测无结果数据
            qas_id=gt["id"]
            if qas_id in predict_data:
                span=predict_data[qas_id]
                if span!="empty" and span!="":
                    pred.append({"type":gt["type"],"start":None,"end":None,"span":span})
        o["ground_truth"]=gts
        o["predict"]=pred
        data.append(o)
    output_data["data"]=data
    return output_data


def question_answer_evaluation(
    all_examples,
    all_features,
    all_results,
    tokenizer,
    n_best_size,
    max_answer_length,
    do_lower_case,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
):
    all_predictions = compute_predictions_logits(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        verbose_logging,
        version_2_with_negative,
        null_score_diff_threshold,
        tokenizer,
    )
    r = squad_evaluate(all_examples, all_predictions)
    print(r)
    # print(all_examples)
    # print(all_predictions)
    return report_metric(all_examples,all_predictions,convert_fast_example_pred)


class QuestionAnswerMetric(Metric):

    def __init__(self, evaluation):
        super().__init__(compute_on_step=False)
        self.evaluation = evaluation
        self.add_state("start_logits", [])
        self.add_state("end_logits", [])
        self.add_state("unique_ids", [])

    def update(self, unique_ids: torch.Tensor, start_logits: torch.Tensor, end_logits: torch.Tensor):
        self.unique_ids += unique_ids
        self.start_logits += start_logits
        self.end_logits += end_logits

    def compute(self):
        all_results = []
        for unique_id, start, end in zip(self.unique_ids, self.start_logits, self.end_logits):
            all_results.append(
                QuestionAnswerOutputResult(unique_id=int(unique_id.detach().cpu()),
                                           start_logits=start.squeeze().detach().cpu().tolist(),
                                           end_logits=end.squeeze().detach().cpu().tolist()))
        return self.evaluation(all_results=all_results)





        
