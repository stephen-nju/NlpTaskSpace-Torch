# # -----*----coding:utf8-----*----
import numpy as np
import torch
from torchmetrics import Metric


class NerF1Metric(Metric):

    def __init__(self):

        super().__init__()
        self.add_state("true_positives", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, mapk2ij, label_encode, threshold=0):
        for i, score in enumerate(preds):
            R = set()
            for pair_id, tag_id in zip(*np.where(score.cpu().numpy() > threshold)):
                start, end = mapk2ij[pair_id][0], mapk2ij[pair_id][1]
                R.add((start, end, tag_id))

            T = set()
            for pair_id, tag_id in zip(*np.where(targets[i].cpu().numpy() > threshold)):
                start, end = mapk2ij[pair_id][0], mapk2ij[pair_id][1]
                T.add((start, end, tag_id))

            self.true_positives += len(R & T)
            self.false_positives += (len(R) - len(R & T))
            self.false_negative += (len(T) - len(R & T))

    def compute(self):
        precision = self.true_positives.sum().float() / (
                self.false_positives.sum().float() + self.true_positives.sum().float())
        recall = self.true_positives.sum().float() / (
                self.false_negative.sum().float() + self.true_positives.sum().float())

        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
