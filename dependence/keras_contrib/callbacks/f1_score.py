"""
@author: 18073701
@email:  18073701@suning.com
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@software: PyCharm
@file: f1_score.py
@time: 2020/3/10 11:12
"""
import numpy as np
import itertools
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import classification_report as sk_classification_report
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score


class ClsMetrics(Callback):

    def __init__(self, validation_data):
        super(ClsMetrics, self).__init__()
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred = []
        val_true = []
        for (x_val, y_val) in self.validation_data:
            val_pred_batch = self.model.predict_on_batch(x_val)
            val_pred_batch = np.argmax(np.asarray(val_pred_batch).round(), axis=1)
            val_pred.append(val_pred_batch)
            val_true.append(np.argmax(np.asarray(y_val).round(), axis=1))
        val_pred = np.asarray(list(itertools.chain.from_iterable(val_pred)))
        val_true = np.asarray(list(itertools.chain.from_iterable(val_true)))

        _val_f1 = f1_score(val_true, val_pred, average='macro')
        _val_recall = recall_score(val_true, val_pred, average='macro')
        _val_precision = precision_score(val_true, val_pred, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        # print('— val_f1: %f — val_precision: %f — val_recall %f' % (_val_f1, _val_precision, _val_recall))
        # print(' — val_f1:', _val_f1)
        print(sk_classification_report(val_true, val_pred, digits=4))
        return


class F1Metrics(Callback):

    def __init__(self, id2label, pad_value=0, validation_data=None, digits=4, name="f1"):
        """
        Args:
            id2label (dict): id to label mapping.
            (e.g. {1: 'B-LOC', 2: 'I-LOC'})
            pad_value (int): padding value.
            digits (int or None): number of digits in printed classification report
              (use None to print only F1 score without a report).
        """
        super(F1Metrics, self).__init__()
        self.id2label = id2label
        self.pad_value = pad_value
        self.validation_data = validation_data
        self.digits = digits
        self.is_fit = validation_data is None
        self.name = name

    def convert_idx_to_name(self, y, array_indexes):
        """Convert label index to name.

        Args:
            y (np.ndarray): label index 2d array.
            array_indexes (list): list of valid index arrays for each row.

        Returns:
            y: label name list.
        """
        y = [[self.id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    def predict(self, X, y):
        """Predict sequences.

        Args:
            X (np.ndarray): input data.
            y (np.ndarray): tags.

        Returns:
            y_true: true sequences.
            y_pred: predicted sequences.
        """
        y_pred = self.model.predict_on_batch(X)

        # reduce dimension.
        y_true = np.argmax(y, -1)
        y_pred = np.argmax(y_pred, -1)

        non_pad_indexes = [np.nonzero(y_true_row != self.pad_value)[0] for y_true_row in y_true]

        y_true = self.convert_idx_to_name(y_true, non_pad_indexes)
        y_pred = self.convert_idx_to_name(y_pred, non_pad_indexes)

        return y_true, y_pred

    def score(self, y_true, y_pred):
        """Calculate f1 score.

        Args:
            y_true (list): true sequences.
            y_pred (list): predicted sequences.

        Returns:
            score: f1 score.
        """
        score = f1_score(y_true, y_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        if self.digits:
            print(classification_report(y_true, y_pred, digits=self.digits))
        return score

    def on_epoch_end(self, epoch, logs={}):
        if self.is_fit:
            self.on_epoch_end_fit(epoch, logs)
        else:
            self.on_epoch_end_fit_generator(epoch, logs)

    def on_epoch_end_fit(self, epoch, logs={}):
        X = self.validation_data[0]
        y = self.validation_data[1]
        y_true, y_pred = self.predict(X, y)
        score = self.score(y_true, y_pred)
        logs['f1'] = score

    def on_epoch_end_fit_generator(self, epoch, logs={}):
        y_true = []
        y_pred = []
        for X, y in self.validation_data:
            y_true_batch, y_pred_batch = self.predict(X, y)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
        score = self.score(y_true, y_pred)
        logs[self.name] = score
