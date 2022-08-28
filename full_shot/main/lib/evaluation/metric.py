import logging

import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_curve


def accuracy(y_label, y_pred):
    return np.mean((y_label == y_pred).astype(np.float64))

def mean_per_class_accuracy(y_label, y_pred):
    accuracy_list = []
    cmatrix = confusion_matrix(y_label, y_pred)
    for i in range(len(cmatrix)):
        accuracy_list.append(cmatrix[i][i]/np.sum(cmatrix[i]))
    return np.mean(accuracy_list)

def mAP_11points(y_label, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_label, y_pred_proba)
    recall_thresholds = np.linspace(1, 0, 11, endpoint=True).tolist()
    precision_sum = 0
    recall_idx = 0
    precision_tmp = 0
    for threshold in recall_thresholds:
        while recall_idx < len(recall) and threshold <= recall[recall_idx]:
            precision_tmp = max(precision_tmp, precision[recall_idx])
            recall_idx += 1
        precision_sum += precision_tmp
    return precision_sum/11

def roc_auc_score_ovr(y_true, y_score):
    return roc_auc_score(y_true, y_score, multi_class="ovr")

def get_metric(metric_name):
    if metric_name == "accuracy":
        return accuracy
    if metric_name == "mean-per-class":
        return balanced_accuracy_score
    if metric_name == "11point_mAP":
        return mAP_11points
    if metric_name == "roc_auc":
        return roc_auc_score
    logging.error("Undefined metric.")
