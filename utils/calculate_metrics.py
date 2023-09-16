import torch

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix, multilabel_confusion_matrix)

class MultiLeadECG_Classification_Metrics_Calculator(object):
    def __init__(self):
        super(MultiLeadECG_Classification_Metrics_Calculator).__init__()

        self.threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
        self.disease_label = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
        self.metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        self.total_metrics_dict = dict()
        self.confusion_matrix = np.array([[[0, 0],
                                          [0, 0]] for _ in range(len(self.disease_label))])
        for metric in self.metrics_list:
            self.total_metrics_dict[metric] = dict()
            for disease_label in self.disease_label:
                self.total_metrics_dict[metric][disease_label] = 0
        print(self.total_metrics_dict)

    def get_metrics_dict(self, y_pred, y_true):
        y_pred = (y_pred.cpu().detach().numpy() >= self.threshold).astype(np.int_)
        y_true = y_true.cpu().detach().numpy().astype(np.int_)

        multilabel_confusion_matrix_sample = multilabel_confusion_matrix(y_true, y_pred)
        for class_idx in range(len(self.disease_label)):
            self.confusion_matrix[class_idx] += multilabel_confusion_matrix_sample[class_idx]

        return self.confusion_matrix

        # # [[TN, FT]
        # #  [FN, TP]]
        #
        # metrics_dict = dict()
        # for metric in self.metrics_list:
        #     metrics_dict[metric] = dict()
        #     for class_idx, disease_label in enumerate(self.disease_label):
        #         metrics_dict[metric][disease_label] = dict()
        #         metrics_dict[metric][disease_label] = self.get_metrics(metric, class_idx, y_pred, y_true)
        #
        # return metrics_dict

    def get_metrics(self, metric, class_idx, y_pred, y_true):
        if metric == 'Accuracy': return accuracy_score(y_true[:, class_idx], y_pred[:, class_idx])
        elif metric == 'Precision': return precision_score(y_true[:, class_idx], y_pred[:, class_idx])
        elif metric == 'Recall': return recall_score(y_true[:, class_idx], y_pred[:, class_idx])
        elif metric == 'F1-Score': return f1_score(y_true[:, class_idx], y_pred[:, class_idx])

def calculate_top1_error(output, target) :
    _, rank1 = torch.max(output, 1)
    correct_top1 = (rank1 == target).sum().item()

    return correct_top1

def calculate_top5_error(output, target) :
    _, top5 = output.topk(5, 1, True, True)
    top5 = top5.t()
    correct5 = top5.eq(target.view(1, -1).expand_as(top5))

    for k in range(6):
        correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

    correct_top5 = correct_k.item()

    return correct_top5

def metrics(true, pred) :
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pred = (pred >= 0.5).astype(np.int_)

    true = np.asarray(true.flatten(), dtype=np.int64)
    pred = np.asarray(pred.flatten(), dtype=np.int64)

    acc = accuracy_score(true, pred)
    pre = precision_score(true, pred, average='macro')
    rec = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')
    iou = jaccard_score(true, pred, average='macro')

    return acc, f1, pre, rec, iou

def get_scores(y_true, y_pred, score_fun, nclasses=6):
    print(y_true.shape)
    print(y_pred.shape)
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_true[:, k], y_pred[:, k]) for k in range(nclasses)]]
    return np.array(scores).T

def specificity_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])
    return spc