import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def perf_evaluate(y_actual, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred).ravel()
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, y_pred)

    print("===== metrics =====")
    print("confusion matrix: TP {}, FP {}, TN {}, FN {}".format(tp, fp, tn, fn))
    print("precision: {:>2.4f}%".format(precision * 100))
    print("recall   : {:>2.4f}%".format(recall * 100))
    print("accuracy : {:>2.4f}%".format(accuracy * 100))


if __name__ == '__main__':
    y_true = [1, 0, 0, 1]
    y_pred = [1, 0, 0, 0]
    perf_evaluate(y_true, y_pred)