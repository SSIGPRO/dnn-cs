import torch
import torch.nn as nn
import torch.nn.functional as F

def number_positive(y_pred, y_true, th=0.5, reduce=True):
    P = torch.sum(y_true, dim=-1)
    if reduce:
        P = torch.mean(P)
    return P

def number_negative(y_pred, y_true, th=0.5, reduce=True):
    N = torch.sum(1 - y_true, dim=-1)
    if reduce:
        N = torch.mean(N)
    return N

def true_negative(y_pred, y_true, th=0.5, reduce=True):
    # Apply threshold to predictions
    y_pred = ((y_pred > th).float())
    TN = torch.sum( (1 - y_pred) * (1 - y_true), dim=-1)
    if reduce:
        TN = torch.mean(TN)
    return TN

def tnr(y_pred, y_true, th=0.5, reduce=True):
    TN = true_negative(y_pred, y_true, th=th, reduce=False)
    N = number_negative(y_pred, y_true, th=th, reduce=False)
    TNR = TN / N
    if reduce:
        TNR = torch.mean(TNR)
    # Return TNR
    return TNR

def true_positive(y_pred, y_true, th=0.5, reduce=True):
    # Apply threshold to predictions
    y_pred = ((y_pred > th).float())
    TP = torch.sum(y_pred * y_true, dim=-1)
    if reduce:
        TP = torch.mean(TP)
    # Return TP
    return TP

def tpr(y_pred, y_true, th=0.5, reduce=True):
    TP = true_positive(y_pred, y_true, th=th, reduce=False)
    P = number_positive(y_pred, y_true, th=th, reduce=False)
    TPR = TP / P
    if reduce:
        TPR = torch.mean(TPR)
    # Return TPR
    return TPR

def accuracy(y_pred, y_true, th=0.5, reduce=True):
    
    TP = true_positive(y_pred, y_true, th=th, reduce=False)
    TN = true_negative(y_pred, y_true, th=th, reduce=False)
    P = number_positive(y_pred, y_true, th=th, reduce=False)
    N = number_negative(y_pred, y_true, th=th, reduce=False)

    ACC = (TP + TN) / (P + N)

    if reduce:
        ACC = torch.mean(ACC)
    # Return accuracy
    return ACC

def compute_metrics(y_pred, y_true, th=0.5):
    return {
        'P': number_positive(y_pred, y_true, th).item(),
        'TP': true_positive(y_pred, y_true, th).item(), 
        'TPR': tpr(y_pred, y_true, th).item(),
        'TNR': tnr(y_pred, y_true, th).item(),
        'ACC': accuracy(y_pred, y_true, th).item()
        }

def update_metrics(metrics_accumulator, batch_metrics):
    for key in batch_metrics.keys():
        metrics_accumulator[key] += batch_metrics[key]
    return metrics_accumulator