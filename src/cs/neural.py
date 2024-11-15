import torch
import torch.nn as nn
import torch.nn.functional as F

class TSOC(nn.Module):

    def __init__(self, n, m):
        super(TSOC, self).__init__()
        self.n = n
        self.m = m
        self.fc1 = nn.Linear(self.m, 2*self.n)
        self.fc2 = nn.Linear(2*self.n, 2*self.n)
        self.fc3 = nn.Linear(2*self.n, self.n)
        self.fc4 = nn.Linear(self.n, self.n)

    def forward(self, y):
        
        z = F.relu(self.fc1(y))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        output = F.sigmoid(self.fc4(z))
        return output
    
class ClippedCrossEntropyLoss(nn.Module):
    
    def __init__(self, epsilon=1e-7, reduction='mean'):
        super(ClippedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, y_pred, y_true):
        # Clip the predicted values to avoid log(0), so y_pred is limited between epsilon and 1-epsilon
        y_pred = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute the log values
        log_y_pred = torch.log(y_pred)
        log_1_minus_y_pred = torch.log(1 - y_pred)
        log_epsilon = torch.log(torch.tensor(self.epsilon))
        log_1_minus_epsilon = torch.log(torch.tensor(1 - self.epsilon))
        
        # Compute the clipped log values
        clipped_log_y_pred = torch.minimum(log_1_minus_epsilon, torch.maximum(log_epsilon, log_y_pred))
        clipped_log_1_minus_y_pred = torch.minimum(log_1_minus_epsilon, torch.maximum(log_epsilon, log_1_minus_y_pred))
        
        # Compute the loss for each component
        loss_pos = torch.sum(y_true * clipped_log_y_pred, dim=-1)
        loss_neg = torch.sum((1 - y_true) * clipped_log_1_minus_y_pred, dim=-1)
        
        # Sum the losses
        loss = loss_pos + loss_neg
        
        # Apply reduction (mean, sum, or none)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
def multiclass_loss_alpha(y_pred, y_true, alpha=0.5, reduce=True):
    myEPS = 1e-5
    # Clip the predicted values to avoid log(0)
    y_pred = torch.clamp(y_pred, myEPS, 1 - myEPS)
    
    # Compute the loss
    loss = -(1 - alpha) * (1 - y_true) * torch.log(1 - y_pred) - alpha * y_true * torch.log(y_pred)
    if reduce:
        loss = torch.mean(loss)
    return loss

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