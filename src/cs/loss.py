import torch
import torch.nn as nn
import torch.nn.functional as F

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
    loss = torch.mean(loss, dim=-1)
    if reduce:
        loss = torch.mean(loss)
    return loss

