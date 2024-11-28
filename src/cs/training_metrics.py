import torch
import torch.nn as nn
import torch.nn.functional as F

def number_positive(output, z_true, th=0.5, reduce=True):
    P = torch.sum(z_true, dim=-1)
    if reduce:
        P = torch.mean(P)
    return P

def number_negative(output, z_true, th=0.5, reduce=True):
    N = torch.sum(1 - z_true, dim=-1)
    if reduce:
        N = torch.mean(N)
    return N

def true_negative(output, z_true, th=0.5, reduce=True):
    # Apply threshold to predictions
    z_pred= ((output> th).float())
    TN = torch.sum( (1 - z_pred) * (1 - z_true), dim=-1)
    if reduce:
        TN = torch.mean(TN)
    return TN

def tnr(output, z_true, th=0.5, reduce=True):
    TN = true_negative(output, z_true, th=th, reduce=False)
    N = number_negative(output, z_true, th=th, reduce=False)
    TNR = TN / N
    if reduce:
        TNR = torch.mean(TNR)
    # Return TNR
    return TNR

def true_positive(output, z_true, th=0.5, reduce=True):
    # Apply threshold to predictions
    z_pred = ((output > th).float())
    TP = torch.sum(z_pred * z_true, dim=-1)
    if reduce:
        TP = torch.mean(TP)
    # Return TP
    return TP

def tpr(output, z_true, th=0.5, reduce=True):
    TP = true_positive(output, z_true, th=th, reduce=False)
    P = number_positive(output, z_true, th=th, reduce=False)
    TPR = TP / P
    if reduce:
        TPR = torch.mean(TPR)
    # Return TPR
    return TPR

def accuracy(output, z_true, th=0.5, reduce=True):
    
    TP = true_positive(output, z_true, th=th, reduce=False)
    TN = true_negative(output, z_true, th=th, reduce=False)
    P = number_positive(output, z_true, th=th, reduce=False)
    N = number_negative(output, z_true, th=th, reduce=False)

    ACC = (TP + TN) / (P + N)

    if reduce:
        ACC = torch.mean(ACC)
    # Return accuracy
    return ACC

def rsnr(x_pred, x_true, reduce=False):
    ''' Compute Reconstruction Signal-to-Noise Ratio '''
    norm_signal = torch.norm(x_true, dim=-1)
    norm_noise = torch.norm(x_true - x_pred, dim=-1)
    rsnr = 20 * torch.log10(norm_signal / norm_noise)
    if reduce:
        rsnr = torch.mean(rsnr)
    return rsnr

def compute_rsnr(cs):
    def metric(output, x_true, th=0.5, reduce=True):
        y_true = cs.encode(x_true.cpu().numpy())  # Convert to NumPy
        z_pred = (output > th).cpu().numpy().astype(bool) # Convert to NumPy

        # signal reconstruction
        x_pred = torch.empty_like(x_true)  # Create a tensor with the same shape as x_true
        for i in range(output.size(0)):  # Iterate over batch size
            x_pred[i] = torch.tensor(cs.decode_with_support(y_true[i], z_pred[i]))  # Convert back to PyTorch tensor
        
        return rsnr(x_true, x_pred, reduce)
    return metric

def compute_metrics(output, z_true, th=0.5):
    return {
        'P': number_positive(output, z_true, th).item(),
        'TP': true_positive(output, z_true, th).item(), 
        'TPR': tpr(output, z_true, th).item(),
        'TNR': tnr(output, z_true, th).item(),
        'ACC': accuracy(output, z_true, th).item(),
        }

def update_metrics(metrics_accumulator, batch_metrics):
    for key in batch_metrics.keys():
        metrics_accumulator[key] += batch_metrics[key]
    return metrics_accumulator