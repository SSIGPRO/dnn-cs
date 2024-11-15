import numpy as np
from numpy import linalg


def compute_rsnr(x, x_hat, axis=-1):
    ''' Compute Reconstruction Signal-to-Noise Ratio '''
    norm_signal = linalg.norm(x, axis=axis)
    norm_noise = linalg.norm(x - x_hat, axis=axis)
    rsnr = 20 * np.log10(norm_signal / norm_noise)
    return rsnr

def reconstructor(dnn_output, y, A, D, th=0.5):
    ''' Recover the singal based on the oracle prior '''
    B = A @ D
    zhat = dnn_output > th
    BTS = linalg.pinv(B[:, zhat])
    xihat = BTS @ y
    xhat = D[:, zhat] @ xihat
    return xhat
