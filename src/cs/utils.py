import numpy as np
from scipy import linalg


def compute_rsnr(x, x_hat, axis=-1):
    ''' Compute Reconstruction Signal-to-Noise Ratio '''
    norm_signal = linalg.norm(x, axis=axis)
    norm_noise = linalg.norm(x - x_hat, axis=axis)
    rsnr = 20 * np.log10(norm_signal / norm_noise)
    return rsnr