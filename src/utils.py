import numpy as np
from numpy import random
from numpy import linalg


def add_noise(X, snr, axis=-1, seed=None):
    '''
    Add noise to a dataset.

    Parameters
    ----------
    X: numpy.ndarray,
        data to which add noise
    snr: float,
        signal-to-noise ratio 
    axis: int, (default -1)
        it specifies the axis of a along which to noise variance is computed.
    seed: {None, int}, optional
        A seed to initialize the random number generator (see detail in 
        numpy.random.default_rng)
    
    Retruns
    -------
    numpy.ndarry,
        data with noise added
    '''
    rng = random.default_rng(seed)
    norm_signal = np.mean(linalg.norm(X, axis=axis))
    norm_noise = norm_signal / 10**(snr / 20)
    noise = rng.normal(scale=norm_noise / np.sqrt(X.shape[axis]), size=X.shape)
    return X + noise

def compute_rsnr(x, x_hat, axis=-1):
    ''' Compute Reconstruction Signal-to-Noise Ratio '''
    norm_signal = linalg.norm(x, axis=axis)
    norm_noise = linalg.norm(x - x_hat, axis=axis)
    rsnr = 20 * np.log10(norm_signal / norm_noise)
    return rsnr