import numpy as np
import torch
from scipy.signal import  welch
from scipy.fft import dct
from spectrum import burg

def compute_rsnr(x, x_hat, axis=-1):
    ''' Compute Reconstruction Signal-to-Noise Ratio '''
    norm_signal = np.linalg.norm(x, axis=axis)
    norm_noise = np.linalg.norm(x - x_hat, axis=axis)
    rsnr = 20 * np.log10(norm_signal / norm_noise)
    return rsnr

def sliding_window(X, n, stride):
    """
    Generate overlapping windows using a sliding approach with variable stride.
    Ensures the output shape is (N, n).
    """
    return np.array([X[i:i + n] for i in range(0, X.shape[0] - n + 1, stride)])

def select_random_windows(X_windows, M, seed_training):
    """
    Select M random windows from the dataset, using a fixed random seed.
    """
    np.random.seed(seed_training)  # Set random seed for reproducibility
    N = X_windows.shape[0]
    indices = np.random.choice(N, M, replace=False)  # Random unique indices
    return X_windows[indices], indices

def pca(Corr, X, m):
    _, V = np.linalg.eigh(Corr)
    V = V[:, ::-1]
    U = V[:, :m]
    Y = X @ U
    Xhat = Y @ U.T
    return Xhat

def recon_reencode(data, ae, device):
    """Processes data through an autoencoder and returns numpy arrays."""
    data = torch.from_numpy(data).float().to(device)  # Move data to device
    latent = ae.encode(data)  # Encode input data
    reconstructed = ae.decode(latent)  # Decode to reconstruct
    latent_reconstructed = ae.encode(reconstructed)  # Encode reconstructed data

    data, reconstructed, latent, latent_reconstructed = tuple(
        t.cpu().detach().numpy()
          for t in [data, reconstructed, latent, latent_reconstructed]
          )

    # Convert all tensors to numpy arrays
    return reconstructed, latent, latent_reconstructed

def estimate_spectrum(n, fs, data, K, resolution, method='burg'):
    data = sliding_window(data.flatten(), resolution*n, resolution*n)
    if method == 'dct':
        D = dct(np.eye(resolution*n), axis=0, norm='ortho')
        coeff = (data @ D)**2
        freqs = np.arange(resolution * n) * (fs / (2 * resolution * n))
    elif method == 'welch':
        window = 'boxcar'
        nperseg = resolution*n
        # noverlap = nperseg // 2
        noverlap = 0
        coeff = []
        for data_ in data:
            freqs, coeff_ = welch(data_, fs, window, nperseg, noverlap)
            coeff.append(coeff_)
        coeff = np.array(coeff)
    elif method == 'burg': 
        order = resolution*n // 50
        order = 30
        coeff = []
        freqs = np.linspace(0, fs / 2, resolution * n)
        for data_ in data:
            psd = burg.pburg(data_, order=order, NFFT=resolution*n, sampling = fs)
            freqs = np.array(psd.frequencies())
            coeff_ = psd.psd
            coeff.append(coeff_)
        coeff = np.array(coeff)
    coeff = coeff.reshape(-1, K//resolution, coeff.shape[1]).mean(axis=1)
    cc = np.mean(coeff, axis=0)
    return freqs, coeff, cc

def itakura_saito_divergence(P, Q, eps):
    """
    Compute the Itakura-Saito Divergence (ISD) between two power spectra.
    ISD(P || Q) = sum( P/Q - log(P/Q) - 1 )
    """
    if eps is None:
        eps = min(np.min(P), np.min(Q)) * 1e-3
    P = np.maximum(P, eps)  # Avoid division by zero
    Q = np.maximum(Q, eps)
    return np.mean(P / Q - np.log(P / Q) - 1)

def symmetrized_ISD(P, Q, eps=None):
    if eps is None:
        eps = min(np.min(P), np.min(Q)) * 1e-3
    P = np.maximum(P, eps)  # Avoid division by zero
    Q = np.maximum(Q, eps)
    return np.mean(P / Q - np.log(P / Q) + Q / P - np.log(P / Q) - 2)

def extract_subband_coefficients(spectrum, freqs, subbands):
    """
    Extract power in each frequency subband.
    """
    subband_coefficients = []
    band_indices = []
    # freqs = np.linspace(0, fs / 2, n // 2)
    # freqs = np.arange(n) * (fs / (2 * n))

    for fmin, fmax in subbands:
        band_indices_ = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        subband_coefficients.append(np.array(spectrum[band_indices_]))
        band_indices.append(band_indices_)
    return subband_coefficients, band_indices

def MSISSD(P, Q, subbands, freqs, eps=None):
    
    P_subband_coefficients, _ = extract_subband_coefficients(P, freqs, subbands)
    Q_subband_coefficients, _ = extract_subband_coefficients(Q, freqs, subbands)
     
    ISSD_subbands = np.array([symmetrized_ISD(P, Q, eps) for P, Q in zip(P_subband_coefficients, Q_subband_coefficients)])
    
    return ISSD_subbands