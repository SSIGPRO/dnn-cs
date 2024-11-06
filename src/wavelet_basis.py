import numpy as np
import pywt


def wavelet_basis(n, wavelet='haar', level=0):
    '''
    Generate Wavelet Basis

    Parameters
    ----------
    n: int,
        Basis dimension. It must be a power of 2.
    wavelet: pywt.Wavelet or str (default 'haar')
        wavelet (pywt.Wavelet or str)
    level: int (default 0),
        coarsest decomposition level.

    Returns
    -------
    (n, n) numpy.ndarray,
        Orthogonal matrix
    '''

    basis = np.empty((n, n))
    for i in range(n):
        basis[i] = FWT_PO(np.eye(n)[i], level=level, wavelet=wavelet)

    return basis


def FWT_PO(x, level=0, wavelet='db1'):
    '''
    Forward Wavelet Transform (periodized, orthogonal)
    It follows MATLAB implementation as in 
    https://viewer.mathworks.com›files›FWT_PO.m

    Parameters
    ----------
    x: 1D numpy.ndarray,
        input vector. It must have a length that is a power of 2.
    level: int (default 0),
        coarsest decomposition level
    wavelet: pywt.Wavelet or str (default 'haar')
        wavelet (pywt.Wavelet or str)

    Returns
    -------
    1D numpy.ndarray,
        wavelet coefficients arranged as a 1D vector starting from the
        approximation coefficients, then the detail coefficients with
        increasing level. (e.g., with len(x)=8 and level=1, output is
        [cA0, cA1, cD1_0, cD1_1, cD2_0, cD2_1, cD2_2, cD2_3])
    '''
    
    if not type(wavelet) is pywt._extensions._pywt.Wavelet:
        wavelet = pywt.Wavelet(wavelet)

    n = len(x)                    # input length
    L = int(np.ceil(np.log2(n)))  # max number of levels
    if 2**L != n:  # check input length
        raise ValueError('`x` must be power-of-2 length, instead len(x)={n}')
    
    wcoef = np.zeros(n, dtype=float)  # wavelet coefficients
    approx = x.copy()
    for i in range(L, level, -1):
        details = down_dyad_hi(approx, wavelet)  # details coefficients
        approx = down_dyad_lo(approx, wavelet)  # approx coefficients
        wcoef[2**(i-1):2**i] = details
    wcoef[:2**level] = approx
    return wcoef

def down_dyad_hi(x, wavelet):
    '''
    Hi-Pass Downsampling operator (periodized)
    It follows MATLAB implementation as in 
    https://viewer.mathworks.com›files›DownDyadHi.m
    '''
    d = iconv(wavelet.rec_hi, x)
    return d[1::2]

def down_dyad_lo(x, wavelet):
    '''
    Lo-Pass Downsampling operator (periodized)
    It follows MATLAB implementation as in 
    https://viewer.mathworks.com›files›DownDyadLo.m
    '''
    d = aconv(wavelet.rec_lo, x)
    return d[::2]

def iconv(f, x):
    '''
    Convolution Tool for Two-Scale Transform
    It follows MATLAB implementation as in 
    https://viewer.mathworks.com›files›iconv.m
    '''
    n, p = len(x), len(f)
    if p <= n:
        xpadded = np.concatenate((x[-p:], x))
    else:
        xpadded = np.tile(x, 2 + p//n)[-p-n:]
    y = np.convolve(f, xpadded)[p:-(p - 1)]
    return y

def aconv(f, x):
    '''
    Convolution Tool for Two-Scale Transform
    It follows MATLAB implementation as in 
    https://viewer.mathworks.com›files›aconv.m
    '''
    n, p = len(x), len(f)
    if p < n:
        xpadded = np.concatenate((x, x[:p]))
    else:
        xpadded = np.tile(x, 2 + p//n)[:n + p]
    y = np.convolve(f, xpadded)[p-1:-p]
    return y