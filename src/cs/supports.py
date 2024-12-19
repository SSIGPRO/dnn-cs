import numpy as np
from scipy import linalg

from .utils import compute_rsnr


def find_support_GR(x, D, eta=0.99):
    '''
    Identify signal support with GR algortithm which identify the support by
    considering the smallest subsets of compontest that retains a given 
    energy fraction.

    Parameters
    ----------
    x: (n, ) numpy.ndarray,
        input signal
    D: (n, n) numpy.ndarray,
        sparsity basis
    eta: float, optional (default 0.99)
        energy fraction retained by the components selected by the support.

    Returns
    -------
    (n, ) numpy.ndarray,
        support of the input x.
    '''

    xi = D.T @ x  # sparse representation of the input

    xi2 = xi**2
    xi2 = xi2 / np.sum(xi2) # input squared and normalized to unit-energy

    i_sort = np.argsort(xi2)[::-1]  # sort components in descending order
    # index of the smallest component in the support
    i_th = np.argmax(np.cumsum(xi2[i_sort]) > eta)
    th = xi2[i_sort[i_th]]  # magnitude of the smallest component in the support
    # support as all components greater or equal the smallest in the support
    z = xi2 >= th

    return z


def find_support_TSOC(x, cs):
    '''
    Identify signal support with TSOC algortithm which identify the support as 
    the one maximizing the quality of reconstruction.

    Parameters
    ----------
    x: (n, ) numpy.ndarray,
        input signal
    cs: CompressedSensing,
        compressed sensing system

    Returns
    -------
    (n, ) numpy.ndarray,
        support of the input x.
    '''
    
    y = cs.encode(x)  # measurements
    xi = cs.D.T @ x   # sparse representation of the input

    # empty support that is filled one element at every iteration
    z = np.zeros(cs.n, dtype=bool)
    # indexes of xi sorted by magnitude in descending order
    argsort_xi = np.argsort(np.abs(xi))[::-1] 
    rsnr = np.empty(cs.m, dtype=float)  # stores rsnr at each iteration
    for i in range(cs.m):
        z[argsort_xi[i]] = True  # include element in the support
        x_hat = cs.decoder._decode_with_support(y, z) # reconstructed input
        rsnr[i] = compute_rsnr(x, x_hat)
    k = 1 + np.argmax(rsnr) # number of components to include in the support

    # build best support
    z = np.zeros(cs.n, dtype=bool)
    z[argsort_xi[:k]] = True

    return z

def find_support_TSOC2(x, cs):
    '''
    Identify signal support with TSOC algortithm which identify the support as 
    the one maximizing the quality of reconstruction.

    Parameters
    ----------
    x: (n, ) numpy.ndarray,
        input signal
    cs: CompressedSensing,
        compressed sensing system

    Returns
    -------
    (n, ) numpy.ndarray,
        support of the input x.
    '''
    
    y = cs.encode(x)  # measurements

    _m = min(cs.n, cs.m + 4)  # compute RSNR up to m+4 elements
    rsnr1 = np.empty(_m, dtype=float)  # stores RSNR of the outer loop
    s1 = np.zeros(cs.n, dtype=bool)  # support of the outer loop
    # stores support elements index in descending order
    s_idx = -np.ones(_m, dtype=int)
    for i in range(_m):
        rsnr2 = -np.inf * np.ones(cs.n, dtype=float)  # RSNR of the inner loop
        # iterate over the indexes of the null elements of the support
        for j in np.where(~s1)[0]:  
            s2 = s1.copy()  # support of the inner loop
            s2[j] = True
            x_hat = cs.decoder._decode_with_support(y, s2)
            rsnr2[j] = compute_rsnr(x, x_hat)
        # find element that maximizes RSNR and update support
        s_idx[i] = np.argmax(rsnr2)
        s1[s_idx[i]] = True
        rsnr1[i] = rsnr2[s_idx[i]]

    # build support setting the elements corresponding to highest RSNR
    s = np.zeros(cs.n, dtype=bool)
    s[s_idx[:np.argmax(rsnr1) + 1]] = True

    return s