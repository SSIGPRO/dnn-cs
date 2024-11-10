import numpy as np
from numpy import linalg
from numpy import random

import warnings


def compute_localization(X):
    ''' 
    Compute Localization of a correlation matrix, where localization is defined
    as:
        Loc(X) = tr(X @ X) - 1/n

    Parameters
    ----------
    X: (n, n) numpy.ndarray,
        correlation matrix

    Return
    ------
    float,
        localization
    '''
    n = X.shape[0]
    localization = np.trace(X @ X) / np.trace(X)**2 - 1 / n
    return localization

def solve_rakeness(corr, loc=.25):
    '''
    Returns the correlation matrix of the sensing matrix that solves the 
    Rakeness Optimization Problem for Gaussian random sensing sequences as 
    described in [1].

    The rakeness optimization problem consists in
    C = arg max tr(C X)
    such that:
        - Loc(C) <= tau * Loc(X)
        - C semi-definite positive matrix
        - trace(C) = n
    where Loc computes the localization of a matrix as define in [1], and tr
    computes the trace of a matrix.

    [1] M. Mangia, R. Rovatti and G. Setti, "Rakeness in the Design of 
    Analog-to-Information Conversion of Sparse and Localized Signals," 
    in IEEE Transactions on Circuits and Systems I: Regular Papers, 
    vol. 59, no. 5, pp. 1001-1014, May 2012, doi: 10.1109/TCSI.2012.2191312

    Parameters
    ----------
    corr: (n, n) numpy.ndarray,
        signal correlation matrix
    loc: float, optional (default .25)
        rakeness scaling factor that controls the localization of the
        output correlation matrix. When loc is 0 output correlation is white, 
        while loc is 1 returns a correlation matrix equal to input correlation.
    
    Return
    ------
    (n, n) numpy.ndarray,
        correlation matrix of the sensing sequences
    
    '''
    n = corr.shape[0]  # signal dimension
    corr = corr / np.trace(corr)  # normalization to unit-energy

    # eigendecomposition of the input correlation matrix
    w, v = linalg.eigh(corr)
    w, v = np.abs(w)[::-1], v[..., ::-1]

    # apply solution algorithm
    loc = 1/n + loc * np.sum((w - 1/n)**2)
    for k in range(n, 1, -1):
        w_sum = np.sum(w[:k])
        num = k * w - w_sum
        den = np.sqrt((np.sum(w[:k]**2) - 1/k * w_sum**2) / (loc - 1/k))
        lmbda = 1/k * (1 + num/den)
        if np.min(lmbda) >= 0:
            break
    else:
        lmbda = np.eye(n)[0]

    out_corr = v[:, :k] @ np.diag(lmbda[:k]) @ v[:, :k].T
    return out_corr


def generate_gaussian_sequences(m, corr, seed=None):
    '''
    Generate Gaussian sensing sensing sequences with a given correlation matrix

    Parameters
    ----------
    m: int,
        number of sequences.
    corr: (n, n) numpy.ndarray,
        correlation matrix characterizing the sensing sequences.
    seed: {None, int, ...}, optional (default None)
        seed to initialize the random number generator 
        `numpy.random.default_rng`. If None, then fresh, unpredictable entropy 
        will be pulled from the OS.

    Returns
    -------
    (m, n) numpy.ndarray,
        Sensing sequences
    '''

    rng = random.default_rng(seed)  # random number generator

    # eigendecomposition of the correlation matrix 
    w, v = linalg.eigh(corr)
    w, v = np.abs(w)[::-1], v[..., ::-1]

    # rank of corr as the number of eigenvalues significantly grater than 0
    k = np.sum(~((w <= 0) | np.isclose(w, 0, atol=np.finfo(np.float64).eps)))

    # generate sensing matrix
    G = rng.normal(size=(m, k))  # generate white sequences
    A = G @ np.diag(np.sqrt(w[:k])) @ v.T[:k]  # color white sequences

    return A

def generate_antipodal_sequences(m, corr, seed=None):
    '''
    Generate Gaussian sensing sensing sequences with a given correlation matrix

    Parameters
    ----------
    m: int,
        number of sequences.
    corr: (n, n) numpy.ndarray,
        correlation matrix characterizing the sensing sequences.
    seed: {None, int, ...}, optional (default None)
        seed to initialize the random number generator 
        `numpy.random.default_rng`. If None, then fresh, unpredictable entropy 
        will be pulled from the OS.

    Returns
    -------
    (m, n) numpy.ndarray,
        Sensing sequences
    '''

    n = corr.shape[0]
    corr_norm = n * corr / np.trace(corr)  # normalize correlation matrix

    corr_dist = np.sin(np.pi /2 * corr_norm)  # correlation pre-distortion

    # check if predistortion can be applied
    w, _ = linalg.eigh(corr_dist)
    if np.min(np.real(w)) < 0:
        corr_dist = corr_norm
        warnings.warn('Correlation cannot be pre-distorted')
    
    # generate sensing sequences with pre-distorted correlation
    A = generate_gaussian_sequences(m, corr_dist, seed=seed)
    A = np.sign(A).astype(int)  # turn gaussian sequences to antipodal

    return A