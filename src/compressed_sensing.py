import multiprocessing as mp

import numpy as np
from numpy import random
from scipy import linalg

from utils import compute_rsnr
from rakeness import solve_rakeness 
from rakeness import generate_gaussian_sequences, generate_antipodal_sequences


class CompressedSensing:

    def __init__(self, A, D=None):
        
        self.m, self.n = A.shape
        self.A = A
        if D is None:
            D = np.eye(self.n)
        self.D = D
        self.B = self.A @ self.D

    def encode(self, x):
        y = x @ self.A.T
        return y
    
    def decode(self, y, s=None, processes=None):
        if s is None:
            # TODO: support estimator
            # # s = self.estimate_support(y)
            pass
        
        if y.ndim == 1:
            x_hat = self.decode_with_support(y, s)
        elif y.ndim == 2:
            if processes is None:
                x_hat = np.empty(s.shape, dtype=float)
                for i, (_y, _s) in enumerate(zip(y, s)):
                    x_hat[i, :] = self.decode_with_support(_y, _s)
            else:
                with mp.Pool(processes) as pool:
                    x_hat = pool.starmap(
                        self.decode_with_support, zip(y, s), chunksize=100)
                x_hat = np.stack(x_hat)
        return x_hat
    
    def decode_with_support(self, y, s):
        try:
            x_hat = y @ linalg.pinv(self.B[:, s]).T @ self.D[:, s].T
        except linalg.LinAlgError as e:
            x_hat = np.nan * np.ones(self.n, dtype=float)
        return x_hat
    


def generate_sensing_matrix(
        shape, 
        mode='standard', 
        antipodal=False, 
        orthogonal=False,
        correlation=None,
        tau=None, 
        seed=None):
    '''
    Generate sensing matrix for a Compressed Sensing (CS) encoder

    Parameters
    ----------
    shape: sequence of two ints (m, n),
        Shape of the sensing matrix. `n` is the signal dimension, while `m` is
        the number of measurements.
    mode: {'standard', 'rakeness'}, optional (default: 'standard')
        type of CS encoding. For rakeness mode to be effective, `correlation` 
        and `tau` arguments should be provided.
    antipodal: bool, optional (default False)
        whether the sensing matrix has real or antipodal (i.e., +1 or -1)
        values.
    orthogonal: bool, optional (default False)
        whether the sensing matrix is forced to be orthogonal or not.
    correlation: (n, n) numpy.ndarray, optional (default None)
        correlation matrix of the signal to be acquired. Effective only with
        rakeness mode. If None, Identity matrix is used which, however, is 
        equivalent to standard mode.
    tau: float, optional (default 1.0)
        rakeness factor that controls adaptation of the sensing sequences to
        the signal to be acquired.
    seed: {None, int, ...}, optional (default None)
        seed to initialize the random number generator 
        `numpy.random.default_rng`. If None, then fresh, unpredictable entropy 
        will be pulled from the OS.

    Returns
    -------
    (m, n) numpy.ndarray,
        Sensing matrix
    '''
    
    m, n = shape

    if mode == 'standard':
        rng = random.default_rng(seed)  # random number generator
        A = rng.normal(size=shape)  # generate gaussian matrix
        if antipodal:
            A = np.sign(A)  # turn real values into antipodal

    elif mode == 'rakeness':

        if correlation is None:
            # if not given, use identity matrix as correlation
            corr = np.eye(n)  

        # compute correlation of sensing sequences by solving rakeness
        # optimization problem
        corr = solve_rakeness(correlation, tau=tau)

        # generate gaussian/antipodal sensing sequences
        if antipodal:
            A = generate_antipodal_sequences(m, corr, seed)
        else:
            A = generate_gaussian_sequences(m, corr, seed)

    else:
        raise ValueError(f'mode "{mode}" not supported')
    
    # orthogonalize only if gaussian
    if not antipodal and orthogonal:
        A = linalg.orth(A.T).T

    return A


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
        # compute reconstructed signal by pseudo-inversion 
        x_hat = y @ linalg.pinv(cs.B[:, z]).T @ cs.D[:, z].T
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
            x_hat = cs.decode_with_support(y, s2)
            rsnr2[j] = compute_rsnr(x, x_hat)
        # find element that maximizes RSNR and update support
        s_idx[i] = np.argmax(rsnr2)
        s1[s_idx[i]] = True
        rsnr1[i] = rsnr2[s_idx[i]]

    # build support setting the elements corresponding to highest RSNR
    s = np.zeros(cs.n, dtype=bool)
    s[s_idx[:np.argmax(rsnr1) + 1]] = True

    return s