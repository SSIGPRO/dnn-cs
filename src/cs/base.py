import multiprocessing as mp

import numpy as np
from numpy import random
from scipy import linalg

from .decoders import BasisPursuit, BasisPursuitDenoise
from .decoders import SupportOracle
from .rakeness import solve_rakeness 
from .rakeness import generate_gaussian_sequences, generate_antipodal_sequences

class CompressedSensing:
    """
    Compressed Sensing System

    Attributes
    ----------
    n: int,
        output dimension
    m: int,
        input dimension (number of measurements)
    A: (m, n) numpy.ndarray,
        sensing matrix
    D: (n, n) numpy.ndarray or None,
        sparsity basis 
    decoder: cs.Decoder,
        compressed sensing decoder

    Methods
    -------
    encode(x):
        compress input signal `x` into a mesurement vector `x`

    decode(y):
        reconstruct the signal from the measurement vector `y`.
    """

    def __init__(self, A, D=None, decoder=None, **kwargs):
        """
        Compressed Sensing System

        Parameters
        ----------
        A: (m, n) numpy.ndarray,
            sensing matrix
        D: (n, n) numpy.ndarray or None,
            sparsity basis 
        decoder: {'BP', 'BPDN', 'SO'},
            compressed sensing decoder
        kwargs: 
            decoder keyword arguments
        """
        
        self.m, self.n = A.shape
        self.A = A
        if D is None:
            D = np.eye(self.n)
        self.D = D
        self.B = self.A @ self.D

        if decoder is None:
            decoder = 'BPDN'
        
        if decoder == 'BP':
            self.decoder = BasisPursuit(self.B, self.D)
        elif decoder == 'BPDN':
            self.decoder = BasisPursuitDenoise(self.B, self.D, **kwargs)
        elif decoder == 'SO': 
            self.decoder = SupportOracle(self.B, self.D, **kwargs)
        else:
            raise ValueError(f'decoder {decoder} not supoorted')


    def encode(self, x):
        y = x @ self.A.T
        return y
    
    def decode(self, y):
        return self.decoder.decode(y)
    
# class CompressedSensing:

#     def __init__(self, A, D=None):
        
#         self.m, self.n = A.shape
#         self.A = A
#         if D is None:
#             D = np.eye(self.n)
#         self.D = D
#         self.B = self.A @ self.D

#     def encode(self, x):
#         y = x @ self.A.T
#         return y
    
#     def decode(self, y, s=None, processes=None):
#         if s is None:
#             # TODO: support estimator
#             # # s = self.estimate_support(y)
#             pass
        
#         if y.ndim == 1:
#             x_hat = self.decode_with_support(y, s)
#         elif y.ndim == 2:
#             if processes is None:
#                 x_hat = np.empty(s.shape, dtype=float)
#                 for i, (_y, _s) in enumerate(zip(y, s)):
#                     x_hat[i, :] = self.decode_with_support(_y, _s)
#             else:
#                 with mp.Pool(processes) as pool:
#                     x_hat = pool.starmap(
#                         self.decode_with_support, zip(y, s), chunksize=100)
#                 x_hat = np.stack(x_hat)
#         return x_hat
    
#     def decode_with_support(self, y, s):
#         try:
#             x_hat = y @ linalg.pinv(self.B[:, s]).T @ self.D[:, s].T
#         except linalg.LinAlgError as e:
#             x_hat = np.nan * np.ones(self.n, dtype=float)
#         return x_hat


def generate_sensing_matrix(
        shape, 
        mode='standard', 
        antipodal=False, 
        orthogonal=False,
        correlation=None,
        loc=.25, 
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
    loc: float, optional (default .25)
        rakeness scaling factor that controls the localization of the
        output correlation matrix. Effective only with rakeness mode. 
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
        corr = solve_rakeness(correlation, loc=loc)

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


