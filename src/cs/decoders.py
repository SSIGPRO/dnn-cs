from abc import ABC, abstractmethod

from itertools import product

import multiprocessing as mp
import numpy as np

from scipy import linalg
from spgl1 import spg_bp, spg_bpdn


class Decoder(ABC):
    """
    Compressed Sensing Decoder

    Attributes
    ----------
    n: int,
        output dimension
    m: int,
        input dimension (number of measurements)
    B: (m, n) numpy.ndarray,
        measurement matrix or effective sensing matrix
    D: (n, n) numpy.ndarray or None,
        sparsity basis 
    processes: int or None,
        number of threads for multiprocessing.

    Methods
    -------
    decode(y):
        reconstruct the signal from the measurement vector `y`.
    """

    def __init__(self, B, D=None, processes=None):
        """
        Initialize a Decoder instance.

        Parameters:
        ----------
        B: (m, n) numpy.ndarray,
            measurement matrix or effective sensing matrix
        D: (n, n) numpy.ndarray or None,
            sparsity basis 
        processes: int or None,
            number of threads for multiprocessing.
        """
        self.B = B
        self.m, self.n = B.shape
        self.D = D
        self.processes = processes

    def decode(self, y):
        """
        Reconstruct the signal from the measurements.

        Parameters:
        ----------
        y: (m, ) or (N, m) numpy.ndarray,
            measurement vector or vectors
        """
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=object)

        if y.ndim == 1:
            x_hat = self._decode(y)
        
        elif y.ndim == 2:
            if self.processes is None:
                x_hat = np.empty((len(y), self.n), dtype=float)
                for i, _y in enumerate(y):
                    x_hat[i, :] = self._decode(_y)
            else:
                with mp.Pool(self.processes) as pool:
                    x_hat = pool.starmap(self._decode, y, chunksize=100)
                x_hat = np.stack(x_hat)
        
        return x_hat

    @abstractmethod
    def _decode(self, y):
        pass


class BasisPursuit(Decoder):
    """
    Basis Pursuit Decoding Algorithm

    Wrapper over `spg_bp` function of the `spgl1` library
    https://spgl1.readthedocs.io/en/latest/index.html
    """
    
    def _decode(self, y):
        x_hat = spg_bp(self.B, y)[0]
        if self.D is not None:
            x_hat = self.D @ x_hat
        return x_hat
    

class BasisPursuitDenoise(Decoder):
    """
    Basis Pursuit Denoise Decoding Algorithm

    Wrapper over `spg_bpdn` function of the `spgl1` library
    https://spgl1.readthedocs.io/en/latest/index.html
    """

    def __init__(self, B, D=None, sigma=1e-3, processes=None):
        super().__init__(B, D=D, processes=processes)
        self.sigma = sigma
    
    def _decode(self, y, sigma=None):
        if sigma is None:
            sigma = self.sigma

        x_hat = spg_bpdn(self.B, y, sigma=self.sigma)[0]

        if self.D is not None:
            x_hat = self.D @ x_hat
        return x_hat


class SupportOracle(Decoder):
    """
    Decoder based on a Support Oracle

    [1] M. Mangia et al, "Deep Neural Oracles for Short-Window Optimized 
        Compressed Sensing of Biosignals," in IEEE Transactions on Biomedical 
        Circuits and Systems, vol. 14, no. 3, pp. 545-557, June 2020, 
        doi: 10.1109/TBCAS.2020.2982824
    [2] L. Prono et al, "Deep Neural Oracle With Support Identification in the 
        Compressed Domain," in IEEE Journal on Emerging and Selected Topics in 
        Circuits and Systems, vol. 10, no. 4, pp. 458-468, Dec. 2020, 
        doi: 10.1109/JETCAS.2020.3039731
    """
    
    def _decode(self, y):
        x_hat = self._decode_with_support(*y)
        return x_hat
    
    def _decode_with_support(self, y, s):
        try:
            x_hat = linalg.pinv(self.B[:, s]) @ y
            if self.D is not None:
                x_hat = self.D[:, s] @ x_hat

        except linalg.LinAlgError:
            x_hat = np.nan * np.ones(self.n, dtype=float)
        return x_hat
    

# class TrainedSupportOracle(SupportOracle):

#     def __init__(self, B, D=None, path=None, processes=None):
#         super().__init__(B, D=D, processes=processes)
#         self.oracle = load_path(path)
    
#     def _decode(self, y):
#         s = self.oracle(y)
#         x_hat = self._decode_with_support(y, s)
#         return x_hat
    
# class DecoderUNet(Decoder):      
    
#     def __init__(self, B, D=None, path=None, processes=None):
#         super().__init__(B, D=D, processes=processes)
#         self.model = load_path(path)

#     def _decode(self, y):
#         x_hat = self.model(y)
#         return x_hat