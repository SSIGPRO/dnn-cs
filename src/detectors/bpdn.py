import numpy as np
from cs.decoders import BasisPursuitDenoise
from wombats.detectors._base import Detector

class BPDNDetector(Detector):

    def __init__(self, cs, mode='self-assessment'):
        assert isinstance(cs.decoder, BasisPursuitDenoise), "This detector requires a CS instance initialized with 'BasisPursuitDenoise' decoder"
        self.cs = cs
        self.n = cs.n
        self.m = cs.m
        self.mode = mode
       
    def fit(self, *args, **kwargs):
        return self
    
    def score(self, X_test):
        # encode test data
        Y = self.cs.encode(X_test)
        Xhat = self.cs.decode(Y)

        if self.mode=='autoencoder':
            # estiamte the difference between the reconstructed data and data
            score = np.sqrt(np.sum((Xhat - X_test)**2, axis = -1))

        elif self.mode=='self-assessment':
            # encode reconstructed test data
            Yhat = self.cs.encode(Xhat)
            # estiamte the difference between the encoded reconstructed data and encoded data 
            score = np.sqrt(np.sum((Yhat - Y)**2, axis = -1))
        
        return score