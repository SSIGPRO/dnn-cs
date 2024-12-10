import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import linalg
from cs.wavelet_basis import wavelet_basis
from cs.utils import reconstructor
from cs import CompressedSensing, generate_sensing_matrix
from models.tsoc import TSOC
from wombats.detectors._base import Detector

class TSOCDetector(Detector):

    def __init__(self, cs, model_path, batch_size=100, mode='self-assessment', threshold=0.5,  gpu=None):
        
        self.cs = cs
        self.n = cs.n
        self.m = cs.m
        self.threshold = threshold

        # init TSOC
        self.tsoc = TSOC(cs.n, cs.m)
        if gpu is not None :
            self.gpu = gpu
            self.tsoc.to(self.gpu) # move the network to specified device
        else:
            self.gpu = None
        self.model_path = model_path
        # batch_size for evaluation
        self.batch_size = batch_size
        self.mode = mode
            
    def fit(self, *args, **kwargs):
        # load trained TSOC model
        self.tsoc.load_state_dict(torch.load(self.model_path, weights_only=True))
        return self
    
    def score(self, X_test):
        # encode test data
        Y = self.cs.encode(X_test)

        # compute support estimation
        self.tsoc.eval()  
        O = torch.empty(X_test.shape)
        dataset = TensorDataset(torch.from_numpy(Y).float())  # Create a dataset from the tensor
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        for batch_idx, Y_batch in enumerate(loader):
            Y_batch = Y_batch[0]
            if self.gpu is not None:
                Y_batch = Y_batch.to(self.gpu) 
            O_batch = self.tsoc(Y_batch)
            O[self.batch_size*batch_idx: self.batch_size*(batch_idx + 1)] = O_batch

        if self.gpu is not None:
            O = O.cpu().detach().numpy()
        else:
            O = O.detach().numpy()

        # calculate supports
        Zhat = O > self.threshold

        if self.mode == 'sparsity':
            score = np.mean(O, axis=-1)

        elif self.mode == 'sparsity-threshold':
            score = np.mean(Zhat, axis=-1)

        elif self.mode in ['autoencoder', 'self-assessment', 'self-assessment-complement', 'complement']:
            # reconstruct data
            Xhat = np.empty(X_test.shape)
            for i in range(X_test.shape[0]):
                Xhat[i] = self.cs.decode_with_support(Y[i], Zhat[i])

            if self.mode=='autoencoder':
                # estiamte the difference between the reconstructed data and data
                score = np.sqrt(np.sum((Xhat - X_test)**2, axis = -1))

            elif self.mode=='self-assessment':
                # encode reconstructed test data
                Yhat = self.cs.encode(Xhat)
                # estiamte the difference between the encoded reconstructed data and encoded data 
                score = np.sqrt(np.sum((Yhat - Y)**2, axis = -1))

            elif 'complement' in self.mode:
                # find the orthogonal complement of A
                Q, _ = np.linalg.qr(self.cs.A.T, mode='complete')
                Ac = Q[:, self.m:].T

                # estimate the energy along the complement
                Ybar = Xhat @ Ac.T
                score = np.mean(Ybar**2, axis = -1)
            
                if self.mode=='self-assessment-complement':
                    Yhat = self.cs.encode(Xhat)
                    # combine two scores
                    score = score + np.mean((Yhat - Y)**2, axis = -1) 

        else:
            raise ValueError(f'mode "{self.mode}" not supported')
        
        return score