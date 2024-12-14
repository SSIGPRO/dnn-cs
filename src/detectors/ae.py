import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.NNdecoder import NNdecoder
from wombats.detectors._base import Detector

class AEDetector(Detector):

    def __init__(self, cs, model_path, batch_size=100, mode='self-assessment', gpu=None):
        
        self.cs = cs
        self.n = cs.n
        self.m = cs.m

        # init DEC
        self.dec = NNdecoder(cs.n, cs.m)
        if gpu is not None :
            self.gpu = gpu
            self.dec.to(self.gpu) # move the network to specified device
        else:
            self.gpu = None
        self.model_path = model_path
        # batch_size for evaluation
        self.batch_size = batch_size
        self.mode = mode
            
    def fit(self, *args, **kwargs):
        # load trained DEC model
        self.dec.load_state_dict(torch.load(self.model_path, weights_only=True))
        return self
    
    def score(self, X_test):
        # encode test data
        Y = self.cs.encode(X_test)

        # reconstruct data
        self.dec.eval()  
        Xhat = torch.empty(X_test.shape)
        dataset = TensorDataset(torch.from_numpy(Y).float())  # Create a dataset from the tensor
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        for batch_idx, Y_batch in enumerate(loader):
            Y_batch = Y_batch[0]
            if self.gpu is not None:
                Y_batch = Y_batch.to(self.gpu) 
            Xhat_batch = self.dec(Y_batch)
            Xhat[self.batch_size*batch_idx: self.batch_size*(batch_idx + 1)] = Xhat_batch

        if self.gpu is not None:
            Xhat = Xhat.cpu().detach().numpy()
        else:
            Xhat = Xhat.detach().numpy()

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