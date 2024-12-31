import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.ae import AE
from wombats.detectors._base import Detector

class AEDetector(Detector):

    def __init__(self, n, m, model_path, batch_size=100, mode='self-assessment', gpu=None):
        
        self.n = n
        self.m = m

        # init the autoencoder
        self.ae = AE(self.n, self.m)
        if gpu is not None :
            self.gpu = gpu
            self.ae.to(self.gpu) # move the network to specified device
        else:
            self.gpu = None
        self.model_path = model_path
        # batch_size for evaluation
        self.batch_size = batch_size
        self.mode = mode
            
    def fit(self, *args, **kwargs):
        # load trained DEC model
        self.ae.load_state_dict(torch.load(self.model_path, weights_only=True))
        return self
    
    def score(self, X_test):
        self.ae.eval()
        dataset = TensorDataset(torch.from_numpy(X_test).float())  # Create a dataset from the tensor
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.mode=='autoencoder':
            # reconstruct data
            Xhat = torch.empty(X_test.shape)
            
            for batch_idx, X_batch in enumerate(loader):
                X_batch = X_batch[0]
                if self.gpu is not None:
                    X_batch = X_batch.to(self.gpu) 
                Xhat_batch = self.ae(X_batch)
                Xhat[self.batch_size*batch_idx: self.batch_size*(batch_idx + 1)] = Xhat_batch

            if self.gpu is not None:
                Xhat = Xhat.cpu().detach().numpy()
            else:
                Xhat = Xhat.detach().numpy()
            # estiamte the difference between the reconstructed data and data
            score = np.sqrt(np.sum((Xhat - X_test)**2, axis = -1))

        elif self.mode=='self-assessment':
            # encode reconstructed test data
            Y = torch.empty((X_test.shape[0], self.m))
            Yhat = torch.empty((X_test.shape[0], self.m))
            # reconstruct data
            for batch_idx, X_batch in enumerate(loader):
                X_batch = X_batch[0]
                if self.gpu is not None:
                    X_batch = X_batch.to(self.gpu) 
                Y_batch = self.ae.encode(X_batch)
                Xhat_batch = self.ae.decode(Y_batch)
                Yhat_batch = self.ae.encode(Xhat_batch)
                Y[self.batch_size*batch_idx: self.batch_size*(batch_idx + 1)] = Y_batch
                Yhat[self.batch_size*batch_idx: self.batch_size*(batch_idx + 1)] = Yhat_batch

            if self.gpu is not None:
                Yhat = Yhat.cpu().detach().numpy()
                Y = Y.cpu().detach().numpy()
            else:
                Yhat = Yhat.detach().numpy()
                Y = Y.detach().numpy()
            score = np.sqrt(np.sum((Yhat - Y)**2, axis = -1))

        else:
            raise ValueError(f'mode "{self.mode}" not supported')
        
        return score