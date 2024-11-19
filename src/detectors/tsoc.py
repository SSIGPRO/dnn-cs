import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cs.wavelet_basis import wavelet_basis
from cs.utils import reconstructor
from cs import CompressedSensing, generate_sensing_matrix
from models.tsoc import TSOC
from wombats.detectors._base import Detector

class TSOCDetector(Detector):

    def __init__(self, n, m, model_path, seed, batch_size=100, mode='self-assessment', basis='sym6', gpu=None):
        
        # # extract parameters from model's name
        # parts = model_name.split('_')
        # self.n = int([part for part in parts if 'n' in part][0].split('=')[1])
        # self.m = int([part for part in parts if 'm' in part][0].split('=')[1])
        # self.seed = int([part for part in parts if 'seed' in part][0].split('=')[1])

        self.n = n
        self.m = m
        self.seed = seed

        # init TSOC
        self.tsoc = TSOC(self.n, self.m)
        if gpu is not None :
            self.gpu = gpu
            self.tsoc.to(self.gpu) # move the network to specified device
        else:
            self.gpu = None
        self.model_path = model_path
        # init Compressed Sensing
        self.A = generate_sensing_matrix((self.m, self.n), seed=self.seed)
        self.D = wavelet_basis(self.n, basis, level=2)
        self.cs = CompressedSensing(self.A, self.D)
        # batch_size for evaluation
        self.batch_size = batch_size
        self.mode = mode
        self.basis = basis
            
    def fit(self, *args, **kwargs):
        # load trained TSOC model
        self.tsoc.load_state_dict(torch.load(self.model_path, weights_only=True))
        return self
    
    def score(self, X_test):
        # encode test data
        Y = self.cs.encode(X_test)

        # compute support estimation
        self.tsoc.eval()  
        Ytensor = torch.from_numpy(Y).float()
        O = torch.empty(X_test.shape)
        dataset = TensorDataset(Ytensor)  # Create a dataset from the tensor
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        for batch_idx, Y_batch in enumerate(loader):
            if self.gpu is not None:
                Y_batch = Y_batch[0].to(self.gpu) 
            O_batch = self.tsoc(Y_batch)
            O[self.batch_size*batch_idx: self.batch_size*(batch_idx + 1)] = O_batch

        if self.gpu is not None:
            O = O.cpu().detach().numpy()
        else:
            O = O.detach().numpy()

        # reconstruct data
        Xhat = np.empty(X_test.shape)
        for i in range(X_test.shape[0]):
            Xhat[i] = reconstructor(O[i], Y[i], self.A, self.D)

        if self.mode=='autoencoder':
            # estiamte the difference between the reconstructed data and data
            score = np.sqrt(np.sum((Xhat - X_test)**2, axis = -1))
        
        elif self.mode=='self-assessment':
            # encode reconstructed test data
            Yhat = self.cs.encode(Xhat)

            # estiamte the difference between the encoded reconstructed data and encoded data 
            score = np.sqrt(np.sum((Yhat - Y)**2, axis = -1))
        return score