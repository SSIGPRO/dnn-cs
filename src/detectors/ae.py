import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from models.ae import AE
from wombats.detectors._base import Detector
# from wombats.detectors.gaussian_distribution_based import MD

class MD(Detector):
    
    def __init__(self, epsilon=1e-7):
        super().__init__(None)
        self.epsilon = epsilon
        
    def fit(self, X_train):
        super().fit(X_train)
        N = X_train.shape[-2]
        self.Sok = 1/(N-1) * X_train.T @ X_train
        eigvals, eigvecs = np.linalg.eigh(self.Sok)
        self.lok = np.clip(eigvals[::-1], a_min=self.epsilon, a_max=None)
        self.Uok = eigvecs[:, ::-1]
        # print(self.lok)
        return self
        
    def score(self, X_test):
        # compute the projections on the major subspace
        projections = X_test @ self.Uok   
        return np.sqrt(np.sum(projections**2 / self.lok, axis=1))
    

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
        #  init a Mahalanobis distance-based detector
        if self.mode in ['MD', 'norm']:
            self.scaler = StandardScaler(with_std=False)
            if self.mode=='MD':
                self.MD =  MD()
            
    def fit(self, *args, **kwargs):
        # load trained DEC model
        self.ae.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.ae.eval()
        if self.mode in ['MD', 'norm']:
            X_train = kwargs['X_train']
            if X_train is None:
                raise ValueError("A training set is required to train the Mahalanobis distance-based detector")
            self.ae.eval()
            dataset = TensorDataset(torch.from_numpy(X_train).float())  # Create a dataset from the tensor
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            Y_train = torch.empty((X_train.shape[0], self.m))
            for batch_idx, X_batch in enumerate(loader):
                X_batch = X_batch[0]
                if self.gpu is not None:
                    X_batch = X_batch.to(self.gpu) 
                Y_batch = self.ae.encode(X_batch)
                Y_train[self.batch_size*batch_idx: self.batch_size*(batch_idx + 1)] = Y_batch
                
            if self.gpu is not None:
                Y_train = Y_train.detach().numpy()   
            self.scaler = self.scaler.fit(Y_train)  
            if self.mode=='MD':
                Y_train = self.scaler.transform(Y_train)  
                self.MD = self.MD.fit(Y_train)
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

        elif self.mode in ['MD', 'norm']:
            Y = torch.empty((X_test.shape[0], self.m))
            for batch_idx, X_batch in enumerate(loader):
                X_batch = X_batch[0]
                if self.gpu is not None:
                    X_batch = X_batch.to(self.gpu) 
                Y_batch = self.ae.encode(X_batch)
                Y[self.batch_size*batch_idx: self.batch_size*(batch_idx + 1)] = Y_batch

            if self.gpu is not None:
                Y = Y.cpu().detach().numpy()
            else:
                Y = Y.detach().numpy()
            Y = self.scaler.transform(Y)
            # print(self.scaler.mean_)
            if self.mode == 'MD':
                # print(self.MD.lok)
                score = self.MD.score(Y)
            else:
                score = np.sqrt(np.sum(Y**2, axis = -1))

        else:
            raise ValueError(f'mode "{self.mode}" not supported')
        
        return score