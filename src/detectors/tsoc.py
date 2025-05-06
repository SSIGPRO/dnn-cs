import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import linalg
from statsmodels.distributions.empirical_distribution import ECDF
from cs.wavelet_basis import wavelet_basis
from cs.utils import reconstructor
from cs import CompressedSensing, generate_sensing_matrix
from cs.decoders import SupportOracle
from models.tsoc import TSOC
from wombats.detectors._base import Detector

class MD_rtol(Detector):
    
    def __init__(self, rtol=1e-6):
        super().__init__(None)
        self.rtol = rtol
        
    def fit(self, X_train):
        super().fit(X_train)
        N = X_train.shape[-2]
        self.Sok = 1/(N-1) * X_train.T @ X_train
        self.Sokinv = linalg.pinvh(self.Sok, rtol=self.rtol)
        return self
        
    def score(self, X_test):
        score = np.sqrt(np.sum((X_test @ self.Sokinv) * X_test, axis=-1)) 
        return score 
    
class CDF():
    def __init__(self, norm=True):
        self.norm = norm
    
    def fit(self, score_train):
        self.ecdf = ECDF(score_train)
        self.mean = self.ecdf(np.mean(score_train))
        return self
    
    def predict(self, score_test):
        cdf = self.ecdf(score_test)
        if self.norm:
            # Normalize CDF values
            normalized = np.maximum(0, (cdf - self.mean) / (1 - self.mean))
            return normalized
        else:
            return cdf
    

class TSOCDetector(Detector):

    def __init__(self, cs, model_path, batch_size=100, mode='self-assessment', threshold=0.5, filter=True, gpu=None):
        assert isinstance(cs.decoder, SupportOracle), "This detector requires a CS instance initialized with 'SO' decoder"
        self.cs = cs
        self.n = cs.n
        self.m = cs.m
        self.threshold = threshold
        self.filter = filter

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

    def _estimate_support(self, Y):
        
        # compute support estimation
        self.tsoc.eval()  
        O = torch.empty((Y.shape[0], self.n))
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
        
        # ensure k < m
        if self.filter:
            th = np.sort(O, axis=-1)[:, self.n-(self.m-1)]
            th = th.reshape(-1, 1)
            index = O < th
            O[index] = 0
        return O
            
    def fit(self, *args, **kwargs):
        # load trained TSOC model
        self.tsoc.load_state_dict(torch.load(self.model_path, weights_only=True))
        if 'X_train' in kwargs:
            print('fitting')
            X_train = kwargs['X_train']
            Y = self.cs.encode(X_train)
            O = self._estimate_support(Y)
            # calculate supports
            Zhat = O > self.threshold
            # reconstruct
            Xhat = self.cs.decode(list(zip(Y, Zhat)))            

            # fit MD on reconstructed signal
            self.mean_xhat = np.mean(Xhat, axis=0)
            self.MD_xhat = MD_rtol().fit(Xhat-self.mean_xhat)

            # fit MD on the difference between reconstructed y and received y
            # encode reconstructed data
            Yhat = self.cs.encode(Xhat)
            difference = Yhat - Y
            self.mean_ydiff = np.mean(difference, axis=0)
            self.MD_ydiff = MD_rtol().fit(difference - self.mean_ydiff)
            self.CDF_ydiff = CDF().fit(self.MD_ydiff.score(difference - self.mean_ydiff))

            # MD on the y computed through the complement
            # compute the complement
            Q, _ = np.linalg.qr(self.cs.A.T, mode='complete')
            self.Ac = Q[:, self.m:].T
            Ybar = Xhat @ self.Ac.T
            self.mean_ycompl = np.mean(Ybar, axis=0)
            self.MD_ycompl = MD_rtol().fit(Ybar - self.mean_ycompl)
            self.CDF_ycompl = CDF().fit(self.MD_ycompl.score(Ybar - self.mean_ycompl))

        return self
    
    def score(self, X_test):
        # encode test data
        Y = self.cs.encode(X_test)
        O = self._estimate_support(Y)

        # calculate supports
        Zhat = O > self.threshold

        if self.mode == 'sparsity':
            score = np.mean(O, axis=-1)

        elif self.mode == 'sparsity-threshold':
            score = np.mean(Zhat, axis=-1)

        elif self.mode in [
                    'autoencoder',
                    'self-assessment',
                    'self-assessment-MD',
                    'complement',
                    'complement-MD',
                    'self-assessment-complement',
                    'self-assessment-complement-MD',
                    'self-assessment-complement-MD-CDF'
                  ]:
            # reconstruct data
            Xhat = self.cs.decode(list(zip(Y, Zhat)))

            if self.mode=='autoencoder':
                # estiamte the difference between the reconstructed data and data
                score = np.sqrt(np.sum((Xhat - X_test)**2, axis = -1))

            elif self.mode in ['self-assessment', 'self-assessment-MD', 'MDxhat']:
                # encode reconstructed test data
                Yhat = self.cs.encode(Xhat)
                # estiamte the difference between the encoded reconstructed data and encoded data 
                difference = Yhat - Y
                if self.mode == 'self-assessment':
                    score = np.sqrt(np.sum((difference)**2, axis = -1))
                elif self.mode == 'self-assessment-MD':
                    score = self.MD_ydiff.score(difference - self.mean_ydiff)
                else:
                    score = self.MD_xhat.score(Xhat - self.mean_xhat)

            elif self.mode in ['complement', 'complement-MD']:
                # find the orthogonal complement of A
                Q, _ = np.linalg.qr(self.cs.A.T, mode='complete')
                Ac = Q[:, self.m:].T

                # estimate the energy along the complement
                Ybar = Xhat @ Ac.T

                if self.mode == 'complement':
                    score = np.sqrt(np.sum((Ybar)**2, axis = -1))
                else:
                    score = self.MD_ycompl.score(Ybar - self.mean_ycompl)
            
            elif self.mode in ['self-assessment-complement', 'self-assessment-complement-MD', 'self-assessment-complement-MD-CDF']:
                Yhat = self.cs.encode(Xhat)
                difference = Yhat - Y
                Q, _ = np.linalg.qr(self.cs.A.T, mode='complete')
                Ac = Q[:, self.m:].T
                Ybar = Xhat @ Ac.T
                # combine two scores
                if self.mode == 'self-assessment-complement':
                    score = np.sqrt(np.sum((Ybar)**2, axis = -1)) \
                        + np.sqrt(np.sum((difference)**2, axis = -1)) 
                else:
                    score_ydiff = self.MD_ydiff.score(difference - self.mean_ydiff)
                    score_ycompl = self.MD_ycompl.score(Ybar - self.mean_ycompl)
                    if self.mode == 'self-assessment-complement-MD':
                        score = score_ydiff + score_ycompl
                    else:
                        score = self.CDF_ydiff.predict(score_ydiff) + self.CDF_ycompl.predict(score_ycompl)

        else:
            raise ValueError(f'mode "{self.mode}" not supported')
        
        return score