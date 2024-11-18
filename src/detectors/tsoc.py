import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cs.wavelet_basis import wavelet_basis
from cs.utils import reconstructor
from cs import CompressedSensing, generate_sensing_matrix
from models.tsoc import TSOC

class TSOCDetector(Detector):

    def __init__(self, mode='self-assessment', **kwargs):
        # init TSOC
        self.tsoc = TSOC(kwargs['n'], kwargs['m'])
        if 'device' in kwargs.keys():
            self.gpu = kwargs['device']
            self.tsoc.to(self.gpu) # move the network to GPU
        else:
            self.gpu = None
        self.file_model = f"TSOC-N={kwargs['N_train']}_n={kwargs['n']}_fs={kwargs['fs']}_hr={kwargs['heart_rate'][0]}-{kwargs['heart_rate'][1]}"\
                        f"_isnr={kwargs['isnr']}_seed={kwargs['seed']}-epochs={kwargs['epochs']}-bs={kwargs['batch_size']}-lr={kwargs['lr']}.pth"
        # init Compressed Sensing
        self.A = generate_sensing_matrix((kwargs['m'], kwargs['n']), seed=kwargs['seed'])
        self.D = wavelet_basis(n, basis='sym6', level=2)
        self.cs = CompressedSensing(self.A, self.D)
        # batch_size for evaluation
        self.batch_size = 100
        self.mode = mode
            
    def fit(self):
        # load trained TSOC model
        self.tsoc.load_state_dict(torch.load(self.file_model, weights_only=True))
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
            score = np.sqrt(np.sum(Xhat - X_test)**2, axis = -1)
        
        elif self.mode=='self-assessment':
            # encode reconstructed test data
            Yhat = self.cs.encode(Xhat)

            # estiamte the difference between the encoded reconstructed data and encoded data 
            score = np.sqrt(np.sum((Yhat - Y)**2, axis = -1))
        return score