import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNet

class AEUnet(nn.Module):
    def __init__(self,
                 n=128,
                 m=16,
                 in_channels=1, 
                 num_classes=1,
                 expanded_channels=64, 
                 steps_num=4, 
                 kernel_size=3,
                 use_batch_norm=True, 
                 simple_pool=False,
                 encoder_mode='default',
                 decode_mode='lstsq'): 
        super(AEUnet, self).__init__()
        self.n = n
        self.m = m
        self.in_channels=in_channels 
        self.num_classes=num_classes
        self.expanded_channels=expanded_channels 
        self.steps_num=steps_num 
        self.kernel_size=kernel_size
        self.use_batch_norm=use_batch_norm 
        self.simple_pool=simple_pool
        self.encoder_mode=encoder_mode
        if encoder_mode == 'orth':
            decode_mode = 'transpose'
        self.decode_mode = decode_mode
        self._learned_Apinv = None

        # trainable encoding matrix
        self.fc0 = nn.Linear(self.n, self.m, bias=False)
        # decoder
        # trainable decoding matrix
        self.fc1 = nn.Linear(self.m, self.n, bias=False)
        self.unet = UNet(in_channels=in_channels, 
                 num_classes=self.num_classes,
                 expanded_channels=self.expanded_channels, 
                 steps_num=self.steps_num, 
                 kernel_size=self.kernel_size,
                 residual=False,
                 use_batch_norm=self.use_batch_norm, 
                 simple_pool=self.simple_pool)

    # def _pinv(self):
    #     if self.training:
    #         return torch.linalg.pinv(self.fc0.weight)
    #     else:
    #         if self._learned_Apinv is None:
    #             with torch.no_grad():
    #                 self._learned_Apinv = torch.linalg.pinv(self.fc0.weight).detach()
    #         return self._learned_Apinv

    def _pinv(self):
        return torch.linalg.pinv(self.fc0.weight, rcond=1e-5)

    def encode(self, x):
        if self.encoder_mode == 'orth':
            A = self.fc0.weight
            Aort, _ = torch.linalg.qr(A.T, mode='reduced')
            # Aort = Aort[:, :self.m].T
            return x @ Aort
        else:
            return self.fc0(x)
    
        # def encode(self, x):
        #     return self.fc0(x)
    
    def decode_coarse(self, y):
        if self.decode_mode == 'pinv':
            A_pinv = self._pinv()
            xbar = y @ A_pinv.T
        elif self.decode_mode == 'svd':
            Y = y.T                                   # (m, B)
            U, S, Vh = torch.linalg.svd(self.fc0.weight, full_matrices=False)
            X = Vh.mT @ ((U.mT @ Y) / S.unsqueeze(-1))  # (n, B) == A^+ @ Y
            xbar = X.T   
        elif self.decode_mode == 'lstsq':
            xbar = torch.linalg.lstsq(self.fc0.weight, y.T, driver="gels").solution.T
        elif self.decode_mode == 'transpose':
            if self.encoder_mode == 'orth':
                A = self.fc0.weight
                Aort, _ = torch.linalg.qr(A.T, mode='reduced')
                # Aort = Aort[:, :self.m].T
                xbar = y @ Aort.T
            else:
                xbar = y @ self.fc0.weight
        elif self.decode_mode == 'learn':
            xbar = self.fc1(y)
        return xbar
    
    def _decode(self, y):
        xbar = self.decode_coarse(y)
        xbar = xbar.view(-1, self.in_channels, self.n // self.in_channels) 
        residual = self.unet(xbar)
        return xbar.squeeze(1) , residual.squeeze(1)  

    def decode(self, y):
        xbar, residual = self._decode(y)
        return xbar + residual 

    def forward(self, x):
        y = self.encode(x)
        output = self.decode(y)
        return output
    

class AEUnetLegacy(nn.Module):
    def __init__(self,
                 n=128,
                 m=16,
                 in_channels=1, 
                 num_classes=1,
                 expanded_channels=64, 
                 steps_num=4, 
                 kernel_size=3,
                 residual='False',
                 use_batch_norm='True', 
                 simple_pool='False',
                 decode_mode='lstsq'): 
        super(AEUnetLegacy, self).__init__()
        self.n = n
        self.m = m
        self.in_channels=in_channels 
        self.num_classes=num_classes
        self.expanded_channels=expanded_channels 
        self.steps_num=steps_num 
        self.kernel_size=kernel_size
        self.residual=residual
        self.use_batch_norm=use_batch_norm 
        self.simple_pool=simple_pool
        self.decode_mode = decode_mode

        # trainable encoding matrix
        self.fc0 = nn.Linear(self.n, self.m, bias=False)
        # decoder
        # trainable decoding matrix
        self.unet = UNet(in_channels=in_channels, 
                 num_classes=self.num_classes,
                 expanded_channels=self.expanded_channels, 
                 steps_num=self.steps_num, 
                 kernel_size=self.kernel_size,
                 residual=self.residual,
                 use_batch_norm=self.use_batch_norm, 
                 simple_pool=self.simple_pool)

    def _pinv(self):
        if self.training:
            return torch.linalg.pinv(self.fc0.weight)
        else:
            if self._learned_Apinv is None:
                with torch.no_grad():
                    self._learned_Apinv = torch.linalg.pinv(self.fc0.weight).detach()
            return self._learned_Apinv

    # def _pinv(self):
    #     return torch.linalg.pinv(self.fc0.weight)

    def encode(self, x):
        return self.fc0(x)
    
    def decode_coarse(self, y, mode='lstsq'):
        if mode == 'pinv':
            A_pinv = self._pinv()
            xbar = y @ A_pinv.T
        elif mode == 'svd':
            Y = y.T                                   # (m, B)
            U, S, Vh = torch.linalg.svd(self.fc0.weight, full_matrices=False)
            X = Vh.mT @ ((U.mT @ Y) / S.unsqueeze(-1))  # (n, B) == A^+ @ Y
            xbar = X.T   
        elif mode == 'lstsq':
            xbar = torch.linalg.lstsq(self.fc0.weight, y.T, driver="gels").solution.T
        elif mode == 'transpose':
            xbar = y @ self.fc0.weight
        return xbar
    
    def _decode(self, y):
        xbar = self.decode_coarse(y, self.decode_mode)
        xbar = xbar.view(-1, self.in_channels, self.n // self.in_channels) 
        residual = self.unet(xbar)
        return xbar.squeeze(1) , residual.squeeze(1)  

    def decode(self, y):
        xbar, residual = self._decode(y)
        return xbar + residual 

    def forward(self, x):
        y = self.encode(x)
        output = self.decode(y)
        return output