import torch
import torch.nn as nn
 
import os
import numpy as np

"""
IMPROVEMENTS:
- Batch Norm
- Resize Convolution instead of Strided Deconvolution for Upsampling
- Strided Convolution instead of Strided Max Pooling for Downsampling
- Partial Convolution (https://github.com/NVIDIA/partialconv) which correct the errors introduced by zero-padding
"""

class CS_enc(nn.Module):
    def __init__(self, n, m):
        super().__init__()

        self.enc = nn.Linear(n, m, bias=False),

    def forward(self, x):
        return self.enc(x)
    
class CS_dec_first_guess(nn.Module):
    def __init__(self, m, n, bias=False):
        super().__init__()

        self.expand = nn.Linear(m, n, bias=bias),

    def forward(self, x):
        return self.expand(x)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, ), batch_norm=False, 
                 padding=None):
        super().__init__()


        if padding is None:
            padding = kernel_size//2

        if not batch_norm:
            self.conv_op = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )

        else:
            self.conv_op = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=None, pool_kernel_size=(2, ), 
                 simple_pool=True, batch_norm=False):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, batch_norm=batch_norm)

        if simple_pool:
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        else:
            self.pool = nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2)


    def forward(self, x):
        
        down = self.conv(x)

        p = self.pool(down)
        
        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, ), batch_norm=False):
        super().__init__()

        self.up = nn.ConvTranspose1d(in_channels, in_channels//2, 
                                     kernel_size=2, stride=2,)
        
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, batch_norm=batch_norm)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat([x1, x2], -2)
        x = self.conv(x)
        
        return x


class DownSeq(nn.Module):
    def __init__(self, in_channels, channels, steps_num=4,
                 kernel_size=None, simple_pool=True, use_batch_norm=False):
        super().__init__()

        layers_list = []
        for i in range(steps_num):

            if i==0:
                in_ch = in_channels
                out_ch = channels

            else:
                in_ch = channels * (2**(i-1))
                out_ch = in_ch * 2
    
            l = DownSample(in_ch, out_ch, kernel_size, simple_pool=simple_pool, batch_norm=use_batch_norm)

            self.__setattr__('down_' + str(i), l)

            layers_list += [l]

        self.list_l = layers_list

        # self._set_layers_as_attr(self.list_l)

    def _set_layers_as_attr(self, list_layers, name_l_root=None):
        
        if name_l_root is None:
            name_l_root = self.__name__
        
        name_all_l = [i for i in self.__dict__.keys() if i[:1] != '_'] # all layers name so far

        for i, l in enumerate(list_layers):
            name_l = name_l_root + str(i)

            while name_l in name_all_l:
                name_l = name_l + '_'
            
            self.__setattr__(name_l, l)

    def forward(self, x):

        down_list = []
        for l in self.list_l:
            down, x = l(x)
            down_list += [down]

        return x, down_list


class UpSeq(nn.Module):
    def __init__(self, channels, kernel_size=None, steps_num=4, use_batch_norm=False):
        super().__init__()

        layers_list = []
        for i in range(steps_num, 0, -1):

            in_ch = channels * (2**i)
            out_ch = in_ch // 2

            l = UpSample(in_ch, out_ch, kernel_size, batch_norm=use_batch_norm)
            
            self.__setattr__('up_' + str(i), l)
            
            layers_list += [l]

        self.list_l = layers_list

        # self._set_layers_as_attr(self.list_l)

    def _set_layers_as_attr(self, list_layers, name_l_root=None):
        
        if name_l_root is None:
            name_l_root = self.__name__
        
        name_all_l = [i for i in self.__dict__.keys() if i[:1] != '_'] # all layers name so far

        for i, l in enumerate(list_layers):
            name_l = name_l_root + str(i)

            while name_l in name_all_l:
                name_l = name_l + '_'
            
            self.__setattr__(name_l, l)

    def forward(self, x, down_list):

        for l, down in zip(self.list_l, down_list):
            x = l(x, down)
            
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, expanded_channels, 
                 steps_num=4, kernel_size=3, simple_pool=True,
                 use_batch_norm=False, residual=False, x_as_input=False, 
                 n=None, m=None, A_init=None, A_freeze=True, verbose=False):
        super().__init__()

        self.residual = residual
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.x_as_input = x_as_input

        if x_as_input:
            assert n is not None and m is not None, 'must specify "m" and "n"'
            self.encoder = nn.Linear(n, m, bias=False)
            self.dec_first_guess = nn.Linear(m, n, bias=False)
            if A_init is not None:
                assert A_init.shape == (m, n), 'A shape must be (m, n)'
                # check A type (torch)

                if type(A_init) is not torch.Tensor:
                    A_init = torch.tensor(A_init, dtype=torch.float32)

                self.encoder.weight.data = A_init
                self.dec_first_guess.weight.data = A_init.transpose(-1, -2)
                self.encoder.weight.requires_grad = not(A_freeze)
                self.dec_first_guess.weight.requires_grad = not(A_freeze)

                if verbose:
                    print(f'A init: {A_init != None}. A grad: {self.encoder.weight.requires_grad}')

                pass


        self.down = DownSeq(
            in_channels=in_channels,
             channels=expanded_channels,
              steps_num=steps_num,
               kernel_size=kernel_size,
                simple_pool=simple_pool,
                 use_batch_norm=use_batch_norm)
        
        ch = expanded_channels * (2**(steps_num-1))
        self.bottle_neck = DoubleConv(ch, ch*2, kernel_size, batch_norm=use_batch_norm)

        self.up = UpSeq(
            channels=expanded_channels,
             kernel_size=kernel_size,
              steps_num=steps_num,
               use_batch_norm=use_batch_norm,)

        self.out = nn.Conv1d(in_channels=expanded_channels, out_channels=in_channels, kernel_size=1)
        if self.out_channels != self.in_channels:
            self.out2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x_input):

        if self.x_as_input:
            x = self.encoder(x_input)
            x_first_guess = self.dec_first_guess(x)
        else:
            x_first_guess = x_input

        x, down_list = self.down(x_first_guess)

        x = self.bottle_neck(x)
        
        x = self.up(x, down_list[::-1])

        x = self.out(x)

        if self.residual:
            x = x_first_guess  - x

        if self.out_channels != self.in_channels:
            x = self.out2(x)

        return x