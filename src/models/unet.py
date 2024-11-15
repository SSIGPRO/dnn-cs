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
            self.pool = nn.Conv1d(in_channels, out_channels, kernel_size=2, stride=2)


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

            l = UpSample(in_ch,out_ch, kernel_size, batch_norm=use_batch_norm)
            
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
    def __init__(self, in_channels, num_classes, channels, steps_num=4,
                 kernel_size=3, simple_pool=True, use_batch_norm=False,
                 residual=False):
        super().__init__()

        self.residual = residual
        if residual:
            assert in_channels == num_classes, 'if "residual==True", must have "in_channels==num_classes"'

        self.down = DownSeq(
            in_channels=in_channels,
             channels=channels,
              steps_num=steps_num,
               kernel_size=kernel_size,
                simple_pool=simple_pool,
                 use_batch_norm=use_batch_norm)
        
        ch = channels * (2**(steps_num-1))
        self.bottle_neck = DoubleConv(ch, ch*2, kernel_size, batch_norm=use_batch_norm)

        self.up = UpSeq(
            channels=channels,
             kernel_size=kernel_size,
              steps_num=steps_num,
               use_batch_norm=use_batch_norm,)

        self.out = nn.Conv1d(in_channels=channels, out_channels=num_classes, kernel_size=1)

    def forward(self, x_input):

        x, down_list = self.down(x_input)

        x = self.bottle_neck(x)
        
        x = self.up(x, down_list[::-1])

        x = self.out(x)

        if self.residual:
            x = x_input - x

        return x

# class UNet(nn.Module):
#     def __init__(self, in_channels, num_classes, channels, steps_num=4,
#                  kernel_size=None, simple_pool=True, use_batch_norm=False,
#                  residual=True):
#         super().__init__()

#         self.residual = residual
#         if residual:
#             assert in_channels == num_classes, 'if "residual==True", must have "in_channels==num_classes"'

#         l_list = []
#         for i in range(steps_num):

#             if i==0:
#                 in_down_ch = in_channels
#                 out_down_ch = channels

#                 in_up_ch = in_channels * (2**steps_num)
#                 out_up_ch = in_up_ch / 2

#             elif i>0 and i<(steps_num-1):
#                 in_down_ch = channels * (2**(i-1))
#                 out_down_ch = in_down_ch * 2

#                 in_up_ch = in_channels * (2**(steps_num-i))
#                 out_up_ch = in_up_ch / 2

#             elif i==(steps_num-1):
#                 in_down_ch = channels * (2**(i-1))
#                 out_down_ch = in_down_ch * 2

#                 in_up_ch = in_channels * (2**(steps_num-i))
#                 out_up_ch = in_up_ch / 2
    
#             layers_down_list += [DownSample(in_down_ch, out_down_ch, kernel_size, simple_pool=simple_pool, batch_norm=use_batch_norm)]
#             layers_up_list += [UpSample(in_up_ch, out_up_ch, kernel_size, simple_pool=simple_pool, batch_norm=use_batch_norm)]

#         self.down_seq = nn.Sequential(*layers_down_list)
#         self.up_seq = nn.Sequential(*layers_up_list)

#         # self.down_convolution_1 = DownSample(in_channels, channels, kernel_size, simple_pool=simple_pool, batch_norm=use_batch_norm)
#         # self.down_convolution_2 = DownSample(channels, channels*2, kernel_size, simple_pool=simple_pool, batch_norm=use_batch_norm)
#         # self.down_convolution_3 = DownSample(channels*2, channels*4, kernel_size, simple_pool=simple_pool, batch_norm=use_batch_norm)
#         # self.down_convolution_4 = DownSample(channels*4, channels*8, kernel_size, simple_pool=simple_pool, batch_norm=use_batch_norm)

#         self.bottle_neck = DoubleConv(out_down_ch, out_down_ch*2, kernel_size, batch_norm=use_batch_norm)

#         # self.up_convolution_1 = UpSample(channels*16, channels*8, kernel_size, batch_norm=use_batch_norm)
#         # self.up_convolution_2 = UpSample(channels*8, channels*4, kernel_size, batch_norm=use_batch_norm)
#         # self.up_convolution_3 = UpSample(channels*4, channels*2, kernel_size, batch_norm=use_batch_norm)
#         # self.up_convolution_4 = UpSample(channels*2, channels, kernel_size, batch_norm=use_batch_norm)

#         self.out = nn.Conv1d(in_channels=channels, out_channels=num_classes, kernel_size=1)

#     def forward(self, x):

#         # down_1, p1 = self.down_convolution_1(x)
        
#         # down_2, p2 = self.down_convolution_2(p1)
        
#         # down_3, p3 = self.down_convolution_3(p2)
        
#         # down_4, p4 = self.down_convolution_4(p3)
        
#         # b = self.bottle_neck(p4)
        
#         # up_1 = self.up_convolution_1(b, down_4)
        
#         # up_2 = self.up_convolution_2(up_1, down_3)
        
#         # up_3 = self.up_convolution_3(up_2, down_2)
        
#         # up_4 = self.up_convolution_4(up_3, down_1)
        
#         # out = self.out(up_4)

#         self.up_seq()

#         if self.residual:
#             out = x - out

#         return out