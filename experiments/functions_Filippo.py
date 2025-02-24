import numpy as np
import itertools
import os
import sys
import sys; sys.path.append('..')
from tqdm import tqdm
import pickle as pkl
import warnings
import copy
import time
import wfdb
import pandas as pd
from scipy import signal

# import io
from contextlib import redirect_stdout

import torch
torch.manual_seed(0);
import torch.utils
import torch.utils.data
import torch.nn as nn

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))


from dataset import dataset_dir
from models.unet import UNet


# def get_dict_default_args_cmd(augmented=True):
#     #### DATASET PARAMETERS ####
#     size_ecg_default = 10_000
#     n_default = 128
#     seed_ecg_default = 0
#     isnr_default = 35
#     hr0_default = 60
#     hr1_default = 100
#     fs_default = 256

#     #### A PARAMTERS ####
#     m_default = 48
#     modeA_default = 'rakeness'
#     str_orth_default = "True"
#     seed_A_default = 0
#     index_A_default = 0
#     N_try_A_default = 1_000
#     str_corr_default = '96af96a7ddfcb2f6059092c250e18f2a'
#     loc_default = 0.25
#     ## Dataset used for generating A
#     _ecg_N_forA_default = 10_000
#     _n_forA_default = 128
#     _fs_forA_default = 256
#     _hr0_forA_default = 60
#     _hr1_forA_default = 100
#     _isnr_forA_default = 35
#     _seed_forA_default = 0

#     #### UNET PARAMETERS ####
#     in_channels_default = 1
#     expanded_channels_default = 64
#     step_number_default = 4
#     kernel_size_default = 3
#     str_residual_default = "True"
#     str_use_batch_norm_default = "True"
#     str_simple_pool_default = "False"
#     seed_torch_default = 0
#     str_retrain_default = "False"
#     str_resume_train_default = "False"
#     out_channels_default = 1
#     str_x_as_input_default = "True"
#     str_A_init_default = "True"
#     str_A_freeze_default = "True"

#     #### HARDWARE PARAMETERS ####
#     str_gpus_default = "5"
#     threads_default = 1
#     workers_default = 1

#     #### TRAINING PARAMS ####
#     batch_size_default = 64
#     white_noise_var_default = 0.0
#     white_noise_isnr_default = 100

#     early_stopping_patience = 30
#     min_improvement = 1e-8
#     lr_plateaut_patience = 15
#     min_lr = 1e-6
#     lr_default = 0.001
#     criterion = nn.MSELoss()
#     num_epochs = 100_000

#     default_args = locals().copy()

#     default_args = {_k.replace('_default', ''): _v for _k, _v in  default_args.items()}

#     if augmented: # remove some chr in case compact keywords are also needed
#         _tmp_dict_str = {_k.replace('str_', ''): _v for _k, _v in  default_args.items()}
#         _tmp_dict__ = {_k.replace('_', ''): _v for _k, _v in  default_args.items()}
#         _tmp_dict_str_ = {_k.replace('_', ''): _v for _k, _v in  _tmp_dict_str.items()}
#         default_args.update(_tmp_dict_str)
#         default_args.update(_tmp_dict__)
#         default_args.update(_tmp_dict_str_)

#     return default_args

def get_bundle_default(include_default_in_key=True,
                       include_underscore_in_key=True,
                       include_str_in_key=True):
    #### DATASET PARAMETERS ####
    size_ecg_default = 10_000
    n_default = 128
    seed_ecg_default = 0
    isnr_default = 35
    hr0_default = 60
    hr1_default = 100
    fs_default = 256

    #### A PARAMTERS ####
    m_default = 48
    modeA_default = 'rakeness'
    str_orth_default = "True"
    seed_A_default = 0
    index_A_default = 0
    N_try_A_default = 1_000
    str_corr_default = '96af96a7ddfcb2f6059092c250e18f2a'
    loc_default = 0.25
    ## Dataset used for generating A
    _ecg_N_forA_default = 10_000
    _n_forA_default = 128
    _fs_forA_default = 256
    _hr0_forA_default = 60
    _hr1_forA_default = 100
    _isnr_forA_default = 35
    _seed_forA_default = 0

    #### UNET PARAMETERS ####
    in_channels_default = 1
    expanded_channels_default = 64
    step_number_default = 4
    kernel_size_default = 3
    str_residual_default = "True"
    str_use_batch_norm_default = "True"
    str_simple_pool_default = "False"
    seed_torch_default = 0
    str_retrain_default = "False"
    str_resume_train_default = "False"
    out_channels_default = 1
    str_x_as_input_default = "True"
    str_A_init_default = "True"
    str_A_freeze_default = "True"

    #### HARDWARE PARAMETERS ####
    str_gpus_default = "5"
    threads_default = 1
    workers_default = 1

    #### TRAINING PARAMS ####
    batch_size_default = 64
    white_noise_var_default = 0.0
    white_noise_isnr_default = 100

    early_stopping_patience = 30
    min_improvement = 1e-8
    lr_plateaut_patience = 15
    min_lr = 1e-6
    lr_default = 0.001
    criterion = nn.MSELoss()
    num_epochs_default = 100_000

    default_args = locals().copy()

    default_args.pop('include_default_in_key')
    default_args.pop('include_underscore_in_key')
    default_args.pop('include_str_in_key')

    if not(include_default_in_key):
        default_args = {_k.replace('_default', ''): _v for _k, _v in  default_args.items()}

    if not(include_str_in_key):
        default_args = {_k.replace('str_', ''): _v for _k, _v in  default_args.items()}

    if not(include_underscore_in_key):
        default_args = {_k.replace('_', ''): _v for _k, _v in  default_args.items()}


    return default_args

def dict_bundle_to_path_arg(include_default_in_key=True,
                            include_underscore_in_key=True,
                            include_str_in_key=True):
    
    #### DATASET PARAMETERS ####
    size_ecg_default = 10_000
    n_default = 128
    seed_ecg_default = 0
    isnr_default = 35
    hr0_default = 60
    hr1_default = 100
    fs_default = 256

    #### A PARAMTERS ####
    m_default = 48
    modeA_default = 'rakeness'
    str_orth_default = "True"
    seed_A_default = 0
    index_A_default = 0
    N_try_A_default = 1_000
    str_corr_default = '96af96a7ddfcb2f6059092c250e18f2a'
    loc_default = 0.25
    ## Dataset used for generating A
    _ecg_N_forA_default = 10_000
    _n_forA_default = 128
    _fs_forA_default = 256
    _hr0_forA_default = 60
    _hr1_forA_default = 100
    _isnr_forA_default = 35
    _seed_forA_default = 0

    #### UNET PARAMETERS ####
    in_channels_default = 1
    expanded_channels_default = 64
    step_number_default = 4
    kernel_size_default = 3
    str_residual_default = "True"
    str_use_batch_norm_default = "True"
    str_simple_pool_default = "False"
    seed_torch_default = 0
    str_retrain_default = "False"
    str_resume_train_default = "False"
    out_channels_default = 1
    str_x_as_input_default = "True"
    str_A_init_default = "True"
    str_A_freeze_default = "True"

    #### HARDWARE PARAMETERS ####
    str_gpus_default = "5"
    threads_default = 1
    workers_default = 1

    #### TRAINING PARAMS ####
    batch_size_default = 64
    white_noise_var_default = 0.0
    white_noise_isnr_default = 100

    early_stopping_patience = 30
    min_improvement = 1e-8
    lr_plateaut_patience = 15
    min_lr = 1e-6
    lr_default = 0.001
    criterion = nn.MSELoss()
    num_epochs_default = 100_000

    default_args = locals().copy()

    if not(include_default_in_key):
        default_args = {_k.replace('_default', ''): _v for _k, _v in  default_args.items()}

    if not(include_str_in_key):
        default_args = {_k.replace('str_', ''): _v for _k, _v in  default_args.items()}

    if not(include_underscore_in_key):
        default_args = {_k.replace('_', ''): _v for _k, _v in  default_args.items()}

    return default_args

def parser_add_arguments_from_bundle(b, parser):
    for k, v in b.items():
        str_arg = '-' + str(k)
        if len(k)>1: str_arg += '-' 
        parser.add_argument(str_arg, action='store', 
                            type=type(v), dest=k, default=v)
        
    return parser


def color(text, text_before='', text_after='', code="ff0000"):
    return f"{text_before}<span style='color: #{code}'>{text}</span>{text_after}"

def set_all_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    worker_seed = torch.initial_seed() % 2**32

    generator = torch.Generator()
    generator.manual_seed(seed)


    return generator

def get_cell_vars_as_dict(ipy, out, glob, offset=0):

    with redirect_stdout(out):
        ipy.run_line_magic("history", str(ipy.execution_count - offset))

    #process each line...
    x = out.getvalue().replace(" ", "").split("\n")
    x = [a.split("=")[0] for a in x if "=" in a] #all of the variables in the cell
    # g = globals()
    result = {k:glob[k] for k in x if k in glob}
    return result

def x_to_dataset(x, A, xbar_index=[1], 
                    return_x=True, 
                    return_y=True, 
                    torch_dtype=torch.float32, 
                    verbose=False):

    if x.shape[-1] != 1:
        x = np.swapaxes(x, -1, -2)
    y = A @ x

    
    d_outputs = {}
    if return_x:
        d_outputs['x'] = x
    if return_y:
        d_outputs['y'] = y

    if 1 in xbar_index:
        d_outputs['xbarT'] = A.T @ y
    if 2 in xbar_index:
        d_outputs['xbarPinv'] = np.linalg.pinv(A) @ y
    # if 3 in xbar_index:
    #     d_outputs['xbar0Padding'] = np.pad(y, )

    if x.shape[-1] == 1:
        d_outputs = {k: np.swapaxes(v, -1, -2) for k, v in d_outputs.items()}
    
    ## change to torch tensor specified type
    d_outputs = {k: torch.tensor(v, dtype=torch_dtype) for k, v in d_outputs.items()}

    if verbose:
        print('out contains:', d_outputs.keys())
    return d_outputs


def get_first_l(model, l_name):
    return dict(model.named_children())[l_name]
def get_generic_l(model, l_name):
    for _l_name in l_name:
        model = get_first_l(model, _l_name)
    return model
def get_size_model(model):
    return sum(p.numel() for p in model.parameters())

def unravel_bundle_param(bundle_param, verbose=False):

    v_list, k_list = [], []
    b_before = bundle_param.copy()
    for k, v in bundle_param.items():
        if type(v) is list and len(v)>1:
            k_list += [k]
            v_list += [v]
            b_before.pop(k)

    b_unravel_list, b_add_list = [], []
    for _v_tuple in itertools.product(*v_list):
        b_add = dict(zip(k_list, _v_tuple))
        if verbose:
            [print(f"{_k}: {_v}") for _k, _v in b_add.items()]
            print()
        b_unravel_list += [b_before | b_add]
        b_add_list += [b_add]
    
    return b_unravel_list, b_add_list

def to_list(*args):
     return [arg if type(arg) is list else [arg] for arg in args]

def from_bundle_to_cmd_bash(path_py, bundle_param, nohup=True, nohup_file=None, verbose=False):
    
    str_cmd = 'python ' + path_py

    if nohup:
        str_cmd = 'nohup ' + str_cmd

    for k, v in bundle_param.items():

        str_cmd += ' '

        # str_left = k[1:-5]
        if len(k)==1:
            str_left = '-' + k
        else:
            str_left = '--' + k

        str_cmd += str_left + ' ' + str(v)
        
    if nohup and (nohup_file is not None):
        str_cmd += ' &>' + nohup_file + ' &'

    if verbose: print(str_cmd)

    return str_cmd

def from_bundle_to_ordered_model_kwds(b, version='old'):

    kwds = []

    if b['isnr']>=100:
        b['isnr'] = 'None'
    
    if version=='old':
        
        if b.get('N', None) is not None:
            kwds += [f"N={b['N']}"]
        elif b.get('size_ecg', None) is not None:
            kwds += [f"N={b['size_ecg']}"]
        else:
            assert False, 'no "N"'
        
        kwds += [f"n={b['n']}"]
        kwds += [f"m={b['m']}"]
        kwds += [f"isnr={b['isnr']}"]
        kwds += [f"seedecg={b['seed_ecg']}"]
        
        if b.get('mode_A', None) is not None:
            kwds += [f"modeA={b['mode_A']}"]
        elif b.get('modeA', None) is not None:
            kwds += [f"modeA={b['modeA']}"]
        else:
            assert False, 'no "modeA"'

        kwds += [f"seedtorch={b['seed_torch']}"]
        
        if b.get('orth', None) is not None:
            kwds += [f"orth={b['orth']}"]
        elif b.get('str_orth', None) is not None:
            kwds += [f"orth={b['str_orth']}"]
        else:
            assert False, "no 'orth'"

        kwds += [f"corr={b['str_corr']}"]
        kwds += [f"inchannels={b['in_channels']}"]
        kwds += [f"expandedchannels={b['expanded_channels']}"]
        kwds += [f"stepnumber={b['step_number']}"]
        kwds += [f"kernelsize={b['kernel_size']}"]

        if b.get('residual', None) is not None:
            kwds += [f"residual={b['residual']}"]
        elif b.get('str_residual', None) is not None:
             kwds += [f"residual={b['str_residual']}"]
        else:
            assert False, "no 'residual'"

        if b.get('use_batch_norm', None) is not None:
            kwds += [f"usebatchnorm={b['use_batch_norm']}"]
        elif b.get('str_use_batch_norm', None) is not None:
             kwds += [f"usebatchnorm={b['str_use_batch_norm']}"]
        else:
            assert False, "no 'use_batch_norm'"

        if b.get('simple_pool', None) is not None:
            kwds += [f"simplepool={b['simple_pool']}"]
        elif b.get('str_simple_pool', None) is not None:
             kwds += [f"simplepool={b['str_simple_pool']}"]
        else:
            assert False, "no 'simple_pool'"

        if b.get('white_noise_isnr', None) is not None:
            if b['white_noise_isnr'] < 100:
                kwds += [f"white_noise_isnr={b['white_noise_isnr']}"]

        if b.get('white_noise_var', None) is not None:
            if b['white_noise_var'] > 0:
                kwds += [f"white_noise_var={b['white_noise_var']}"]

        
        if b.get('str_x_as_input', None) is not None:
            if b['str_x_as_input'] == True:
                kwds += [f"str_x_as_input={b['str_x_as_input']}"]

            if b.get('str_A_init', None) is not None:
                if b['str_A_init'] == True:
                    kwds += [f"str_A_init={b['str_A_init']}"]
            if b.get('str_A_freeze', None) is not None:
                if b['str_A_freeze'] == False:
                    kwds += [f"str_A_freeze={b['str_A_freeze']}"]

    return kwds

def from_bundle_to_model_path(b, verbose=False):

    # b = copy.deepcopy(b)

    # if b['isnr']>=100:
    #     b['isnr'] = 'None'

    # str_model_setting = f"N={b['N']}_"\
    #     f"n={b['n']}_"\
    #         f"m={b['m']}_"\
    #             f"isnr={b['isnr']}_"\
    #                 f"seedecg={b['seed_ecg']}_"\
    #                     f"modeA={b['mode_A']}_"\
    #                         f"seedtorch={b['seed_torch']}_"\
    #                             f"orth={b['orth']}_"\
    #                                 f"corr={b['str_corr']}_"\
    #                                     f"inchannels={b['in_channels']}_"\
    #                                         f"expandedchannels={b['expanded_channels']}_"\
    #                                             f"stepnumber={b['step_number']}_"\
    #                                                 f"kernelsize={b['kernel_size']}_"\
    #                                                     f"residual={b['residual']}_"\
    #                                                         f"usebatchnorm={b['use_batch_norm']}_"\
    #                                                             f"simplepool={b['simple_pool']}"
    
    # if b.get('white_noise_isnr', None) is not None:
    #     if b['white_noise_isnr'] < 100:
    #         str_model_setting = str_model_setting + f'_white_noise_isnr={b['white_noise_isnr']}'

    # if b.get('white_noise_var', None) is not None:
    #     if b['white_noise_var'] > 0:
    #         str_model_setting = str_model_setting + f'_white_noise_var={b['white_noise_var']}'

    
    # if b.get('str_x_as_input', None) is not None:
    #     if b['str_x_as_input'] == True:
    #         str_model_setting = str_model_setting + f'_str_x_as_input={b['str_x_as_input']}'

    #     if b.get('str_A_init', None) is not None:
    #         if b['str_A_init'] == True:
    #             str_model_setting = str_model_setting + f'_str_A_init={b['str_A_init']}'
    #     if b.get('str_A_freeze', None) is not None:
    #         if b['str_A_freeze'] == False:
    #             str_model_setting = str_model_setting + f'_str_A_freeze={b['str_A_freeze']}'

    kwds = from_bundle_to_ordered_model_kwds(b)
    str_model_setting = '_'.join(kwds)
    if len(str_model_setting) >= 255:
        if verbose: print('\nPath for saving the model was too long, the following kwds have been removed:')
        b_default = get_bundle_default(
            include_default_in_key=False)
        
        kwds_default = from_bundle_to_ordered_model_kwds(b_default)

        for _kwd_default in kwds_default:
            if _kwd_default in kwds:
                kwds.remove(_kwd_default)
                if verbose: print(_kwd_default)

        str_model_setting = '_'.join(kwds)

    path_model = os.path.join(dataset_dir,
                            '..',
                            'trained_models',
                            'dummy_unet',
                            str_model_setting+'.pt')

    return path_model
    
def from_bundle_to_A_path(b):
    b = copy.deepcopy(b)
    str_A_dataset = f"ecg_N={b['_ecg_N_forA']}_"\
        f"n={b['_n_forA']}_"\
            f"fs={b['_fs_forA']}_"\
                f"hr={b['_hr0_forA']}-{b['_hr1_forA']}"\
                    f"_isnr={b['_isnr_forA']}_"\
                        f"seed={b['_seed_forA']}"
    
    str_A_setting = f"sensing_matrix_M={b['N_try_A']}_"\
        f"m={b['m']}_"\
            f"mode={b['mode_A']}_"\
                f"seed={b['seed_A']}"
    
    if b['mode_A']=='rakeness': 
        str_A_setting = str_A_setting + f"_loc={b['loc']}_corr={b['str_corr']}"
    
    path_A = os.path.join(dataset_dir, str_A_dataset, 
                        'A_Filippo', str_A_setting + '.pkl')
    return path_A

def from_bundle_to_ecg_path(b):
    b = copy.deepcopy(b)
    if b['isnr']>=100:
        b['isnr'] = 'None'

    str_ecg_setting = f"ecg_N={b['N']}_"\
        f"n={b['n']}_"\
            f"fs={b['fs']}_"\
                f"hr={b['hr0']}-{b['hr1']}"\
                    f"_isnr={b['isnr']}_"\
                        f"seed={b['seed_ecg']}"

    path_ecg = os.path.join(dataset_dir, str_ecg_setting + '.pkl')

    return path_ecg


def from_bundle_to_model(b, device=None, A=None, verbose=False):

    if device is None:
        use_cuda = torch.cuda.is_available()
        cuda_index = torch.cuda.device_count() - 1
        device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
        print(f"Using {device} device")

    path_model = from_bundle_to_model_path(b)
    
    
    if os.path.isfile(path_model):
        
        if b['str_A_init']:
            assert A is not None, 'cannot initialize A without input A'
            A_init = A
        else:
            A_init = None

        model = UNet(in_channels=b['in_channels'], out_channels=b['in_channels'],
                    expanded_channels=b['expanded_channels'], steps_num=b['step_number'],
                    kernel_size=b['kernel_size'], residual=b['residual'],
                    use_batch_norm=b['use_batch_norm'], simple_pool=b['simple_pool'],
                    x_as_input=b['str_x_as_input'], n=b['n'], m=b['m'], 
                    A_init=A_init, A_freeze=b['str_A_freeze'])

        model.to(device);

        model.load_state_dict(torch.load(path_model, weights_only=True))
        model.eval()

        if verbose: print(f"LOADED: {path_model}")


    else:
        model = None

        if verbose: print(f"MISSING: {path_model}")
    

    return model


def split_with_overlap(array, segment_length, overlap):
    step = segment_length - overlap
    segments = [array[i:i + segment_length] for i in range(0, len(array) - segment_length + 1, step)]
    return np.array(segments)


def _get_param_unet(expanded_channels, step_number, kernel_size):

    model = UNet(in_channels=in_channels, out_channels=in_channels,
                expanded_channels=expanded_channels, steps_num=step_number,
                kernel_size=kernel_size, residual=residual,
                use_batch_norm=use_batch_norm, simple_pool=simple_pool)

    model.eval()

    unet_param = get_size_model(model)

    d = {'unet_param': unet_param,
        'expanded_channels': expanded_channels,
        'step_number': step_number,
        'kernel_size': kernel_size,}
    
    return d

def _unet_many_param(list_expanded_channels, list_step_number, 
                     list_kernel_size, file_param=None, verbose=True):
    
    if file_param is not None:
        with open(file_param, 'wb') as f:
            pkl.dump([], f)

    for ch, st, kr in itertools.product(list_expanded_channels, list_step_number, list_kernel_size):

        d = _get_param_unet(ch, st, kr)

        if file_param is not None:
            with open(file_param, 'rb') as f:
                d_list = pkl.load(f)
        
        d_list += [d]
        
        if file_param is not None:
            with open(file_param, 'wb') as f:
                pkl.dump(d_list, f)
    if verbose:
        [print(_d) for _d in d_list]


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class DatasetFromSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    
class DatasetFromSubset_y_transform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    def __len__(self):
        return len(self.subset)
    

def from_ecg_to_df_loader(ecg, 
                          dict_A, 
                          in_channels=1, 
                          str_x_as_input=True, 
                          batch_size=256,
                          transform=None,
                          transform_train=True,
                          transform_val=False,
                          transform_test=False,
                          apply_transform_to_y=False,
                          ):

    if not(apply_transform_to_y):
        add_transform = lambda apply, subset: DatasetFromSubset(
            subset=subset, transform=transform 
        ) if apply and (transform is not None) else subset
    else:
        add_transform = lambda apply, subset: DatasetFromSubset_y_transform(
            subset=subset, transform=transform 
        ) if apply and (transform is not None) else subset

    df_loader = pd.DataFrame()

    for _mode_A, _A in dict_A.items():

        xall = x_to_dataset(ecg, _A, xbar_index=[1, 2])
        _in_channels = in_channels

        if str_x_as_input:
            samples = xall['x']
        else:
            if _in_channels==1:
                samples = xall['xbarT']
            elif _in_channels==2:
                samples = torch.cat((xall['xbarT'], xall['xbarPinv']), axis=-2)

        ds = [(xb_, x_) for xb_, x_ in zip(samples, xall['x'])]

        train_size = int(0.7 * len(ecg))
        val_size = int(0.1 * len(ecg))
        test_size = len(ecg) - train_size - val_size
        subsets = torch.utils.data.random_split(ds, [train_size, val_size, test_size])

        subsets = [add_transform(apply, _s) 
                   for _s, apply 
                   in zip(subsets,
                          [transform_train, transform_val, transform_test])]

        # if transform_train:
        #     train_dataset = DatasetFromSubset(
        #         subset=train_dataset,
        #         transform=transform)
        # if transform_val:    
        #     val_dataset = DatasetFromSubset(
        #         subset=val_dataset,
        #         transform=transform)
        # if transform_test:
        #     test_dataset = DatasetFromSubset_y_transform(
        #         subset=test_dataset,
        #         transform=transform)
        
        
        ### create DataLoaders
        
        
        loaders = [torch.utils.data.DataLoader(_subset,  batch_size=batch_size, shuffle=_shuffle)
                   for _subset, _shuffle in zip(subsets, [True, False, False])]

        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #       batch_size=batch_size,
        #         shuffle=True)
        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset,
        #       batch_size=batch_size,
        #         shuffle=False)
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset,
        #       batch_size=batch_size,
        #         shuffle=False)
        
        # df_loader = [pd.concat([df_loader, 
        #                         pd.DataFrame({'purpose': [_p],
        #                                       'mode_A': [_mode_A], 
        #                                       'data_loader': [_l]})])
        #             for _l, _p in zip(loaders, ['train', 'val', 'test'])]
        
        df_concat = [pd.DataFrame({'purpose': [_p],
                                    'mode_A': [_mode_A], 
                                    'data_loader': [_l]})
                    for _l, _p in zip(loaders, ['train', 'val', 'test'])]
        
        df_concat = pd.concat(df_concat)
        df_loader = pd.concat([df_loader, df_concat])

        # df_loader.update(
        #     {'purpose': 'train',
        #      'mode_A': _mode_A,
        #      'data_loader': train_loader})
        # dict_loader[_mode_A] = test_loader

    return df_loader
    
def test_model(model_list, param_unique_list, 
               bundle_loaded, dict_test_loader, 
               device, criterion, compute_rsnr):

    list_res = []
    for _model, _p_dict, _b_dict in tqdm(zip(model_list, 
                                             param_unique_list, 
                                             bundle_loaded), 
                                        total=len(model_list)):
        
        _t_loader = dict_test_loader[_b_dict['mode_A']]

        snr_ = 0.0
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in _t_loader:

                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = _model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                snr_ += np.mean(compute_rsnr(outputs.detach().cpu().numpy(),
                                            targets.detach().cpu().numpy()))
                
        test_loss /= len(_t_loader)
        snr_ /= len(_t_loader)

        _p_dict['SNR'] = snr_

        list_res += [_p_dict]

    return list_res


def load_real_ecg(db_dir, dl_dir, n, overlap, fs):

    if not(os.path.exists(dl_dir)):
        wfdb.io.dl_database(db_dir, dl_dir, records='all', annotators=None, keep_subdirs=True, overwrite=False)

    # ecg_real = np.empty(ecg.shape[1:])
    # ecg_real_filt = np.empty(ecg.shape[1:])

    ecg_real = None
    ecg_real_filt = None

    for _p in os.listdir(dl_dir):
        _path = os.path.join(dl_dir, _p)
        
        for _p2 in os.listdir(_path):

            if _p2[-4:] == '.dat':

                _p2 = _p2.replace('.dat', '')

                _path2 = os.path.join(_path, _p2)

                record = wfdb.io.rdrecord(_path2)
                
                _fs = record.fs

                record, record_filt = record.to_dataframe()['ECG I'].to_numpy(), record.to_dataframe()['ECG I filtered'].to_numpy()
                
                record = signal.resample_poly(record, fs, _fs)
                record_filt = signal.resample_poly(record_filt, fs, _fs)

                if ecg_real is None:
                    ecg_real = split_with_overlap(record, segment_length=n, overlap=overlap)
                    ecg_real_filt = split_with_overlap(record_filt, segment_length=n, overlap=n//5)

                else:
                    ecg_real = np.append(ecg_real, split_with_overlap(record, segment_length=n, overlap=overlap), axis=-2)
                    ecg_real_filt = np.append(ecg_real_filt, split_with_overlap(record_filt, segment_length=n, overlap=n//5), axis=-2)

    ecg_real = ecg_real[:, np.newaxis]
    ecg_real_filt = ecg_real_filt[:, np.newaxis]

    return ecg_real, ecg_real_filt