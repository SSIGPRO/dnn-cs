import numpy as np
import itertools
import os
import sys
import sys; sys.path.append('..')
from tqdm import tqdm
import pickle as pkl
import warnings

# import io
from contextlib import redirect_stdout

import torch
torch.manual_seed(0);
import torch.utils
import torch.utils.data


root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))


from dataset import dataset_dir
from models.unet import UNet

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
        if type(v) is list:
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

def from_bundle_to_model_path(b):

    str_model_setting = f"N={b['N']}_"\
        f"n={b['n']}_"\
            f"m={b['m']}_"\
                f"isnr={b['isnr']}_"\
                    f"seedecg={b['seed_ecg']}_"\
                        f"modeA={b['mode_A']}_"\
                            f"seedtorch={b['seed_torch']}_"\
                                f"orth={b['orth']}_"\
                                    f"corr={b['str_corr']}_"\
                                        f"inchannels={b['in_channels']}_"\
                                            f"expandedchannels={b['expanded_channels']}_"\
                                                f"stepnumber={b['step_number']}_"\
                                                    f"kernelsize={b['kernel_size']}_"\
                                                        f"residual={b['residual']}_"\
                                                            f"usebatchnorm={b['use_batch_norm']}_"\
                                                                f"simplepool={b['simple_pool']}"
    
    path_model = os.path.join(dataset_dir,
                            '..',
                            'trained_models',
                            'dummy_unet',
                            str_model_setting+'.pt')
    
    return path_model

def from_bundle_to_A_path(b):
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
        
    str_ecg_setting = f"ecg_N={b['N']}_"\
        f"n={b['n']}_"\
            f"fs={b['fs']}_"\
                f"hr={b['hr0']}-{b['hr1']}"\
                    f"_isnr={b['isnr']}_"\
                        f"seed={b['seed_ecg']}"

    path_ecg = os.path.join(dataset_dir, str_ecg_setting + '.pkl')

    return path_ecg


def from_bundle_to_model(b, device=None, verbose=False):

    if device is None:
        use_cuda = torch.cuda.is_available()
        cuda_index = torch.cuda.device_count() - 1
        device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
        print(f"Using {device} device")

    path_model = from_bundle_to_model_path(b)
    
    
    if os.path.isfile(path_model):
        
        model = UNet(in_channels=b['in_channels'], num_classes=b['in_channels'],
                    expanded_channels=b['expanded_channels'], steps_num=b['step_number'],
                    kernel_size=b['kernel_size'], residual=b['residual'],
                    use_batch_norm=b['use_batch_norm'], simple_pool=b['simple_pool'])

        model.to(device);

        model.load_state_dict(torch.load(path_model, weights_only=True))
        model.eval()

        if verbose: print(f"LOADED: {path_model}")


    else:
        model = None

        if verbose: print(f"MISSING: {path_model}")
    

    return model




def _get_param_unet(expanded_channels, step_number, kernel_size):

    model = UNet(in_channels=in_channels, num_classes=in_channels,
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