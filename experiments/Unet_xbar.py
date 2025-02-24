import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, random_split

import time
import logging
import argparse
import warnings
from datetime import datetime
import wfdb
from scipy import signal

import os
import itertools
import pickle as pkl

import functions_Filippo as my_funcs


os.environ['SCIPY_USE_PROPACK'] = "True"

threads = "64"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
import numpy as np

import sys ; sys.path.append('..')  # useful if you're running locally

# import of local modules
root = os.path.dirname(os.path.realpath('__file__'))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir
from cs import CompressedSensing, generate_sensing_matrix
from cs.utils import compute_rsnr
from models.unet import UNet




logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# #### DATASET PARAMETERS ####
# size_ecg_default = 10_000
# n_default = 128
# seed_ecg_default = 0
# isnr_default = 35
# hr0_default = 60
# hr1_default = 100
# fs_default = 256

# #### A PARAMTERS ####
# m_default = 48
# modeA_default = 'rakeness'
# str_orth_default = "True"
# seed_A_default = 0
# index_A_default = 0
# N_try_A_default = 1_000
# str_corr_default = '96af96a7ddfcb2f6059092c250e18f2a'
# loc_default = 0.25
# ## Dataset used for generating A
# _ecg_N_forA_default = 10_000
# _n_forA_default = 128
# _fs_forA_default = 256
# _hr0_forA_default = 60
# _hr1_forA_default = 100
# _isnr_forA_default = 35
# _seed_forA_default = 0

# #### UNET PARAMETERS ####
# in_channels_default = 1
# expanded_channels_default = 64
# step_number_default = 4
# kernel_size_default = 3
# str_residual_default = "True"
# str_use_batch_norm_default = "True"
# str_simple_pool_default = "False"
# seed_torch_default = 0
# str_retrain_default = "False"
# str_resume_train_default = "False"
# out_channels_default = 1
# str_x_as_input_default = "True"
# str_A_init_default = "True"
# str_A_freeze_default = "True"

# #### HARDWARE PARAMETERS ####
# str_gpus_default = "5"
# threads_default = 1
# workers_default = 1

# #### TRAINING PARAMS ####
# batch_size_default = 64
# white_noise_var_default = 0.0
# white_noise_isnr_default = 100

early_stopping_patience = 30
min_improvement = 1e-8
lr_plateaut_patience = 15
min_lr = 1e-6
lr_default = 0.001
criterion = nn.MSELoss()
num_epochs_default = 100_000


def cmdline_args():
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_default = my_funcs.get_bundle_default()

    #### DATASET PARAMETERS ####
    
    parser.add_argument(
        "-N", "--size_ecg",  type=int, dest='N', default=args_default['size_ecg_default'],
        help="number of ECG examples (default: %(default)s)"
    )
    parser.add_argument(
        "-n",  type=int, default=args_default['n_default'],
        help="number of sample in each ECG example (default: %(default)s)"
    )
    parser.add_argument(
        "--seed_ecg", type=int, default=args_default['seed_ecg_default'],
        help="seed used to generate ecg dataset"
    )
    parser.add_argument(
        "--isnr", type=int, default=args_default['isnr_default'],
        help="intrinsic signal to noise ration of the dataset"
    )
    parser.add_argument(
        "--hr0", type=int, default=args_default['hr0_default'],
        help="lower heart rate"
    )
    parser.add_argument(
        "--hr1", type=int, default=args_default['hr1_default'],
        help="higher heart rate"
    )
    parser.add_argument(
        "--fs", type=int, default=args_default['fs_default'],
        help="sampling frequency"
    )
    
    #### A PARAMETERS ####

    parser.add_argument(
        '-m', type=int, default=args_default['m_default'],
        help="encoded dimension (default: %(default)s)"
    )
    parser.add_argument(
        '--modeA', type=str, dest='mode_A', default=args_default['modeA_default'],
        help="whether A is standard or rakeness (default: %(default)s)"
    )
    parser.add_argument(
        "--seed_A", type=int, default=args_default['seed_A_default'],
        help="seed used to find A with 'best_A_bpdn'"
    )
    parser.add_argument(
        "--index_A", type=int, default=args_default['index_A_default'],
        help="chose what A in the best 8 saved by 'best_A_bpdn'"
    )
    parser.add_argument(
        "--str_orth", type=str, dest='orth', default=args_default['str_orth_default'],
        help="whether to make the A with orth rows or not"
    )
    parser.add_argument(
        "--str_corr", type=str, default=args_default['str_corr_default'],
        help="correlation matrix name file"
    )
    parser.add_argument(
        "--N_try_A", type=int, default=args_default['N_try_A_default'],
        help="Samples of A between which the best will be saved"
    )
    parser.add_argument(
        "--loc", type=float, default=args_default['loc_default'],
        help="localization of sensing matrix"
    )
    ## dataet for A generation
    parser.add_argument(
        "--_ecg_N_forA", type=int, default=args_default['_ecg_N_forA_default'],
        help="number of samples of dataset for A gen"
    )
    parser.add_argument(
        "--_n_forA", type=int, default=args_default['_n_forA_default'],
        help="length of ecg for A gen"
    )
    parser.add_argument(
        "--_fs_forA", type=int, default=args_default['_fs_forA_default'],
        help="sampling frequency of the dataset for A gen"
    )
    parser.add_argument(
        "--_hr0_forA", type=int, default=args_default['_hr0_forA_default'],
        help="heart rate low of the dataset for A gen"
    )
    parser.add_argument(
        "--_hr1_forA", type=int, default=args_default['_hr1_forA_default'],
        help="heart rate hight of the dataset for A gen"
    )
    parser.add_argument(
        "--_isnr_forA", type=int, default=args_default['_isnr_forA_default'],
        help="isnr of the dataset for A gen"
    )
    parser.add_argument(
        "--_seed_forA", type=int, default=args_default['_seed_forA_default'],
        help="seed of the dataset for A gen"
    )

    #### UNET PARAMETERS ####

    parser.add_argument(
        "--in_channels", type=int, default=args_default['in_channels_default'],
        help=""
    )
    parser.add_argument(
        "--expanded_channels", type=int, default=args_default['expanded_channels_default'],
        help=""
    )
    parser.add_argument(
        "--step_number", type=int, default=args_default['step_number_default'],
        help=""
    )
    parser.add_argument(
        "--kernel_size", type=int, default=args_default['kernel_size_default'],
        help=""
    )
    parser.add_argument(
        "--str_residual", type=str, dest='residual', default=args_default['str_residual_default'],
        help=""
    )
    parser.add_argument(
        "--str_use_batch_norm", type=str, dest='use_batch_norm', default=args_default['str_use_batch_norm_default'],
        help=""
    )
    parser.add_argument(
        "--str_simple_pool", type=str, dest='simple_pool', default=args_default['str_simple_pool_default'],
        help=""
    )
    parser.add_argument(
        "--seed_torch", type=int, default=args_default['seed_torch_default'],
        help="seed for training"
    )
    parser.add_argument(
        '--str_retrain', type=str, default=args_default['str_retrain_default'],
        help="whether to retrain the model in case the model has\
              already been trained (default: %(default)s)"
    )
    parser.add_argument(
        '--str_resume_train', type=str, default=args_default['str_resume_train_default'],
        help="whether to retrain the model in case the model has\
              already been trained (default: %(default)s)"
    )
    parser.add_argument(
        '--batch_size', type=int, default=args_default['batch_size_default'],
        help="batch size for training (default: %(default)s)"
    )
    parser.add_argument(
        '--lr', type=float, default=args_default['lr_default'],
        help="learning rate for the training (default: %(default)s)",
    )
    parser.add_argument(
        '--out_channels', type=float, default=args_default['out_channels_default'],
        help="number of channels of output (default: %(default)s)",
    )
    parser.add_argument(
        '--white_noise_var', type=float, default=args_default['white_noise_var_default'],
        help="variance of white noise added to dataset (default: %(default)s)",
    )
    parser.add_argument(
        '--white_noise_isnr', type=float, default=args_default['white_noise_isnr_default'],
        help="isnr to reach by adding white noise to dataset (default: %(default)s)",
    )
    parser.add_argument(
        '--str_x_as_input', type=str, default=args_default['str_x_as_input_default'],
        help="use x instead of ATy as input to the model (default: %(default)s)",
    )
    parser.add_argument(
        '--str_A_init', type=str, default=args_default['str_A_init_default'],
        help="whether to initialize the enc layer with A (default: %(default)s)",
    )
    parser.add_argument(
        '--str_A_freeze', type=str, default=args_default['str_A_init_default'],
        help="whether to freeze training for A (enc and first guess) (default: %(default)s)",
    )

    #### HARDWARE PARAMETERS ####

    parser.add_argument(
        "--str_gpus", type=str, default=args_default['str_gpus_default'],
        help="gpus to use for training"
    )
    parser.add_argument(
        "--threads", type=int, default=args_default['threads_default'],
        help="number of threads for cpu acceleration"
    )
    parser.add_argument(
        "--workers", type=int, default=args_default['workers_default'],
        help="number of workers for cpu acceleration"
    )

    #### OTHER ####

    parser.add_argument(
        '--num_epochs', action='store', type=int, default=args_default['num_epochs_default'],
        help="max epochs traning (default: %(default)s)"
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="increase output verbosity (default: %(default)s)"
    )
    
    
    return parser.parse_args()



def main(**bundle):

    print()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"Current Time = {current_time}\n")
    print()
    #### from argparse type "str" to type "bool"
    [bundle.update({k: v=="True"}) for k, v in bundle.items() if v=="True" or v=="False"]

    #### from argparse to variables
    N = bundle['N'] 
    n = bundle['n'] 
    m = bundle['m'] 
    isnr = bundle['isnr']
    hr0 = bundle['hr0']
    hr1 = bundle['hr1']
    fs = bundle['fs']
    mode_A = bundle['mode_A'] 
    seed_ecg = bundle['seed_ecg'] 
    seed_torch = bundle['seed_torch'] 
    seed_A = bundle['seed_A'] 
    index_A = bundle['index_A']
    orth = bundle['orth']
    str_corr = bundle['str_corr']
    N_try_A = bundle['N_try_A']
    loc = bundle['loc']
    in_channels = bundle['in_channels']
    out_channels = bundle['out_channels']
    white_noise_var = bundle['white_noise_var']
    expanded_channels = bundle['expanded_channels'] 
    step_number = bundle['step_number'] 
    kernel_size = bundle['kernel_size'] 
    residual = bundle['residual']
    use_batch_norm = bundle['use_batch_norm']
    simple_pool = bundle['simple_pool']
    str_gpus = bundle['str_gpus']
    threads = bundle['threads']
    workers = bundle['workers']
    str_retrain = bundle['str_retrain']
    str_resume_train = bundle['str_resume_train']
    batch_size = bundle['batch_size']
    lr = bundle['lr']
    _ecg_N_forA = bundle['_ecg_N_forA']
    _n_forA = bundle['_n_forA']
    _fs_forA = bundle['_fs_forA']
    _hr0_forA = bundle['_hr0_forA']
    _hr1_forA = bundle['_hr1_forA']
    _isnr_forA = bundle['_isnr_forA']
    _seed_forA = bundle['_seed_forA']
    white_noise_var = bundle['white_noise_var']
    white_noise_isnr = bundle['white_noise_isnr']
    str_x_as_input = bundle['str_x_as_input']
    str_A_init = bundle['str_A_init']
    str_A_freeze = bundle['str_A_freeze']
    num_epochs = bundle['num_epochs']

    [print(f"{k} = {v}") for k, v in locals().items()];

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str_gpus

    #### Reproducibility
    torch_generator = my_funcs.set_all_torch_seed(seed_torch)


    ################# PATHS #################

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    #### Dataset (ecg)
    path_ecg = my_funcs.from_bundle_to_ecg_path(bundle)

    # #### Model
    path_model = my_funcs.from_bundle_to_model_path(bundle)
    print('\nPATH MODEL:')
    print(path_model)
    print('length path model:', len(path_model))

    #### Sensing Matrix (A)
    path_A = my_funcs.from_bundle_to_A_path(bundle)

    ################# LOAD DATA #################

    #### Check if the model has already been trained    
    if os.path.isfile(path_model):
        if str_retrain==False and str_resume_train==False:
            warnings.warn('model has already been trained, and both str_resume_train and str_retrain are FALSE.')
            return None
        elif str_resume_train and str_retrain:
            warnings.warn('both str_resume_train and str_retrain are TRUE: str_resume_train PREVAILS')
        elif str_resume_train:
            warnings.warn('The model has already been trained, but "str_resume_train=True"')
            time.sleep(5)
        elif str_retrain:
            warnings.warn('The model has already been trained, but "str_retrain=True"')
            time.sleep(5)
        
    print('\n...model training will start soon...\n')
    
    
    #### Load dataset (ecg)
    with open(path_ecg, 'rb') as f:
        ecg = pkl.load(f)[:, np.newaxis]

    # find noise level
    if white_noise_var==0 and white_noise_isnr<100:
        axis = -1
        norm_signal = np.mean(np.linalg.norm(ecg, axis=axis))
        norm_noise = norm_signal / 10**(white_noise_isnr / 20)
        white_noise_var = norm_noise / np.sqrt(ecg.shape[axis])
        _n_ecg = torch.tensor(ecg) + torch.randn(np.shape(ecg)) * white_noise_var
        print(f'EXPECTED white noise ISNR: {white_noise_isnr}')
        print(f'REAL white noise ISNR: {np.mean(compute_rsnr(_n_ecg, ecg))}')
        del _n_ecg

    elif white_noise_var>0 and white_noise_isnr<100:
        warnings.warn('white_noise_var>0 and white_noise_isnr<100. Using white noise var!')
        time.sleep(5)

    #### Load sensing Matrix (A)

    print("\nLOADING SENSING MATRIX (A)\n")
    print(path_A)
    print()


    if os.path.isfile(path_A):
        with open(path_A, 'rb') as f:
            A = pkl.load(f)[index_A]
        print(f"Loading sensing matrix.",
              f"(rsnr: {A['rsnr']:.2f}dB (with bpdn) -- seed: {A['seed']})")
        A = A['matrix']

    else: # if not loaded raise warnings and generate one
        warnings.warn("Sensing matrix generation. "\
                      "Generate with 'best_A_bpdn' for better performances")
        time.sleep(5)
        warnings.warn('You probably want to kill this process generate the sensing matrix')
        warnings.warn(path_A)
        time.sleep(20)
        path_corr = os.path.join(dataset_dir, 'correlation', str_corr + '.pkl')
        with open(path_corr, 'rb') as f: corr = pkl.load(f)
        
        A = generate_sensing_matrix(
                    shape = (m, n), 
                    mode=mode_A, 
                    antipodal=False, 
                    orthogonal=orth,
                    correlation=corr,
                    loc=loc, 
                    seed=seed_A)
        print('generated A')
        print('A shape:', A.shape)

    ################# FROM DATA TO DATASET #################

    #### Generate the input and labels for model
    xall = my_funcs.x_to_dataset(ecg, A, xbar_index=[1, 2])
    [print(k, 'SHAPE:', v.shape) for k, v in xall.items()]

    if str_x_as_input:
        samples = xall['x']
    else:
        if in_channels==1:
            samples = xall['xbarT']
        elif in_channels==2:
            samples = torch.cat((xall['xbarT'], xall['xbarPinv']), axis=-2)

    ds = [(xb_, x_) for xb_, x_ in zip(samples, xall['x'])]

    print(f"dataset size: {len(ds)}")

    #### Split data
    train_size = int(0.7 * len(ecg))
    val_size = int(0.1 * len(ecg))
    test_size = int(0.2 * len(ecg))
    train_dataset, val_dataset, test_dataset = random_split(ds, 
                                                            [train_size, val_size, test_size], 
                                                            generator=torch_generator)
    
    if white_noise_var>0:    
        train_dataset = my_funcs.DatasetFromSubset(
            subset=train_dataset,
            transform=my_funcs.AddGaussianNoise(std=white_noise_var))
        
        val_dataset = my_funcs.DatasetFromSubset(
            subset=val_dataset,
            transform=my_funcs.AddGaussianNoise(std=white_noise_var))

    #### create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    _rsnr_val = 0
    for _x, _y in val_dataset:
        _rsnr_val += compute_rsnr(_x, _y)
    print(f"RSNR on valset {_rsnr_val/len(val_dataset)}")
    
    ################# HARDWARE SELECTION #################

    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    ################# MODEL #################

    if str_A_init:
        str_A_init = A
    else:
        str_A_init = None

    model = UNet(in_channels=in_channels, 
                 out_channels=out_channels,
                 expanded_channels=expanded_channels, 
                 steps_num=step_number, 
                 kernel_size=kernel_size,
                 residual=residual,
                 use_batch_norm=use_batch_norm, 
                 simple_pool=simple_pool,
                 x_as_input=str_x_as_input,
                 n=n,
                 m=m,
                 A_init=str_A_init,
                 A_freeze=str_A_freeze)
    

    #### Loading weights if necessary
    if os.path.isfile(path_model) and str_resume_train:
        model.load_state_dict(torch.load(path_model, weights_only=True))
        print('\nLOADED WEIGHTS\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.2)

    model.to(device)


    snr_ = 0.0
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            snr_ += np.mean(compute_rsnr(outputs.detach().cpu().numpy(),
                                        targets.detach().cpu().numpy()))
            
    test_loss /= len(test_loader)
    snr_ /= len(test_loader)

    print(f'BEFORE TRAINING - TEST Loss: {test_loss}, SNR: {snr_:.2f}')

    ################# TRAINING #################

    patience_counter = {'stop': 0, 'lr': 0}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        #### TRAIN
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        
        #### VALIDATION
        model.eval()
        val_loss = 0.0
        snr_ = 0.0
        
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                snr_ += np.mean(compute_rsnr(outputs.detach().cpu().numpy(),
                                            targets.detach().cpu().numpy()))
            
        #### METRICS
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        snr_ /= len(val_loader)
        
        last_lr = lr_scheduler.get_last_lr()[0]

        #### VERBOSE
        print(f'Epoch [{epoch}/{num_epochs}], ' 
              f'Train Loss: {train_loss:.6f}, '
              f'Val Loss: {val_loss:.6f}, SNR: {snr_:.2f} -- '
              f'lr: {last_lr} -- '
              f"time: {datetime.now()}")

        #### CHECK ON A FREEZ
        if str_A_freeze:
            _A = model.encoder.weight.data.clone().detach()
            assert torch.equal(_A.cpu(), torch.tensor(A, dtype=torch.float32)), 'miao'

        ##### CALLBACKS
        # Early stop
        # LR on Plateau
        # Model checkpoint 
        
        if best_val_loss - val_loss >  min_improvement:
            best_val_loss = val_loss
            for k in patience_counter.keys():
                patience_counter[k] = 0 

            torch.save(model.state_dict(), path_model)

            print('Saving model: validation loss improved')
        
        else:
            for k, v in patience_counter.items():
                patience_counter[k] = v+1 

            if patience_counter['stop'] >= early_stopping_patience:
                print("Early stopping: Validation loss hasn't improved for", early_stopping_patience, "epochs.")
                break

            if patience_counter['lr'] >= lr_plateaut_patience and last_lr > min_lr:
                print("Reduce LR: Validation loss hasn't improved for", 
                      lr_plateaut_patience, "epochs.")
                patience_counter['lr'] = 0 
                lr_scheduler.step()

    ################# EVALUATION ON TEST #################            

    snr_ = 0.0
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            snr_ += np.mean(compute_rsnr(outputs.detach().cpu().numpy(),
                                        targets.detach().cpu().numpy()))
            
    test_loss /= len(test_loader)
    snr_ /= len(test_loader)

    ###### FINE TUNE ON REAL ECG



    print(f'TEST Loss: {test_loss}, SNR: {snr_:.2f}')
        

if __name__ == '__main__':

    args = cmdline_args()
    kwargs = dict(args._get_kwargs())
    if args.verbose == 0:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose > 1:
        logger.setLevel(logging.DEBUG)

    [print(f"{k}: {v}") for k, v in kwargs.items()]

    logger.debug(str(args))

    main(**kwargs)