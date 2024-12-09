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

import os
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
modeA_default = 'standard'
str_orth_default = "True"
seed_A_default = 0
index_A_default = 0
N_try_A_default = 1_000
str_corr_default = '96af96a7ddfcb2f6059092c250e18f2a'
loc_default = 0.25
## Dataset used for generating A
_ecg_N_forA = 10_000
_n_forA = 128
_fs_forA = 256
_hr0_forA = 60
_hr1_forA = 100
_isnr_forA = 35
_seed_forA = 0

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
_out_channels = 1

#### HARDWARE PARAMETERS ####
str_gpus_default = "5"
threads_default = 1
workers_default = 1

#### TRAINING PARAMS ####
batch_size_default = 64

early_stopping_patience = 30
min_improvement = 1e-8
lr_plateaut_patience = 15
min_lr = 1e-6
lr_default = 0.001
criterion = nn.MSELoss()
num_epochs = 100_000


def cmdline_args():
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    #### DATASET PARAMETERS ####
    
    parser.add_argument(
        "-N", "--size_ecg",  type=int, default=size_ecg_default,
        help="number of ECG examples (default: %(default)s)"
    )
    parser.add_argument(
        "-n",  type=int, default=n_default,
        help="number of sample in each ECG example (default: %(default)s)"
    )
    parser.add_argument(
        "--seed_ecg", type=int, default=seed_ecg_default,
        help="seed used to generate ecg dataset"
    )
    parser.add_argument(
        "--isnr", type=int, default=isnr_default,
        help="intrinsic signal to noise ration of the dataset"
    )
    parser.add_argument(
        "--hr0", type=int, default=hr0_default,
        help="lower heart rate"
    )
    parser.add_argument(
        "--hr1", type=int, default=hr1_default,
        help="higher heart rate"
    )
    parser.add_argument(
        "--fs", type=int, default=fs_default,
        help="sampling frequency"
    )
    
    #### A PARAMETERS ####

    parser.add_argument(
        '-m', type=int, default=m_default,
        help="encoded dimension (default: %(default)s)"
    )
    parser.add_argument(
        '--modeA', type=str, default=modeA_default,
        help="whether A is standard or rakeness (default: %(default)s)"
    )
    parser.add_argument(
        "--seed_A", type=int, default=seed_A_default,
        help="seed used to find A with 'best_A_bpdn'"
    )
    parser.add_argument(
        "--index_A", type=int, default=index_A_default,
        help="chose what A in the best 8 saved by 'best_A_bpdn'"
    )
    parser.add_argument(
        "--str_orth", type=str, default=str_orth_default,
        help="whether to make the A with orth rows or not"
    )
    parser.add_argument(
        "--str_corr", type=str, default=str_corr_default,
        help="correlation matrix name file"
    )
    parser.add_argument(
        "--N_try_A", type=int, default=N_try_A_default,
        help="Samples of A between which the best will be saved"
    )
    parser.add_argument(
        "--loc", type=float, default=loc_default,
        help="localization of sensing matrix"
    )

    #### UNET PARAMETERS ####

    parser.add_argument(
        "--in_channels", type=int, default=in_channels_default,
        help=""
    )
    parser.add_argument(
        "--expanded_channels", type=int, default=expanded_channels_default,
        help=""
    )
    parser.add_argument(
        "--step_number", type=int, default=step_number_default,
        help=""
    )
    parser.add_argument(
        "--kernel_size", type=int, default=kernel_size_default,
        help=""
    )
    parser.add_argument(
        "--str_residual", type=str, default=str_residual_default,
        help=""
    )
    parser.add_argument(
        "--str_use_batch_norm", type=str, default=str_use_batch_norm_default,
        help=""
    )
    parser.add_argument(
        "--str_simple_pool", type=str, default=str_simple_pool_default,
        help=""
    )
    parser.add_argument(
        "--seed_torch", type=int, default=seed_torch_default,
        help="seed for training"
    )
    parser.add_argument(
        '--str_retrain', type=str, default=str_retrain_default,
        help="whether to retrain the model in case the model has\
              already been trained (default: %(default)s)"
    )
    parser.add_argument(
        '--str_resume_train', type=str, default=str_resume_train_default,
        help="whether to retrain the model in case the model has\
              already been trained (default: %(default)s)"
    )
    parser.add_argument(
        '--batch_size', type=int, default=batch_size_default,
        help="batch size for training (default: %(default)s)"
    )
    parser.add_argument(
        '--lr', type=float, default=lr_default,
        help="learning rate for the training (default: %(default)s)",
    )
    

    #### HARDWARE PARAMETERS ####

    parser.add_argument(
        "--str_gpus", type=str, default=str_gpus_default,
        help="gpus to use for training"
    )
    parser.add_argument(
        "--threads", type=int, default=threads_default,
        help="number of threads for cpu acceleration"
    )
    parser.add_argument(
        "--workers", type=int, default=workers_default,
        help="number of workers for cpu acceleration"
    )

    #### OTHER ####

    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="increase output verbosity (default: %(default)s)"
    )
    
    
    return parser.parse_args()



def main(N, 
         n, 
         m, 
         isnr,
         hr0,
         hr1,
         fs,
         mode_A, 
         seed_ecg, 
         seed_torch, 
         seed_A, 
         index_A,
         str_orth, 
         str_corr,
         N_try_A,
         loc,
         in_channels, 
         expanded_channels, 
         step_number, 
         kernel_size, 
         str_residual, 
         str_use_batch_norm, 
         str_simple_pool, 
         str_gpus,
         threads,
         workers,
         str_retrain, 
         str_resume_train, 
         batch_size,
         lr,):
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"Current Time = {current_time}\n")

    [print(f"{k}: {v}") for k, v in locals().items()];

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str_gpus

    #### from argparse type "str" to type "bool"
    orth = str_orth == "True"
    residual = str_residual == "True"
    use_batch_norm = str_use_batch_norm == "True"
    simple_pool = str_simple_pool == "True"
    str_retrain = str_retrain == "True"
    str_resume_train = str_resume_train == "True"

    #### Reproducibility
    torch_generator = my_funcs.set_all_torch_seed(seed_torch)


    ################# PATHS #################

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    #### Dataset (ecg)
    str_ecg_setting = f'ecg_N={N}_n={n}_fs={fs}_hr={hr0}-{hr1}'\
                      f'_isnr={isnr}_seed={seed_ecg}'
    path_ecg = os.path.join(dataset_dir, str_ecg_setting + '.pkl')
    

    #### Model
    str_model_setting = f'N={N}_n={n}_m={m}'\
                        f'_isnr={isnr}_seedecg={seed_ecg}_modeA={mode_A}_'\
                        f'seedtorch={seed_torch}_orth={orth}_'\
                        f'corr={str_corr}_'\
                        f'inchannels={in_channels}_'\
                        f'expandedchannels={expanded_channels}_'\
                        f'stepnumber={step_number}_'\
                        f'kernelsize={kernel_size}_residual={residual}_'\
                        f'usebatchnorm={use_batch_norm}_simplepool={simple_pool}'
    
    path_model = os.path.join(dataset_dir,
                              '..',
                              'trained_models',
                              'dummy_unet',
                              str_model_setting+'.pt')

    #### Sensing Matrix (A)
    str_A_dataset = f"ecg_N={_ecg_N_forA}_n={_n_forA}_"\
                    f"fs={_fs_forA}_hr={_hr0_forA}-{_hr1_forA}"\
                    f"_isnr={_isnr_forA}_seed={_seed_forA}"
    
    str_A_setting = f"sensing_matrix_M={N_try_A}_m={m}_mode={mode_A}_seed={seed_A}"
    if mode_A=='rakeness': 
        str_A_setting = str_A_setting + f"_loc={loc}_corr={str_corr}"
    path_A = os.path.join(dataset_dir, str_A_dataset, 
                          'A_Filippo', str_A_setting + '.pkl')

    ################# LOAD DATA #################

    #### Check if the model has already been trained    
    if os.path.isfile(path_model):
        assert str_retrain or str_resume_train, 'model has already been trained'
        if str_resume_train and str_retrain:
            warnings.warn('both str_resume_train and str_retrain are TRUE: str_resume_train PREVAILS')
        if str_resume_train:
            warnings.warn('The model has already been trained, but "str_resume_train=True"')
            time.sleep(5)
        elif str_retrain:
            warnings.warn('The model has already been trained, but "str_retrain=True"')
            time.sleep(5)
        
    print('\n...model training will start soon...\n')
    
    
    #### Load dataset (ecg)
    with open(path_ecg, 'rb') as f:
        ecg = pkl.load(f)[:, np.newaxis]

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
        time.sleep(5)
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

    #### Only take some of the generate elements
    ds = [(xb_, x_) for xb_, x_ in zip(xall['xbarT'], xall['x'])]

    #### Split data
    train_size = int(0.7 * len(ecg))
    val_size = int(0.1 * len(ecg))
    test_size = int(0.2 * len(ecg))
    train_dataset, val_dataset, test_dataset = random_split(ds, 
                                                            [train_size, val_size, test_size], 
                                                            generator=torch_generator)

    #### create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ################# HARDWARE SELECTION #################

    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    ################# MODEL #################

    model = UNet(in_channels=in_channels, 
                 out_channels=_out_channels,
                 expanded_channels=expanded_channels, 
                 steps_num=step_number, 
                 kernel_size=kernel_size,
                 residual=residual,
                 use_batch_norm=use_batch_norm, 
                 simple_pool=simple_pool)
    

    #### Loading weights if necessary
    if os.path.isfile(path_model) and str_resume_train:
        model.load_state_dict(torch.load(path_model, weights_only=True))
        print('\nLOADED WEIGHTS\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.2)

    model.to(device)

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

    print(f'TEST Loss: {test_loss}, SNR: {snr_:.2f}')
        

if __name__ == '__main__':

    args = cmdline_args()

    if args.verbose == 0:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose > 1:
        logger.setLevel(logging.DEBUG)

    print(args)
    logger.debug(str(args))

    main(
        args.size_ecg,      
        args.n,     
        args.m,     
        args.isnr,     
        args.hr0,     
        args.hr1,
        args.fs,     
        args.modeA,     
        args.seed_ecg,     
        args.seed_torch,     
        args.seed_A,     
        args.index_A,     
        args.str_orth,     
        args.str_corr,     
        args.N_try_A,     
        args.loc,
        args.in_channels,     
        args.expanded_channels,     
        args.step_number,     
        args.kernel_size,     
        args.str_residual,     
        args.str_use_batch_norm,     
        args.str_simple_pool,     
        args.str_gpus,     
        args.threads,     
        args.workers,     
        args.str_retrain,     
        args.str_resume_train,
        args.batch_size,
        args.lr,
        )