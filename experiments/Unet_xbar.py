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
 
import os
import pickle as pkl
import numpy as np

import sys ; sys.path.append('..')  # useful if you're running locally

# import of local modules
root = os.path.dirname(os.path.realpath('__file__'))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir
from cs import CompressedSensing, generate_sensing_matrix
from cs.utils import compute_rsnr
from models.unet import UNet


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

os.environ['SCIPY_USE_PROPACK'] = "True"
 
threads = "64"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads



logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

hr = (60, 100)
fs = 256
isnr = 35

loc = 0.25
corr_str = '96af96a7ddfcb2f6059092c250e18f2a' # name of corr matrix (file)

# mode_A = 'standard'
orth = True

def cmdline_args():
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-s", "--size",  type=int, default=10_000,
        help="number of ECG examples (default: %(default)s)"
    )
    parser.add_argument(
        "-l", "--length",  type=int, default=128,
        help="number of sample in each ECG example (default: %(default)s)"
    )
    parser.add_argument(
        '-m', type=int, default=48,
        help="encoded dimension (default: %(default)s)"
    )
    parser.add_argument(
        '--modeA', type=str, default='standard',
        help="whether A is standard or rakeness (default: %(default)s)"
    )
    parser.add_argument(
        "--seed_ecg", type=int, default=0,
        help="random seed"
    )
    parser.add_argument(
        "--seed_torch", type=int, default=0,
        help="random seed"
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="increase output verbosity (default: %(default)s)"
    )
    
    return parser.parse_args()



def main(N, n, m, mode_A, seed_ecg, seed_torch):

    torch.manual_seed(seed_torch) # torch see

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    str_ecg_setting = f'ecg_N={N}_n={n}_fs={fs}_hr={hr[0]}-{hr[1]}'\
                      f'_isnr={isnr}_seed={seed_ecg}'

    str_model_setting = f'ecg_N={N}_n={n}_m={m}'\
                        f'_isnr={isnr}_seed={seed_ecg}_modeA={mode_A}'
    
    N_try_A = 1_000
    seed_A = 0
    
    str_A_setting = f'sensing_matrix_M={N_try_A}_m={m}_mode={mode_A}_seed={seed_A}'
    if mode_A=='rakeness': str_A_setting = str_A_setting + f'_loc={loc}_corr={corr_str}'


    path_ecg = os.path.join(dataset_dir, str_ecg_setting + '.pkl')
    
    path_A = os.path.join(dataset_dir, str_ecg_setting, 
                          'A_Filippo', str_A_setting + '.pkl')

    path_model = os.path.join(dataset_dir,
                              '..',
                              'trained_models',
                              'dummy_unet',
                              str_model_setting+'.pt')

    with open(path_ecg, 'rb') as f:
        ecg = pkl.load(f)[:, np.newaxis]

    print(path_A)

    if os.path.isfile(path_A):
        with open(path_A, 'rb') as f:
            A = pkl.load(f)[0]
        print(f"Loading sensing matrix.",
                f"({A['rsnr']:.2f}dB with bpdn, seed:{A['seed']})")
        A = A['matrix']

    else:
        warnings.warn("Sensing matrix generation. "\
                      "Generate with 'best_A_bpdn' for better performances")
        
        path_corr = os.path.join(dataset_dir, 'correlation', corr_str + '.pkl')
        with open(path_corr, 'rb') as f: corr = pkl.load(f)
        
        A = generate_sensing_matrix(
                    shape = (m, n), 
                    mode=mode_A, 
                    antipodal=False, 
                    orthogonal=orth,
                    correlation=corr,
                    loc=.25, 
                    seed=seed_A)
        print('generated A')
        print('A shape:', A.shape)

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

        if x.shape[-1] == 1:
            d_outputs = {k: np.swapaxes(v, -1, -2) for k, v in d_outputs.items()}
        
        ## change to torch tensor specified type
        d_outputs = {k: torch.tensor(v, dtype=torch_dtype) for k, v in d_outputs.items()}

        if verbose:
            print('out contains:', d_outputs.keys())
        return d_outputs
    

    xall = x_to_dataset(ecg, A, xbar_index=[1, 2])
    [print(k, 'SHAPE:', v.shape) for k, v in xall.items()];

    ds = [(xb_, x_) for xb_, x_ in zip(xall['xbarT'], xall['x'])]

    train_size = int(0.7 * len(ecg))
    val_size = int(0.1 * len(ecg))
    test_size = int(0.2 * len(ecg))
    train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])

    # create DataLoaders
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    early_stopping_patience = 30
    min_improvement = 1e-9
    lr_plateaut_patience = 15
    min_lr = 1e-6

    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    channels = 1

    model = UNet(in_channels=channels, num_classes=channels,
                channels=64, steps_num=4, kernel_size=3, residual=True,
                use_batch_norm=True, simple_pool=False)
    

    lr = 0.001
    criterion = nn.MSELoss()
    # criterion = nn.MAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.2)

    model.to(device)

    num_epochs = 1000
    patience_counter = {'stop': 0, 'lr': 0}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # calculate validation loss
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
                

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        snr_ /= len(val_loader)
        
        last_lr = lr_scheduler.get_last_lr()[0]

        # scheduler.step(val_loss)
        print(f'Epoch [{epoch}/{num_epochs}], ' 
              f'Train Loss: {train_loss:.6f}, '
              f'Val Loss: {val_loss:.6f}, SNR: {snr_:.2f} -- '
              f'lr: {last_lr}')
        
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
        args.size, 
        args.length,
        args.m,
        args.modeA,
        args.seed_ecg,
        args.seed_torch,
    )