import os
import sys
# limit number of parallel threads numpy spawns
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pickle
import argparse
import logging

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from models import models_dir
from dataset import dataset_dir
from cs.training_metrics import rsnr
from models.ae import AE

logging.basicConfig(level=logging.DEBUG)

def test(
    n, m, epochs, lr, opt, batch_size, N_train, N_test, fs, heart_rate, isnr,
    seed_training, processes, gpu, train_fraction, factor, 
    min_lr, min_delta, patience
):

    # ------------------ Constants ------------------
    seed_train_data = 11  # seed relative to training datas
    seed_test_data = 66

    # ------------------ Folders ------------------
    results_folder = '/srv/newpenny/dnn-cs/tsoc/results/TSOC/reconstruction'
    model_folder = f'{models_dir}AE'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running training with parameters: {params_str}")

    # ------------------ GPU ------------------
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # ------------------ Signal ------------------
    data_name = f'ecg_N={N_test}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_test_data}'
    data_path = os.path.join(dataset_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)

    # ------------------ Data loaders ------------------
    test_dataset = TensorDataset(torch.from_numpy(X).float())  # Create a dataset from the tensor
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=processes) 

    # ------------------ Neural Network initialization ------------------
    ae = AE(n, m)
    ae.to(device) # move the network to GPU
    model_name = f'AE-N={N_train}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                 f'_isnr={isnr}'\
                 f'_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                 f'_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                 f'_mind={min_delta}_seeddata={seed_train_data}_seedtrain={seed_training}'    
    model_path = os.path.join(model_folder, f'{model_name}.pth')
    ae.load_state_dict(torch.load(model_path, weights_only=True))
    ae.eval()     # Set the model to evaluation mode

    results_path = os.path.join(results_folder, f'{model_name}.pkl')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if os.path.exists(results_path):
        print(f'Model\n{model_name}\nhas already been evaluated')
        sys.exit(0)

    # ------------------ Compute test metric ------------------
    results = pd.Series(index=['RSNR'], dtype=np.float64)
    rsnr_test = 0
    with torch.no_grad():     # disables gradient calculation for the validation phase 
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)     # move validation data to GPU
            output = ae(X_batch)
            rsnr_test += rsnr(output, X_batch)

    num_batches = len(test_loader)
    rsnr_test =  rsnr_test / num_batches
    print(f"RSNR = {rsnr_test.item()}") 
    results['RSNR'] = rsnr_test.item()

    # save peformance
    pd.to_pickle(pd.Series(results), results_path)
# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for AE Model")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--isnr', '-i', type=int, required=True, help="Signal-to-noise ratio (SNR)")
    parser.add_argument('--seed_training', '-s', type=int, required=True, help="Training-related random seed for reproducibility")
    parser.add_argument('--epochs', '-e', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=50, help="Batch size for training")
    parser.add_argument('--N_train', '-N', type=int, default=2_000_000, help="Number of training samples")
    parser.add_argument('--N_test', '-M', type=int, default=10000, help="Number of test samples")
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer for training")
    parser.add_argument('--fs', '-f', type=int, default=256, help="Sampling frequency")
    parser.add_argument('--heart_rate', '-r', type=int, nargs=2, default=(60, 100), help="Heart rate range")
    parser.add_argument('--processes', '-p', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('--gpu', '-g', type=int, default=3, help="GPU index to use for training")
    parser.add_argument('--train_fraction', '-T', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('--factor', '-F', type=float, default=0.2, help="Factor for ReduceLROnPlateau scheduler")
    parser.add_argument('--min_lr', '-L', type=float, default=0.001, help="Minimum learning rate")
    parser.add_argument('--min_delta', '-d', type=float, default=1e-4, help="Minimum delta for early stopping and ReduceLROnPlateau")
    parser.add_argument('--patience', '-P', type=int, default=40, help="Patience for early stopping")
    
    return parser.parse_args()

# ------------------ Main call ------------------
if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()
    
    # Call the training function with parsed arguments
    test(
        n=args.n,
        m=args.m,
        epochs=args.epochs,
        lr=args.lr,
        opt=args.optimizer,
        batch_size=args.batch_size,
        N_train=args.N_train,
        N_test=args.N_test,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
        seed_training=args.seed_training,
        processes=args.processes,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        factor=args.factor,  
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience
    )
