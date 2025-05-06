import os

os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import sys
import torch
from torch.utils.data import random_split

import numpy as np
from scipy.fft import dct
import pickle
import argparse
import logging

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from models import models_dir
from dataset import dataset_dir
from cs.wavelet_basis import wavelet_basis
from detectors.tsoc_bridge import TSOCDetector
from detectors import detectors_dir
from cs import CompressedSensing, generate_sensing_matrix

logging.basicConfig(level=logging.DEBUG)

def training(
    n, m, epochs, window_size, fraction, lr, opt, batch_size, fs, mode, orthogonal, source,
      seed_matrix, alpha, seed_training, processes, threshold, gpu, train_fraction, factor, 
    min_lr, min_delta, patience, version
):

    # ------------------ Constants ------------------
    M = 100 # number of evaluated sensing matrices
    seed_data = 0
    seed_training = 0
    support_method = 'TSOC'

    # ------------------ Folders ------------------
    model_folder = f'{models_dir}{support_method}'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running training with parameters: {params_str}")

    
    # ------------------ Reproducibility ------------------
    # Set the seed for PyTorch (CPU)
    torch.manual_seed(seed_training)

    # Set the seed for PyTorch (GPU)
    torch.cuda.manual_seed(seed_training)
    torch.cuda.manual_seed_all(seed_training)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        # Ensure worker processes have deterministic behavior
        worker_seed = torch.initial_seed() % 2**32
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)

    # Use a PyTorch generator to control randomness
    generator = torch.Generator()
    generator.manual_seed(seed_training)

    # ------------------ GPU ------------------
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # ------------------ Signal ------------------
    # load training data
    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)

    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_fraction={fraction}_seed={seed_data}'
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_augmented = pickle.load(f)

    # ------------------ Compressed Sensing ------------------
    D = dct(np.eye(n), axis=0, norm='ortho')

    # Sensing matrix
    C = (X.T @ X) / len(X)
    loc = np.trace(C @ C) / np.trace(C)**2 - 1 / n
    if source == 'random':
        A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orthogonal, correlation=C, loc=loc, seed=seed_matrix)
    elif source == 'best':
        fname =f'matrices_ort={orthogonal}_mode={mode}_M={M}_m={m}_mode={mode}_seed={seed_data}'
        path = os.path.join(files_dir, f'Best_sensing_matrices', fname+'.pkl')
        with open(path, 'rb') as f:
            A_dict = pickle.load(f)
        A = A_dict[seed_matrix]['matrix']
        seed_matrix = A_dict[seed_matrix]['seed']

    cs = CompressedSensing(A, D, decoder='SO')

    # ------------------ Path initialization ------------------
    if version == 'v1':
        architecture = 'TSOC'
    elif version == 'v2':
        architecture = 'TSOCv2'
    model_name = f'bridge-{architecture}-N={len(X_augmented)}_n={n}_ws={window_size}_fraction={fraction}_m={m}_fs={fs}_mode={mode}'\
                 f'_src={source}_ort={orthogonal}_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                 f'_th={threshold}_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                 f'_mind={min_delta}_seedtraining={seed_training}_seedmatrix={seed_matrix}_seeddata={seed_data}'
    model_path = os.path.join(model_folder, f'alpha={alpha}', f'{model_name}.pth')

    detector_path = os.path.join(detectors_dir, support_method, f'{model_name}.pkl')
    os.makedirs(os.path.dirname(detector_path), exist_ok=True)
    
    # ------------------ Train detector ------------------
    # stop if already trained
    if os.path.exists(detector_path):
        print(f'The detectors has been already trained')
        sys.exit(0)
    # train the detector
    else:
        # init
        detector = TSOCDetector(cs, model_path, threshold=threshold, filter=True, gpu=device, version=version)
        # fit
        detector = detector.fit(X_train=X)
 
        # save
        with open(detector_path,'wb') as f:
            pickle.dump(detector, f)

# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for TSOC Model with Compressed Sensing")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--window_size', '-w', type=int, default=20, help="Number of consecutive windows")
    parser.add_argument('--fraction', '-F', type=int, default=40, help="Fraction of overlapping windows used for training")
    parser.add_argument('--mode', '-M', type=str, choices=['standard', 'rakeness'], required=True, help="Measurement matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('--seed_training', '-s', type=int, required=True, help="Training-related random seed for reproducibility")
    parser.add_argument('--seed_matrix', '-S', type=int, required=True, help="Seed for random or index for (one of) the best Measurement matrix")
    parser.add_argument('--alpha', '-a', type=float, default=0.5, help="Training loss weight")
    parser.add_argument('--epochs', '-e', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=50, help="Batch size for training")
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer for training")
    parser.add_argument('--fs', '-f', type=int, default=256, help="Sampling frequency")
    parser.add_argument('--orthogonal', '-o', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--source', '-c', type=str, choices=['best', 'random'], default='best', help="Measurement matrix type: genereated randomly or leading to best performance")
    parser.add_argument('--processes', '-p', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help="Threshold for metrics")
    parser.add_argument('--gpu', '-g', type=int, default=3, help="GPU index to use for training")
    parser.add_argument('--train_fraction', '-T', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('--factor', '-r', type=float, default=0.2, help="Factor for ReduceLROnPlateau scheduler")
    parser.add_argument('--min_lr', '-L', type=float, default=0.001, help="Minimum learning rate")
    parser.add_argument('--min_delta', '-d', type=float, default=1e-4, help="Minimum delta for early stopping and ReduceLROnPlateau")
    parser.add_argument('--patience', '-P', type=int, default=40, help="Patience for early stopping")
    parser.add_argument('--version', '-V', type=str, default='v1', help="TSOC version")
    
    return parser.parse_args()

# ------------------ Main call ------------------
if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()
    
    # Call the training function with parsed arguments
    training(
        n=args.n,
        m=args.m,
        window_size=args.window_size,
        fraction=args.fraction,
        epochs=args.epochs,
        lr=args.lr,
        opt=args.optimizer,
        batch_size=args.batch_size,
        fs=args.fs,
        mode=args.mode,  
        orthogonal=args.orthogonal,
        source=args.source,
        seed_matrix=args.seed_matrix,
        alpha = args.alpha,
        seed_training=args.seed_training,
        processes=args.processes,
        threshold=args.threshold,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        factor=args.factor,  
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience,
        version=args.version
    )
