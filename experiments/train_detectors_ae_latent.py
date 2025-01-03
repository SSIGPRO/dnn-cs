import os

os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import sys
import torch
from torch.utils.data import random_split
import pickle
import argparse
import logging

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from detectors import detectors_dir
from dataset import dataset_dir
from models import models_dir
from detectors.ae import AEDetector


logging.basicConfig(level=logging.DEBUG)

def training(
    n, m, epochs, lr, opt, beta, gamma, detector_mode, batch_size, N, fs, heart_rate, isnr,
    seed_training, processes, gpu, train_fraction, factor, 
    min_lr, min_delta, patience
):
    # ------------------ Constants ------------------
    seed_data = 11 # seed relative to training datas

    # ------------------ Folders ------------------
    model_folder = f'{models_dir}AE'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running training with parameters: {params_str}")

    # ------------------ Reproducibility ------------------

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
    data_name = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_data}'
    data_path = os.path.join(dataset_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_train = pickle.load(f)

    # Split the dataset into training and validation
    train_size = int(train_fraction * len(X_train))  # 90% for training
    val_size = len(X_train) - train_size  # 10% for validation
    train_dataset, val_dataset = random_split(X_train, [train_size, val_size], generator=generator)
    X_train, _ = train_dataset.dataset, val_dataset.dataset

    # ------------------ Paths ------------------
    model_name = f'AE-N={N}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                 f'_isnr={isnr}_beta={beta}_gamma={gamma}'\
                 f'_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                 f'_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                 f'_mind={min_delta}_seeddata={seed_data}_seedtrain={seed_training}'    
    model_path = os.path.join(model_folder, f'{model_name}.pth')
    detector_path = os.path.join(detectors_dir, 'AE', detector_mode, f'{model_name}.pkl')
    os.makedirs(os.path.dirname(detector_path), exist_ok=True)
    # stop if already trained
    if os.path.exists(detector_path):
        print(f'The detectors has been already trained')
        sys.exit(0)
    # train the detector
    else:
        # init
        detector = AEDetector(n, m, model_path, mode=detector_mode, gpu=device)
        # fit
        detector = detector.fit(X_train=X_train)
 
        # save
        with open(detector_path,'wb') as f:
            pickle.dump(detector, f)

# ------------------ Parser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train various detectors.")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--isnr', '-i', type=int, required=True, help="Signal-to-noise ratio (SNR)")
    parser.add_argument('--seed_training', '-s', type=int, required=True, help="Training-related random seed for reproducibility")
    parser.add_argument('--epochs', '-e', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=50, help="Batch size for training")
    parser.add_argument('-dmd', '--detector_mode', type=str, choices=['norm', 'MD'], help="Mode of operation of the detector")
    parser.add_argument('--N', '-N', type=int, default=2000000, help="Number of training instances")
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer for training")
    parser.add_argument('--beta', '-w', type=float, default=0, help="Latent volume regularization weight")
    parser.add_argument('--gamma', '-q', type=float, default=0, help="Regularization of the latten space reconstruction")
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


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()
    # Call the training function with parsed arguments
    training(
        n=args.n,
        m=args.m,
        epochs=args.epochs,
        lr=args.lr,
        opt=args.optimizer,
        beta=args.beta,
        gamma=args.gamma,
        detector_mode=args.detector_mode,
        batch_size=args.batch_size,
        N=args.N,
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




    
