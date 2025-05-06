import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from scipy.signal import decimate
import pickle
import argparse
import logging

from scipy.io import loadmat

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from models import models_dir
dataset_dir = '/srv/newpenny/dataset/Ponte_S101/Data/'
from cs.training_metrics import rsnr
from models.ae import AE

logging.basicConfig(level=logging.DEBUG)

def sliding_window(X, n, stride):
    """
    Generate overlapping windows using a sliding approach with variable stride.
    Ensures the output shape is (N, n).
    """
    return np.array([X[i:i + n] for i in range(0, X.shape[0] - n + 1, stride)])

def select_random_windows(X_windows, M, seed_training):
    """
    Select M random windows from the dataset, using a fixed random seed.
    """
    np.random.seed(seed_training)  # Set random seed for reproducibility
    N = X_windows.shape[0]
    indices = np.random.choice(N, M, replace=False)  # Random unique indices
    return X_windows[indices], indices

def training(
    n, m, epochs, lr, opt, beta, gamma, batch_size, fs,
    seed_training, processes, gpu, train_fraction, factor, 
    min_lr, min_delta, patience
):

    # ------------------ Folders ------------------
    model_folder = f'{models_dir}AEPonte'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running training with parameters: {params_str}")

    
    # ------------------ Reproducibility ------------------
    # Set the seed for Nympy
    np.random.seed(seed_training)  # Ensure reproducibility
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
    X = loadmat(os.path.join(dataset_dir, 'downsampled_undamage.mat'))
    X = X['downsampled_undamage'][0]
    if fs == 50:
        X = decimate(X, q=2, ftype='iir')
        n = n // 2
        m = m // 2
    # N = (X.shape[0] // n) * n
    # X = X[:N].reshape(-1, n)

    # N_train = X.shape[0] // 100 * 80
    # X = X[:N_train]
    # X = sliding_window(X, n, 1)
    # N = len(X) // 100 * 20
    # X = select_random_windows(X, N, seed_training)

    # perform data augmentation
    # X = sliding_window(X, n, 1)
    # N = X.shape[0] // 100 * 20
    # X, _ = select_random_windows(X, N, seed_training)
    
    X = sliding_window(X, 20*n, 20*n)
    # indices = np.arange(0, len(X))
    # indices_test = indices[::2]
    # indices_train = np.setdiff1d(indices, indices_test)
    

    indices = np.arange(len(X))
    indices = np.random.permutation(indices)  # Shuffle indices

    split_idx = len(indices) // 2  # 50% split for train/test (adjust as needed)
    indices_train, indices_test = indices[:split_idx], indices[split_idx:]

    X = X[indices_train]
    X = np.concatenate([sliding_window(X_, n, 1) for X_ in X])
    N = len(X) // 100 * 40
    X, _ = select_random_windows(X, N, seed_training)
    
    mean = X.mean()
    std = X.std()
    X = (X - mean) / std
    print(X.shape)

    # ------------------ Data loaders ------------------
    dataset = TensorDataset(torch.from_numpy(X).float())  # Create a dataset from the tensors
    # Split sizes for training and validation
    train_size = int(train_fraction * len(dataset))  # 90% for training
    val_size = len(dataset) - train_size  # 10% for validation

    # Split the dataset
    generator = torch.Generator()
    generator.manual_seed(seed_training)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=processes, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=processes, worker_init_fn=seed_worker, generator=generator)

    # ------------------ Neural Network initialization ------------------
    ae = AE(n, m)
    ae.to(device) # move the network to GPU
    model_name = f'AE-N={N}_n={n}_m={m}_fs={fs}-beta={beta}_gamma={gamma}'\
                 f'_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                 f'_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                 f'_mind={min_delta}_seedtrain={seed_training}'    
    model_path = os.path.join(model_folder, f'{model_name}.pth')

    # ------------------ Trining loop ------------------
    # ensure the directories exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        print(f'Model\n{model_name}\nhas already been trained')
        sys.exit(0)
    else:
        if opt=='sgd':
            optimizer = optim.SGD(ae.parameters(), lr=lr)
        elif opt=='adam':
            optimizer = optim.Adam(ae.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, threshold=min_delta, threshold_mode='abs', patience=patience//2, min_lr=min_lr)
        min_val_loss = np.inf
        patience_counter = 0

        mse = nn.MSELoss() 

        for epoch in range(epochs):
            # train loop
            ae.train()     # Set the model to training mode
        
            train_rsnr = 0.0
            train_mse = 0.0
            train_vol = 0.0
            train_mse_y = 0.0
            for X_batch in train_loader:
                X_batch = X_batch[0].to(device)     # move training data to GPU
                # generate signals
                Y_batch = ae.encode(X_batch)
                X_hat = ae.decode(Y_batch)
                Y_hat = ae.encode(X_hat)
                # compute metrics
                mse_X = mse(X_hat, X_batch)
                latent_volume = torch.mean(torch.square(Y_batch))
                mse_Y = mse(Y_hat, Y_batch)
                loss = mse_X + beta * latent_volume + gamma * mse_Y

                optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
                loss.backward()     # Backpropagate
                optimizer.step()     # Update weights
                # check loss value
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    print("Invalid loss, terminating training.")
                    break
                train_rsnr += rsnr(X_hat, X_batch).item()
                train_mse += mse_X.item()
                train_vol += latent_volume.item()
                train_mse_y += mse_Y.item()

            num_batches = len(train_loader)
            train_rsnr = train_rsnr/num_batches
            train_mse = train_mse/num_batches
            train_vol = train_vol/num_batches
            train_mse_y = train_mse_y/num_batches
            train_loss = train_mse + beta * train_vol + gamma * train_mse_y

            # validation loop
            ae.eval()     # Set the model to evaluation mode
            val_rsnr = 0.0
            val_mse = 0.0
            val_vol = 0.0
            val_mse_y = 0.0
            with torch.no_grad():     # disables gradient calculation for the validation phase 
                for X_batch in val_loader:
                    X_batch = X_batch[0].to(device)     # move validation data to GPU
                    # output = ae(X_batch)
                    # generate signals
                    Y_batch = ae.encode(X_batch)
                    X_hat = ae.decode(Y_batch)
                    Y_hat = ae.encode(X_hat)
                    # compute metrics
                    val_mse += mse(X_hat, X_batch).item()
                    val_rsnr += rsnr(X_hat, X_batch).item()
                    val_vol += torch.mean(torch.square(Y_batch)).item()
                    val_mse_y += mse(Y_hat, Y_batch).item()

            num_batches = len(val_loader)
            val_mse = val_mse/num_batches
            val_rsnr = val_rsnr/num_batches
            val_vol = val_vol/num_batches
            val_mse_y = val_mse_y/num_batches
            val_loss = val_mse + beta * val_vol + gamma * val_mse_y
            # callbacks
            # reduce lr on plateau
            scheduler.step(val_loss)
            # early stopping
            if val_loss < min_val_loss - min_delta:
                min_val_loss = np.copy(val_loss)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

            print(f"Epoch [{epoch+1}/{epochs}], LR={scheduler.get_last_lr()[0]}"\
                  f"\nTRAIN Loss: {np.round(train_loss, 8)}   MSE {np.round(train_mse, 8)}   RSNR {np.round(train_rsnr, 3)}   VOL {np.round(train_vol, 8)}   MSE Y {np.round(train_mse_y, 8)}"\
                  f"\nVAL Loss: {np.round(val_loss, 8)}   MSE {np.round(val_mse, 8)}   RSNR {np.round(val_rsnr, 3)}   VOL {np.round(val_vol, 8)}   MSE Y {np.round(val_mse_y, 8)} \n")

        # save trained model
        torch.save(ae.state_dict(), model_path)

# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for AE Model")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--seed_training', '-s', type=int, required=True, help="Training-related random seed for reproducibility")
    parser.add_argument('--epochs', '-e', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=50, help="Batch size for training")
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer for training")
    parser.add_argument('--beta', '-w', type=float, default=0, help="Latent volume regularization weight")
    parser.add_argument('--gamma', '-q', type=float, default=0, help="Regularization of the latten space reconstruction")
    parser.add_argument('--fs', '-f', type=int, default=100, help="Sampling frequency")
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
    training(
        n=args.n,
        m=args.m,
        epochs=args.epochs,
        lr=args.lr,
        opt=args.optimizer,
        beta=args.beta,
        gamma=args.gamma,
        batch_size=args.batch_size,
        fs=args.fs,
        seed_training=args.seed_training,
        processes=args.processes,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        factor=args.factor,  
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience
    )
