import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import pandas as pd
from scipy import linalg
import pickle
import tqdm
import argparse

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset.synthetic_ecg import generate_ecg
from cs.wavelet_basis import wavelet_basis
from cs.supports import find_support_TSOC
from cs.training_metrics import compute_metrics, update_metrics
from cs.loss import multiclass_loss_alpha
from models.tsoc import TSOC
from cs import CompressedSensing, generate_sensing_matrix


def training(
        n = 128,
        m = 32,
        epochs = 500,
        lr = 0.1,
        batch_size = 50,
        N = 2_000_000,
        basis = 'sym6',       
        fs = 256,             
        heart_rate = (60, 100), 
        isnr = 35,
        mode = 'standard',
        orthogonal = False,           
        seed = 0,   
        processes = 48,
        threshold = 0.5,
        gpu = 3,
        train_fraction = 0.9,
        factor = 0.2,
        min_lr = 0.001,
        min_delta = 1e-4,
        patience = 40,
):
    # ------------------ Seeds ------------------
    np.random.seed(seed)

    # Set the seed for PyTorch (CPU)
    torch.manual_seed(seed)

    # Set the seed for PyTorch (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # ------------------ GPU ------------------
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # ------------------ Signal ------------------
    namefile = os.path.join('../../newpenny/dnn-cs/JETCAS2020/data/',f'n{n}_ISNR', f'trSet_n={n}_isnr={isnr}_no-sparse.h5')
    with pd.HDFStore(namefile, mode='r') as store:
        X = store.select('X').values.squeeze()

    # ------------------ Compressed Sensing ------------------
    D = wavelet_basis(n, basis, level=2)
    A = generate_sensing_matrix((m, n), mode='standard', orthogonal=orthogonal, loc=.25, seed=seed)
    cs = CompressedSensing(A, D)
    Y = cs.encode(X)  # measurements

    # ------------------ Labels (support) ------------------
    namefile = os.path.join('../../newpenny/dnn-cs/JETCAS2020/data/',f'n{n}_ISNR', f'trSet_n={n}_m={m}_isnr={isnr}-label.h5')
    with pd.HDFStore(namefile, mode='r') as store:
        Z = store.select('S').values.squeeze()

    # ------------------ Loaders ------------------
    dataset = TensorDataset(torch.from_numpy(Y).float(), torch.from_numpy(Z).float())  # Create a dataset from the tensors
    # Split sizes for training and validation
    train_size = int(train_fraction * len(dataset))  # 90% for training
    val_size = len(dataset) - train_size  # 10% for validation

    # Split the dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=processes)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=processes)

    # ------------------ Neural Network initialization ------------------
    tsoc = TSOC(n, m)
    tsoc.to(device) # move the network to GPU
    file_model = f'TSOC-N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_mode={mode}_ort={orthogonal}_epochs={epochs}_bs={batch_size}_opt=sgd_lr={lr}'\
                f'_th={threshold}_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                f'_mind={min_delta}_seed={seed}.pth'
    
    # ------------------ Trining loop ------------------
    if os.path.exists(file_model):
        print(f'Model\n{file_model}\nhas already been trained')
        sys.exit(0)
    else:
        optimizer = optim.SGD(tsoc.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, threshold=min_delta, threshold_mode='abs', patience=patience//2, min_lr=min_lr)
        min_val_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):
            # train loop
            tsoc.train()     # Set the model to training mode
            train_loss = 0.0
            train_metrics = {'P': 0.0, 'TP': 0.0, 'TPR': 0.0, 'TNR': 0.0, 'ACC': 0.0}
            for batch_idx, (Y_batch, Z_batch) in enumerate(train_loader):
                Y_batch, Z_batch = Y_batch.to(device), Z_batch.to(device)     # move training data to GPU
                output = tsoc(Y_batch)
                loss = multiclass_loss_alpha(output, Z_batch)

                train_metrics_batch = compute_metrics(output, Z_batch, th=threshold)
                optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
                loss.backward()     # Backpropagate
                optimizer.step()     # Update weights
                # check loss value
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    print("Invalid loss, terminating training.")
                    break
                train_loss += loss.item()
                train_metrics = update_metrics(train_metrics, train_metrics_batch)

            num_batches = len(train_loader)
            train_loss = train_loss/num_batches
            train_metrics = {key: value / num_batches for key, value in train_metrics.items()}

            # validation loop
            tsoc.eval()     # Set the model to evaluation mode
            val_loss = 0.0
            val_metrics = {'P': 0.0, 'TP': 0.0, 'TPR': 0.0, 'TNR': 0.0, 'ACC': 0.0}
            with torch.no_grad():     # disables gradient calculation for the validation phase 
                for batch_idx, (Y_batch, Z_batch) in enumerate(val_loader):
                    Y_batch, Z_batch = Y_batch.to(device), Z_batch.to(device)     # move validation data to GPU
                    output = tsoc(Y_batch)
                    val_loss += multiclass_loss_alpha(output, Z_batch).item()
                    val_metrics_batch = compute_metrics(output, Z_batch, th=threshold)
                    val_metrics = update_metrics(val_metrics, val_metrics_batch)

            num_batches = len(val_loader)
            val_loss = val_loss/num_batches
            # callbacks
            # early stopping
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
            val_metrics = {key: value / num_batches for key, value in val_metrics.items()}

            print(f"Epoch [{epoch+1}/{epochs}], LR={scheduler.get_last_lr()[0]}\nTRAIN Loss: {np.round(train_loss, 3)}  " +\
                "  ".join([f'{key}: {np.round(value, 3)}' for key, value in train_metrics.items()])  +\
                    f"\n  VAL Loss: {np.round(val_loss, 3)}  " +\
                        "  ".join([f'{key}: {np.round(value, 3)}' for key, value in val_metrics.items()]) + "\n") 

        # save trained model
        torch.save(tsoc.state_dict(), file_model)

# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for TSOC Model with Compressed Sensing")

    # Short and long argument versions
    parser.add_argument('-n', '--n', type=int, default=128, help="Number of samples per signal")
    parser.add_argument('-m', '--m', type=int, default=32, help="Number of measurements")
    parser.add_argument('--epochs', '-e', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=50, help="Batch size for training")
    parser.add_argument('--N', type=int, default=2_000_000, help="Number of training instances")
    parser.add_argument('--basis', '-B', type=str, default='sym6', help="Wavelet basis function")
    parser.add_argument('--fs', '-f', type=int, default=256, help="Sampling frequency")
    parser.add_argument('--heart_rate', '-hr', type=int, nargs=2, default=(60, 100), help="Heart rate range")
    parser.add_argument('--isnr', '-i', type=int, default=35, help="Signal-to-noise ratio (SNR)")
    parser.add_argument('--mode', '-md', type=str, choices=['standard', 'rakeness'], default='standard', help="Measurement matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('--orthogonal', '-o', type=bool, default=False, help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--seed', '-s', type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--processes', '-p', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help="Threshold for metrics")
    parser.add_argument('--gpu', '-g', type=int, default=3, help="GPU index to use for training")
    parser.add_argument('--train_fraction', '-tf', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('--factor', '-fct', type=float, default=0.2, help="Factor for ReduceLROnPlateau scheduler")
    parser.add_argument('--min_lr', '-minlr', type=float, default=0.001, help="Minimum learning rate")
    parser.add_argument('--min_delta', '-mind', type=float, default=1e-4, help="Minimum delta for early stopping and ReduceLROnPlateau")
    parser.add_argument('--patience', '-pt', type=int, default=40, help="Patience for early stopping")
    
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
        batch_size=args.batch_size,
        N=args.N,
        basis=args.basis,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
        mode=args.mode,  
        orthogonal=args.orthogonal,
        seed=args.seed,
        processes=args.processes,
        threshold=args.threshold,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        factor=args.factor,
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience
    )
