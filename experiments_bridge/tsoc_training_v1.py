import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from scipy.fft import dct
import pickle
import argparse
import logging

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from models import models_dir
from dataset import dataset_dir
from cs.training_metrics import compute_metrics, update_metrics
from cs.loss import multiclass_loss_alpha
from models.tsoc import TSOC
from cs import CompressedSensing, generate_sensing_matrix

logging.basicConfig(level=logging.DEBUG)

def training(
    n, m, window_size, fraction, epochs, lr, opt, batch_size, fs, mode, orthogonal, source,
    #   seed_matrix,
    alpha, seed_training, seed_matrix, seed_data, processes, threshold, gpu, train_fraction, factor, min_lr, min_delta, patience
):
    
    method = 'TSOC'
    M = 100 # number of evaluated sensing matrices

    # ------------------ Folders ------------------
    model_folder = f'{models_dir}TSOC'

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
    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_fraction={fraction}_seed={seed_data}'
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)

    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_ = pickle.load(f)

    # ------------------ Compressed Sensing ------------------
    D = dct(np.eye(n), axis=0, norm='ortho')

    # Sensing matrix
    C = (X_.T @ X_) / len(X_)
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

    cs = CompressedSensing(A, D)
    Y = cs.encode(X)  # measurements

    # ------------------ Labels (support) ------------------
    supports_name = f'bridge_train_supports_method={method}_mode={mode}_m={m}_orth={orthogonal}_ws={window_size}_fraction={fraction}_seed_matrix={seed_matrix}_seed_data={seed_data}'
    data_path = os.path.join(files_dir, supports_name+'.pkl')
    with open(data_path, 'rb') as f:
        Z = pickle.load(f)

    # ------------------ Data loaders ------------------
    dataset = TensorDataset(torch.from_numpy(Y).float(), torch.from_numpy(Z).float())  # Create a dataset from the tensors
    # Split sizes for training and validation
    train_size = int(train_fraction * len(dataset))  # 90% for training
    val_size = len(dataset) - train_size  # 10% for validation

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=processes, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=processes, worker_init_fn=seed_worker, generator=generator)

    # ------------------ Neural Network initialization ------------------
    tsoc = TSOC(n, m)
    tsoc.to(device) # move the network to GPU
    model_name = f'bridge-TSOC-N={len(X)}_n={n}_ws={window_size}_fraction={fraction}_m={m}_fs={fs}_mode={mode}'\
                 f'_src={source}_ort={orthogonal}_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                 f'_th={threshold}_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                 f'_mind={min_delta}_seedtraining={seed_training}_seedmatrix={seed_matrix}_seeddata={seed_data}'
    model_path = os.path.join(model_folder, f'alpha={alpha}', f'{model_name}.pth')

    # ------------------ Trining loop ------------------
    # ensure the directories exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        print(f'Model\n{model_name}\nhas already been trained')
        sys.exit(0)
    else:
        if opt=='sgd':
            optimizer = optim.SGD(tsoc.parameters(), lr=lr)
        elif opt=='adam':
            optimizer = optim.Adam(tsoc.parameters(), lr=lr)
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
                loss = multiclass_loss_alpha(output, Z_batch, alpha)

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
                    val_loss += multiclass_loss_alpha(output, Z_batch, alpha).item()
                    val_metrics_batch = compute_metrics(output, Z_batch, th=threshold)
                    val_metrics = update_metrics(val_metrics, val_metrics_batch)

            num_batches = len(val_loader)
            val_loss = val_loss/num_batches
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
            val_metrics = {key: value / num_batches for key, value in val_metrics.items()}

            print(f"Epoch [{epoch+1}/{epochs}], LR={scheduler.get_last_lr()[0]}\nTRAIN Loss: {np.round(train_loss, 5)}  " +\
                "  ".join([f'{key}: {np.round(value, 5)}' for key, value in train_metrics.items()])  +\
                    f"\n  VAL Loss: {np.round(val_loss, 5)}  " +\
                        "  ".join([f'{key}: {np.round(value, 5)}' for key, value in val_metrics.items()]) + "\n") 

        # save trained model
        torch.save(tsoc.state_dict(), model_path)

# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for TSOC Model with Compressed Sensing")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--window_size', '-w', type=int, default=20, help="Number of consecutive windows")
    parser.add_argument('--fraction', '-F', type=int, default=40, help="Fraction of overlapping windows used for training")
    parser.add_argument('--epochs', '-e', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer for training")
    parser.add_argument('--batch_size', '-b', type=int, default=50, help="Batch size for training")
    parser.add_argument('--fs', '-f', type=int, default=256, help="Sampling frequency")
    parser.add_argument('--mode', '-M', type=str, choices=['standard', 'rakeness'], required=True, help="Measurement matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('--orthogonal', '-o', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--source', '-c', type=str, choices=['best', 'random'], default='best', help="Measurement matrix type: genereated randomly or leading to best performance")
    parser.add_argument('--alpha', '-a', type=float, default=0.5, help="Training loss weight")
    parser.add_argument('--seed_training', '-s', type=int, required=True, help="Training-related random seed for reproducibility")
    parser.add_argument('--seed_matrix', '-S', type=int, default=0, help="Seed for random or index for (one of) the best Measurement matrix")
    parser.add_argument('--seed_data', '-D', type=int, default=0, help="Seed for training data")
    parser.add_argument('--processes', '-p', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help="Threshold for metrics")
    parser.add_argument('--gpu', '-g', type=int, default=3, help="GPU index to use for training")
    parser.add_argument('--train_fraction', '-T', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('--factor', '-r', type=float, default=0.2, help="Factor for ReduceLROnPlateau scheduler")
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
        alpha = args.alpha,
        seed_training=args.seed_training,
        seed_matrix =args.seed_matrix,
        seed_data =args.seed_data,
        processes=args.processes,
        threshold=args.threshold,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        factor=args.factor,  
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience
    )
