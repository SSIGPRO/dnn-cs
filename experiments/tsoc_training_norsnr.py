import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import pickle
import argparse
import logging

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from models import models_dir
from dataset import dataset_dir
from cs.wavelet_basis import wavelet_basis
from cs.training_metrics import compute_metrics, update_metrics
from cs.loss import multiclass_loss_alpha
from models.tsoc import TSOC
from cs import CompressedSensing, generate_sensing_matrix

logging.basicConfig(level=logging.DEBUG)

def training(
    n, m, epochs, lr, opt, batch_size, N, basis, fs, heart_rate, isnr, mode, 
    orthogonal, source, seed_matrix, seed_training, processes, threshold, gpu, train_fraction, factor, 
    min_lr, min_delta, patience
):


    # ------------------ Constants ------------------
    corr_name = '96af96a7ddfcb2f6059092c250e18f2a'
    support_method = 'TSOC'
    seed_data = 11 # seed relative to training datas
    seed_data_matrix = 0 # seed used to generate data for sensing matrix selection
    seed_selection = 0 # seed used to select the best sensing matrix on data geneerated with seed_data_matrix
    M = 1_000 # number of evaluated sensing matrices
    loc = 0.25

    # ------------------ Folders ------------------
    model_folder = f'{models_dir}TSOC'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running test with parameters: {params_str}")

    
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
    data_name = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_data}'
    data_path = os.path.join(dataset_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)

    # ------------------ Compressed Sensing ------------------
    D = wavelet_basis(n, basis, level=2)

    # Sensing matrix
    if source == 'random':
        # Generate a random sensing matrix
        if mode == 'rakeness':
            corr_path = os.path.join(dataset_dir, 'correlation', f'{corr_name}.pkl')
            with open(corr_path, 'rb') as f:
                C = pickle.load(f)
        else:
            C = None
        A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orthogonal, correlation=C, loc=loc, seed=seed_matrix)
    elif source == 'best':
        # Load the best sensing matrix

        A_folder = f'ecg_N=10000_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}_isnr={isnr}_seed={seed_data_matrix}'
        A_name = f'sensing_matrix_M={M}_m={m}_mode={mode}_seed={seed_selection}'
        if mode == 'rakeness':
            A_name = f'{A_name}_loc={loc}_corr={corr_name}'
        data_path = os.path.join(dataset_dir, A_folder, 'A_Filippo', f'{A_name}.pkl')
        with open(data_path, 'rb') as f:
            A_dict = pickle.load(f)
        A = A_dict[seed_matrix]['matrix']
        seed_matrix = A_dict[seed_matrix]['seed']
        logging.info(f'sensing matrix ({m}, {n}) with seed={seed_matrix} loaded')

    cs = CompressedSensing(A, D)
    Y = cs.encode(X)  # measurements

    # ------------------ Labels (support) ------------------
    if mode == 'rakeness':
        supports_name = f'supports_method={support_method}_mode={mode}_m={m}'\
            f'_corr={corr_name}_loc={loc}_orth={orthogonal}_seed={seed_matrix}.pkl'
    else:
        supports_name = f'supports_method={support_method}_mode={mode}_m={m}'\
                f'_orth={orthogonal}_seed={seed_matrix}.pkl'

    data_path = os.path.join(dataset_dir, data_name, supports_name)
    with open(data_path, 'rb') as f:
        Z = pickle.load(f)

    # ------------------ Data loaders ------------------
    dataset = TensorDataset(torch.from_numpy(Y).float(), torch.from_numpy(Z).float())  # Create a dataset from the tensors
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
    tsoc = TSOC(n, m)
    tsoc.to(device) # move the network to GPU
    model_name = f'TSOC-N={N}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_mode={mode}_src={source}_ort={orthogonal}_seedmat={seed_matrix}'\
                f'_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                f'_th={threshold}_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                f'_mind={min_delta}_seeddata={seed_data}_seedtrain={seed_training}'    
    if mode == 'rakeness':
        model_name = f'{model_name}_corr={corr_name}_loc={loc}'
    model_path = os.path.join(model_folder, f'{model_name}.pth')

    # ------------------ Trining loop ------------------
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
        # ensure the directories exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(tsoc.state_dict(), model_path)

# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for TSOC Model with Compressed Sensing")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--isnr', '-i', type=int, required=True, help="Signal-to-noise ratio (SNR)")
    parser.add_argument('--mode', '-M', type=str, choices=['standard', 'rakeness'], required=True, help="Measurement matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('--seed_training', '-s', type=int, required=True, help="Training-related random seed for reproducibility")
    parser.add_argument('--seed_matrix', '-S', type=int, required=True, help="Seed for random or index for (one of) the best Measurement matrix")
    parser.add_argument('--epochs', '-e', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=50, help="Batch size for training")
    parser.add_argument('--N', '-N', type=int, default=2000000, help="Number of training instances")
    parser.add_argument('--basis', '-B', type=str, default='sym6', help="Wavelet basis function")
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer for training")
    parser.add_argument('--fs', '-f', type=int, default=256, help="Sampling frequency")
    parser.add_argument('--heart_rate', '-r', type=int, nargs=2, default=(60, 100), help="Heart rate range")
    parser.add_argument('--orthogonal', '-o', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--source', '-c', type=str, choices=['best', 'random'], default='best', help="Measurement matrix type: genereated randomly or leading to best performance")
    parser.add_argument('--processes', '-p', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help="Threshold for metrics")
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
        batch_size=args.batch_size,
        N=args.N,
        basis=args.basis,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
        mode=args.mode,  
        orthogonal=args.orthogonal,
        source=args.source,
        seed_matrix=args.seed_matrix,
        seed_training=args.seed_training,
        processes=args.processes,
        threshold=args.threshold,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        factor=args.factor,  
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience
    )
