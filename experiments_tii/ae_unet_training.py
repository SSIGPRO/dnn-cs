import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pickle
import argparse
import logging

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from models import models_dir
from dataset import dataset_dir
from cs.training_metrics import rsnr
from models.ae_unet import AEUnet

logging.basicConfig(level=logging.DEBUG)

def training(
    n, m, epochs, lr, opt, coarse, beta, gamma, eta, batch_size, N, fs, heart_rate, isnr,
    seed_training, processes, gpu, train_fraction, factor, 
    min_lr, min_delta, patience
):

    # ------------------ Constants ------------------
    # encoder_mode='orth' 'default'
    encoder_mode='default'

    seed_data = 11 # seed relative to training datas
    kernel_size = 3

    # tiny
    # expanded_channels = 8
    # steps_num = 2

    # # small
    # expanded_channels = 8
    # steps_num = 3

    # large
    expanded_channels = 64
    steps_num = 4

    # ------------------ Folders ------------------
    model_folder = f'{models_dir}AE'

    # if encoder_mode == 'orth':
    #     coarse = 'transpose'
    if coarse == 'transpose':
        encoder_mode = 'orth'
        

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
    data_name = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_data}'
    data_path = os.path.join(dataset_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)

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
    ae = AEUnet(n, m, encoder_mode=encoder_mode, decode_mode=coarse, expanded_channels=expanded_channels, steps_num=steps_num)
    print(ae.decode_mode)
    ae.to(device) # move the network to GPU
    model_name = f'AEUnet-exp_chn={expanded_channels}-stp_nm={steps_num}-krn_sz={kernel_size}-N={N}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                 f'_isnr={isnr}_beta={beta}_gamma={gamma}-eta={eta}-enc={encoder_mode}-coarse={coarse}'\
                 f'_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                 f'_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                 f'_mind={min_delta}_seeddata={seed_data}_seedtrain={seed_training}'    
    model_path = os.path.join(model_folder, f'{model_name}.pth')
    model_path_best = os.path.join(model_folder, f'{model_name}-best.pth')

    tensor_board_path = os.path.join(model_folder, model_name)

    # ------------------ Trining loop ------------------
    # ensure the directories exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(tensor_board_path), exist_ok=True)
    if os.path.exists(model_path):
        print(f'Model\n{model_name}\nhas already been trained')
        sys.exit(0)
    else:
        if opt=='sgd':
            optimizer = optim.SGD(ae.parameters(), lr=lr)
        elif opt=='adam':
            optimizer = optim.Adam(ae.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, threshold=min_delta, threshold_mode='abs', patience=patience//2, min_lr=min_lr)
        # tensorboard writer
        writer = SummaryWriter(log_dir=tensor_board_path)
        writer.add_text("run/params", params_str.replace(",", "\n"), 0)
        min_val_loss = np.inf
        patience_counter = 0
        min_val_loss2 = np.inf

        mse = nn.MSELoss()

        A = ae.fc0.weight
        if eta > 0.0:
            # Initialize an orthonormal matrix
            with torch.no_grad():
                torch.nn.init.orthogonal_(A)
        Im = torch.eye(A.shape[0], device=A.device, dtype=A.dtype) 
        for epoch in range(epochs):
            # train loop
            ae.train()     # Set the model to training mode
        
            train_rsnr = 0.0
            train_mse = 0.0
            train_vol = 0.0
            train_mse_y = 0.0
            train_frob_ort = 0.0
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
                A = ae.fc0.weight
                # frob_ort = torch.linalg.matrix_norm(A @ A.T  - Im, ord='fro')**2 
                frob_ort = torch.mean((A @ A.T  - Im)**2)
                loss = mse_X + beta * latent_volume + gamma * mse_Y + eta * frob_ort

                optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
                loss.backward()     # Backpropagate
                optimizer.step()     # Update weights
                # check loss value
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    print("Invalid loss, terminating training.")
                    writer.close()
                    sys.exit(0)
                train_rsnr += rsnr(X_hat, X_batch).item()
                train_mse += mse_X.item()
                train_vol += latent_volume.item()
                train_mse_y += mse_Y.item()
                train_frob_ort += frob_ort.item()

            num_batches = len(train_loader)
            train_rsnr = train_rsnr/num_batches
            train_mse = train_mse/num_batches
            train_vol = train_vol/num_batches
            train_mse_y = train_mse_y/num_batches
            train_frob_ort = train_frob_ort/num_batches
            train_loss = train_mse + beta * train_vol + gamma * train_mse_y + eta * train_frob_ort

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
            A = ae.fc0.weight
            # val_frob_ort = torch.linalg.matrix_norm(A @ A.T  - Im, ord='fro')
            val_frob_ort = torch.mean((A @ A.T - Im)**2).item()
            val_loss = val_mse + beta * val_vol + gamma * val_mse_y + eta * val_frob_ort
            
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

            if val_loss < min_val_loss2:
                min_val_loss2 = np.copy(val_loss)
                torch.save(ae.state_dict(), model_path_best)
                writer.add_scalar("val/best_loss", float(min_val_loss2), epoch + 1)
                print(f"Best model updated")

            print(f"Epoch [{epoch+1}/{epochs}], LR={scheduler.get_last_lr()[0]}"\
                  f"\nTRAIN Loss: {np.round(train_loss, 8)}   MSE {np.round(train_mse, 8)}   RSNR {np.round(train_rsnr, 3)}   VOL {np.round(train_vol, 8)}   MSE Y {np.round(train_mse_y, 8)} A ort {np.round(train_frob_ort, 8)}"\
                  f"\nVAL Loss: {np.round(val_loss, 8)}   MSE {np.round(val_mse, 8)}   RSNR {np.round(val_rsnr, 3)}   VOL {np.round(val_vol, 8)}   MSE Y {np.round(val_mse_y, 8)} A ort {np.round(val_frob_ort, 8)} \n")
            
            # tensorboard logs
            writer.add_scalar("loss/train", float(train_loss), epoch + 1)
            writer.add_scalar("loss/val", float(val_loss), epoch + 1)
            writer.add_scalar("mse/train", float(train_mse), epoch + 1)
            writer.add_scalar("mse/val", float(val_mse), epoch + 1)
            writer.add_scalar("rsnr/train", float(train_rsnr), epoch + 1)
            writer.add_scalar("rsnr/val", float(val_rsnr), epoch + 1)
            writer.add_scalar("latent_vol/train", float(train_vol), epoch + 1)
            writer.add_scalar("latent_vol/val", float(val_vol), epoch + 1)
            writer.add_scalar("mse_y/train", float(train_mse_y), epoch + 1)
            writer.add_scalar("mse_y/val", float(val_mse_y), epoch + 1)
            writer.add_scalar("frob_ort/train", float(train_frob_ort), epoch + 1)
            writer.add_scalar("frob_ort/val", float(val_frob_ort), epoch + 1)
            writer.add_scalar("lr", float(scheduler.get_last_lr()[0]), epoch + 1)

        # save trained model
        torch.save(ae.state_dict(), model_path)
        writer.close()

# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for AE Model")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--isnr', '-i', type=int, required=True, help="Signal-to-noise ratio (SNR)")
    parser.add_argument('--seed_training', '-s', type=int, required=True, help="Training-related random seed for reproducibility")
    parser.add_argument('--epochs', '-e', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=50, help="Batch size for training")
    parser.add_argument('--N', '-N', type=int, default=2000000, help="Number of training instances")
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer for training")
    parser.add_argument('--coarse', '-c', type=str, default='lstsq', help="Mode to compute the coarse reconstruction")
    parser.add_argument('--beta', '-w', type=float, default=0, help="Latent volume regularization weight")
    parser.add_argument('--gamma', '-q', type=float, default=0, help="Regularization of the latten space reconstruction")
    parser.add_argument('--eta', '-y', type=float, default=0, help="Regularization of the sensing matrix")
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
    training(
        n=args.n,
        m=args.m,
        epochs=args.epochs,
        lr=args.lr,
        opt=args.optimizer,
        coarse=args.coarse,
        beta=args.beta,
        gamma=args.gamma,
        eta=args.eta,
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
