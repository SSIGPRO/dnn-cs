import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torcheval.metrics.aggregation.auc import AUC
from sklearn.metrics import roc_auc_score

import numpy as np
import pickle
import argparse
import logging

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from models import models_dir
from dataset import dataset_dir
from cs.training_metrics import rsnr
from models.ae_discriminator import AED

logging.basicConfig(level=logging.DEBUG)

def training(
    n, m, levels, epochs, lr, opt, beta, batch_size, N, fs, heart_rate, isnr, delta,
    seed_training, processes, gpu, train_fraction, factor, 
    min_lr, min_delta, patience
):

    # ------------------ Constants ------------------
    seed_data = 11 # seed relative to training datas
    seed_ko = 0 # seed relative to anomalous training datas
    num_anomalies = 12

    # ------------------ Folders ------------------
    model_folder = f'{models_dir}AE'

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
    ok_name = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_data}'
    ok_path = os.path.join(dataset_dir, ok_name+'.pkl')
    with open(ok_path, 'rb') as f:
        Xok = pickle.load(f)


    ko_name = f'ecg_anomaly_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                    f'_isnr={isnr}_delta={delta}_seedok={seed_data}_seedko={seed_ko}.pkl'

    ko_path = os.path.join(dataset_dir, ok_name, 'anomalies', f'delta={delta}', f'seed={seed_ko}', ko_name)
    with open(ko_path, 'rb') as f:
        Xko = pickle.load(f)

    N = int((N // num_anomalies) * num_anomalies)
    Xok, Xko = Xok[:N], Xok[:N]

    # ------------------ Data loaders ------------------
    dataset = TensorDataset(torch.from_numpy(Xok).float(), torch.from_numpy(Xko).float())  # Create a dataset from the tensors
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
    ae = AED(n, m, levels)
    ae.to(device) # move the network to GPU
    model_name = f'AE-discriminator-N={N}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                 f'_isnr={isnr}_beta={beta}'\
                 f'_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                 f'_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                 f'_mind={min_delta}_seeddata={seed_data}_seedtrain={seed_training}'    
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
        ce = nn.BCELoss()

        for epoch in range(epochs):
            # train loop
            ae.train()     # Set the model to training mode
        
            train_rsnr = 0.0
            train_mse = 0.0
            train_ceok = 0.0
            train_ceko = 0.0
            train_auc = 0.0
            for Xok_batch, Xko_batch in train_loader:
                Xok_batch, Xko_batch = Xok_batch.to(device), Xko_batch.to(device) # move training data to GPU
                # generate signals
                Yok_batch, Yko_batch = ae.encode(Xok_batch), ae.encode(Xko_batch)
                Yok_batch, Yko_batch = ae.quantize(Yok_batch), ae.quantize(Yko_batch)
                Zok_hat, Zko_hat = ae.discriminate(Yok_batch), ae.discriminate(Yko_batch)
                Xok_hat = ae.decode(Yok_batch)
                Zok, Zko = torch.zeros(Zok_hat.shape, dtype=torch.float).to(device), torch.ones(Zko_hat.shape, dtype=torch.float).to(device)
                # compute metrics
                mseok = mse(Xok_hat, Xok_batch)
                ceok, ceko = ce(Zok_hat, Zok), ce(Zko_hat, Zko)
                auc = roc_auc_score(
                    np.concatenate([Zok.detach().cpu(), Zko.detach().cpu()]),
                      np.concatenate([Zok_hat.detach().cpu(), Zko_hat.detach().cpu()])
                      )
                loss = mseok + beta * (ceok + ceko)

                optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
                loss.backward()     # Backpropagate
                optimizer.step()     # Update weights
                # check loss value
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    print("Invalid loss, terminating training.")
                    break
                train_rsnr += rsnr(Xok_hat, Xok_batch).item()
                train_mse += mseok.item()
                train_ceok += ceok.item()
                train_ceko += ceko.item()
                train_auc += auc

            num_batches = len(train_loader)
            train_rsnr = train_rsnr/num_batches
            train_mse = train_mse/num_batches
            train_ceok = train_ceok/num_batches
            train_ceko = train_ceko/num_batches
            train_auc = train_auc/num_batches
            train_loss = train_mse + beta * (train_ceok  + train_ceko)

            # validation loop
            ae.eval()     # Set the model to evaluation mode
            val_rsnr = 0.0
            val_mse = 0.0
            val_ceok = 0.0
            val_ceko = 0.0
            val_auc = 0.0
            with torch.no_grad():     # disables gradient calculation for the validation phase 
                for Xok_batch, Xko_batch in val_loader:
                    Xok_batch, Xko_batch = Xok_batch.to(device), Xko_batch.to(device) # move training data to GPU
                    # generate signals
                    Yok_batch, Yko_batch = ae.encode(Xok_batch), ae.encode(Xko_batch)
                    Yok_batch, Yko_batch = ae.quantize(Yok_batch), ae.quantize(Yko_batch)
                    Zok_hat, Zko_hat = ae.discriminate(Yok_batch), ae.discriminate(Yko_batch)
                    Xok_hat = ae.decode(Yok_batch)

                    Zok, Zko = torch.zeros(Zok_hat.shape, dtype=torch.float).to(device), torch.ones(Zko_hat.shape, dtype=torch.float).to(device)
                    # compute metrics
                    val_rsnr += rsnr(Xok_hat, Xok_batch).item()
                    val_mse += mse(Xok_hat, Xok_batch).item()
                    val_ceok += ce(Zok_hat, Zok).item()
                    val_ceko += ce(Zko_hat, Zko).item()
                    val_auc += roc_auc_score(
                        np.concatenate([Zok.detach().cpu(), Zko.detach().cpu()]),
                          np.concatenate([Zok_hat.detach().cpu(), Zko_hat.detach().cpu()])
                          )

            num_batches = len(val_loader)
            val_rsnr = val_rsnr/num_batches
            val_mse = val_mse/num_batches
            val_ceok = val_ceok/num_batches
            val_ceko = val_ceko/num_batches
            val_auc = val_auc/num_batches
            val_loss = val_mse + beta * (val_ceok  + val_ceko)
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
                  f"\nTRAIN Loss: {np.round(train_loss, 8)}   MSE {np.round(train_mse, 8)}   RSNR {np.round(train_rsnr, 3)}   CE OK {np.round(train_ceok, 8)}   CE KO {np.round(train_ceko, 8)}   AUC {np.round(train_auc, 3)}"\
                  f"\nVAL Loss: {np.round(val_loss, 8)}   MSE {np.round(val_mse, 8)}   RSNR {np.round(val_rsnr, 3)}   CE OK {np.round(val_ceok, 8)}   CE KO {np.round(val_ceko, 8)}   AUC {np.round(val_auc, 3)} \n")

        # save trained model
        torch.save(ae.state_dict(), model_path)

# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for AE Model")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, default=128, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, default=16, help="Number of measurements")
    parser.add_argument('--isnr', '-i', type=int, default=35, help="Signal-to-noise ratio (SNR)")
    parser.add_argument('--levels', '-v', type=int, default=128, help="Quantization levels for each dimension")
    parser.add_argument('--seed_training', '-s', type=int, default=0, help="Training-related random seed for reproducibility")
    parser.add_argument('--epochs', '-e', type=int, default=1000, help="Number of training epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=128, help="Batch size for training")
    parser.add_argument('--N', '-N', type=int, default=2000000, help="Number of training instances")
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer for training")
    parser.add_argument('--beta', '-w', type=float, required=True, help="discrimitator weight")
    parser.add_argument('--fs', '-f', type=int, default=256, help="Sampling frequency")
    parser.add_argument('--heart_rate', '-r', type=int, nargs=2, default=(60, 100), help="Heart rate range")
    parser.add_argument('--processes', '-p', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('--gpu', '-g', type=int, default=3, help="GPU index to use for training")
    parser.add_argument('--train_fraction', '-T', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('--factor', '-F', type=float, default=0.2, help="Factor for ReduceLROnPlateau scheduler")
    parser.add_argument('--min_lr', '-L', type=float, default=0.001, help="Minimum learning rate")
    parser.add_argument('--min_delta', '-d', type=float, default=1e-4, help="Minimum delta for early stopping and ReduceLROnPlateau")
    parser.add_argument('--patience', '-P', type=int, default=40, help="Patience for early stopping")
    parser.add_argument('--delta', '-D', type=float, default=0.05, help="Intensity of the anomalies (default: %(default)s)")
    
    return parser.parse_args()

# ------------------ Main call ------------------
if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()
    
    # Call the training function with parsed arguments
    training(
        n=args.n,
        m=args.m,
        levels=args.levels,
        epochs=args.epochs,
        lr=args.lr,
        opt=args.optimizer,
        beta=args.beta,
        batch_size=args.batch_size,
        N=args.N,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
        delta=args.delta,
        seed_training=args.seed_training,
        processes=args.processes,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        factor=args.factor,  
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience
    )
