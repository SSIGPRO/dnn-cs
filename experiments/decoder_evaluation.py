import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import tqdm
import argparse
import logging

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from cs.wavelet_basis import wavelet_basis
from cs.utils import compute_rsnr, reconstructor
from cs.training_metrics import compute_metrics, update_metrics
from models import models_dir
from dataset import dataset_dir
from models.tsoc import TSOC
from cs import CompressedSensing, generate_sensing_matrix


logging.basicConfig(level=logging.DEBUG)

def test(
    n, m, epochs, lr, batch_size, N_train, basis, fs, heart_rate, isnr, mode, 
    orthogonal, source, index, seed_training, processes, threshold, gpu, train_fraction, min_lr, 
    min_delta, patience, N_test
):

    # ------------------ Constants ------------------
    corr_name = '96af96a7ddfcb2f6059092c250e18f2a.pkl'
    support_method = 'TSOC'
    seed_train_data = 11
    seed_test_data = 66
    seed_support = 0
    seed_data_matrix = 0
    seed_selection = 0
    M = 1_000

    # ------------------ Folders ------------------
    results_folder = '/srv/newpenny/dnn-cs/tsoc/results/TSOC/reconstruction'
    model_folder = f'{models_dir}TSOC'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running test with parameters: {params_str}")

    # ------------------ GPU ------------------
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')


     # ------------------ Signal ------------------
    data_name = f'ecg_N={N_test}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_test_data}'
    data_path = os.path.join(dataset_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)

    # ------------------ Compressed Sensing ------------------
    # ------------------ Compressed Sensing ------------------
    D = wavelet_basis(n, basis, level=2)

    # Sensing matrix
    if source == 'random':
        # Generate a random sensing matrix
        if mode == 'rakeness':
            corr_path = os.path.join(dataset_dir, 'correlation', corr_name)
            with open(corr_path, 'rb') as f:
                C = pickle.load(f)
        else:
            C = None
        A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orthogonal, correlation=C, loc=.25, seed=index)
    elif source == 'best':
        # Load the best sensing matrix

        A_folder = f'ecg_N=10000_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}_isnr={isnr}_seed={seed_data_matrix}'
        A_name = f'sensing_matrix_M={M}_m={m}_mode={mode}_seed={seed_selection}'
        if mode == 'rakeness':
            A_name = f'{A_name}_loc={.25}_corr={corr_name}'
        data_path = os.path.join(dataset_dir, A_folder, 'A_Filippo', f'{A_name}.pkl')
        with open(data_path, 'rb') as f:
            A_dict = pickle.load(f)
        A = A_dict[index]['matrix']

    cs = CompressedSensing(A, D)
    Y = cs.encode(X)  # measurements

    # ------------------ Labels (support) ------------------
    if mode == 'rakeness':
        supports_name = f'supports_method={support_method}_mode={mode}_m={m}'\
            f'_corr={corr_name}_loc={.25}_orth={orthogonal}_seed={seed_support}.pkl'
    else:
        supports_name = f'supports_method={support_method}_mode={mode}_m={m}'\
                f'_orth={orthogonal}_seed={seed_support}.pkl'
        
    data_path = os.path.join(dataset_dir, data_name, supports_name)
    with open(data_path, 'rb') as f:
        Z = pickle.load(f)

    # ------------------ Data loaders ------------------
    test_dataset = TensorDataset(torch.from_numpy(Y).float(), torch.from_numpy(Z).float())  # Create a dataset from the tensors
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=processes) 

    # ------------------ Decoder initializaion ------------------
    tsoc = TSOC(n, m)
    tsoc.to(device) # move the network to GPU
    print(tsoc)
    model_name = f'TSOC-N={N}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_mode={mode}_src={source}_ort={orthogonal}_seedmat={index}_epochs={epochs}_bs={batch_size}_opt=sgd_lr={lr}'\
                f'_th={threshold}_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                f'_mind={min_delta}_seeddata={seed_train_data}_seedtrain={seed_training}'\
                f'_seedselect={seed_selection}_seedsup={seed_support}'        
    if mode == 'rakeness':
        model_name = f'{model_name}_corr={corr_name}'
    if source == 'best':
        model_name = f'{model_name}_seeddatamat={seed_data_matrix}_M={M}'
    model_path = os.path.join(model_folder, f'{model_name}.pth')
    tsoc.load_state_dict(torch.load(model_path, weights_only=True))
    tsoc.eval()     # Set the model to evaluation mode
    
    # ------------------ Compute test metric ------------------
    with torch.no_grad():     # disables gradient calculation for the validation phase 
        for batch_idx, (Y_batch, Z_batch) in enumerate(test_loader):
            Y_batch, Z_batch = Y_batch.to(device), Z_batch.to(device)     # move validation data to GPU
            output = tsoc(Y_batch)
            test_metrics_batch = compute_metrics(output, Z_batch, th=threshold)
            results = update_metrics(test_metrics, test_metrics_batch)

    num_batches = len(test_loader)
    results = {key: value / num_batches for key, value in results.items()}
    print(f"  ".join([f'{key}: {np.round(value, 3)}' for key, value in results.items()])) 

    # ------------------ Recovery performance ------------------
    O = torch.empty(X.shape)
    for batch_idx, (Y_batch, _) in enumerate(test_loader):
        Y_batch = Y_batch.to(device) 
        O_batch = tsoc(Y_batch)
        O[batch_size*batch_idx: batch_size*(batch_idx + 1)] = O_batch
    
    O = O.cpu().detach().numpy() # Moves the tensor from the GPU to the CPU,
    # removes it from the computation graph and converts it to numpy array

    # calculate supports
    Zhat = O > threshold

    # signal reconstruction
    Xhat = np.empty(X.shape)
    for i in tqdm.tqdm(range(N_test)):
        # Xhat[i] = reconstructor(O[i], Y[i], A, D)
        Xhat[i] = cs.decode_with_support(Y[i], Zhat[i])
    RSNR = compute_rsnr(X, Xhat)
    results['RSNR'] = np.mean(RSNR)

    # save peformance
    results_path = os.path.join(results_folder, f'reconstruction_metrics_{model_name}.pkl')
    pd.to_pickle(pd.Series(results), results_path)

# ------------------ Parser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate various detectors for anomaly detection tasks.")

    # Core arguments
    parser.add_argument('-n', '--n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('-m', '--m', type=int, required=True, help="Number of measurements")
    parser.add_argument('-s', '--seed_training', type=int, required=True, help="Random seed characterizing the training")
    parser.add_argument('-md', '--mode', type=str, choices=['standard', 'rakeness'], required=True, help="Sensing matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('-idx', '--index', type=int, required=True, help="Seed for random or index for (one of) the best sensing matrix")
    parser.add_argument('-i', '--isnr', type=int, required=True, help="Signal-to-noise ratio (SNR) in dB")
    parser.add_argument('-N', '--N_train', type=int, default=2_000_000, help="Number of training samples")
    parser.add_argument('-tf', '--train_fraction', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('-M', '--N_test', type=int, default=10000, help="Number of test samples")
    parser.add_argument('-B', '--basis', type=str, default='sym6', help="Wavelet basis function")
    parser.add_argument('-f', '--fs', type=int, default=256, help="Sampling frequency")
    parser.add_argument('-hr', '--heart_rate', type=str, default='60,100', help="Heart rate range (comma-separated, e.g., 60,100)")
    parser.add_argument('-o', '--orthogonal', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--source', '-src', type=str, choices=['best', 'random'], default='best', help="Sensing matrix type: genereated randomly or leading to best performance")
    parser.add_argument('-p', '--processes', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('-g', '--gpu', type=int, default=3, help="GPU index to use for evaluation")
    parser.add_argument('-minlr', '--min_lr', type=float, required=True, help="Minimum learning rate for optimizers")
    parser.add_argument('-mind', '--min_delta', type=float, required=True, help="Minimum change in monitored metric for early stopping")
    parser.add_argument('-pt', '--patience', type=int, required=True, help="Patience for early stopping in epochs")
    parser.add_argument('-b', '--batch_size', type=int, required=True, help="Batch size for training")
    parser.add_argument('-t', '--threshold', type=float, required=True, help="Threshold for TSOC detector")
    parser.add_argument('-e', '--epochs', type=int, required=True, help="Number of training epochs")
    parser.add_argument('-l', '--lr', type=float, required=True, help="Learning rate")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()
    # Call the test function with parsed arguments
    test(
        n=args.n,
        m=args.m,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        N_train=args.N_train,
        basis=args.basis,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
        mode=args.mode,
        orthogonal=args.orthogonal,
        source=args.source,
        index=args.index,
        seed=args.seed,
        processes=args.processes,
        threshold=args.threshold,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience,
        N_test=args.N_test
    )



    
