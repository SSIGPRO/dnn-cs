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
    n, m, epochs, lr, opt, batch_size, N_train, basis, fs, heart_rate, isnr, mode, 
    orthogonal, source, seed_matrix, alpha, seed_training, processes, threshold, gpu, train_fraction, min_lr, 
    min_delta, patience, N_test
):

    # ------------------ Constants ------------------
    corr_name = '96af96a7ddfcb2f6059092c250e18f2a'
    support_method = 'TSOC'
    seed_train_data = 11
    seed_test_data = 66
    M = 1_000
    loc = 0.25

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
    D = wavelet_basis(n, basis, level=2)

    # Sensing matrix
    if mode == 'rakeness':
        corr_path = os.path.join(dataset_dir, 'correlation', f'{corr_name}.pkl')
        with open(corr_path, 'rb') as f:
            C = pickle.load(f)
    else:
        C = None
    A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orthogonal, correlation=C, loc=loc, seed=seed_matrix)

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
    test_dataset = TensorDataset(torch.from_numpy(Y).float(), torch.from_numpy(Z).float())  # Create a dataset from the tensors
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=processes) 

    # ------------------ Decoder initializaion ------------------
    tsoc = TSOC(n, m)
    tsoc.to(device) # move the network to GPU

    model_name = f'TSOC-N={N_train}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_mode={mode}_src={source}_ort={orthogonal}_seedmat={seed_matrix}'\
                f'_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                f'_th={threshold}_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                f'_mind={min_delta}_seeddata={seed_train_data}_seedtrain={seed_training}'    
    if mode == 'rakeness':
        model_name = f'{model_name}_corr={corr_name}_loc={loc}'
    model_path = os.path.join(model_folder, f'alpha={alpha}', f'{model_name}.pth')
    tsoc.load_state_dict(torch.load(model_path, weights_only=True))
    tsoc.eval()     # Set the model to evaluation mode

    results_path = os.path.join(results_folder, f'alpha={alpha}', f'{model_name}.pkl')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if os.path.exists(results_path):
        print(f'Model\n{model_name}\nhas already been evaluated')
        sys.exit(0)
    
    # ------------------ Compute test metric ------------------
    test_metrics = {'P': 0.0, 'TP': 0.0, 'TPR': 0.0, 'TNR': 0.0, 'ACC': 0.0}
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
    pd.to_pickle(pd.Series(results), results_path)

# ------------------ Parser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate various detectors for anomaly detection tasks.")

    # Core arguments
    parser.add_argument('-n', '--n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('-m', '--m', type=int, required=True, help="Number of measurements")
    parser.add_argument('-s', '--seed_training', type=int, required=True, help="Random seed characterizing the training")
    parser.add_argument('-md', '--mode', type=str, choices=['standard', 'rakeness'], required=True, help="Sensing matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('-idx', '--seed_matrix', type=int, required=True, help="Seed of random sensing matrix")
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help="Training loss weight")
    parser.add_argument('-i', '--isnr', type=int, required=True, help="Signal-to-noise ratio (SNR) in dB")
    parser.add_argument('-N', '--N_train', type=int, default=2_000_000, help="Number of training samples")
    parser.add_argument('-tf', '--train_fraction', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('-M', '--N_test', type=int, default=10000, help="Number of test samples")
    parser.add_argument('-B', '--basis', type=str, default='sym6', help="Wavelet basis function")
    parser.add_argument('-f', '--fs', type=int, default=256, help="Sampling frequency")
    parser.add_argument('-r', '--heart_rate', type=int, nargs=2, default=(60, 100), help="Heart rate range")
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
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer for training")

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
        opt=args.optimizer,
        batch_size=args.batch_size,
        N_train=args.N_train,
        basis=args.basis,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
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
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience,
        N_test=args.N_test
    )



    
