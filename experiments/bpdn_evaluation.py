import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd

import multiprocessing as mp
import itertools as it

import logging
import argparse

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir
from cs.wavelet_basis import wavelet_basis
from cs.utils import compute_rsnr
from cs.decoders import BasisPursuitDenoise
from cs import CompressedSensing, generate_sensing_matrix

logging.basicConfig(level=logging.DEBUG)


def test( n, m, N_test, fs, heart_rate, isnr, basis,
          mode, orthogonal, source, seed_matrix, sigma, processes):
    
    # ------------------ Constants ------------------
    corr_name = '96af96a7ddfcb2f6059092c250e18f2a'
    seed_test_data = 66
    if m==48:
        M = 1_000 # number of evaluated sensing matrices
    else:
        M = 100
    loc = 0.25
    seed_data_matrix = 0
    seed_selection = 0

    # ------------------ Folders ------------------
    results_folder = '/srv/newpenny/dnn-cs/tsoc/results/TSOC/reconstruction'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running test with parameters: {params_str}")

     # ------------------ Signal ------------------
    data_name = f'ecg_N={N_test}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_test_data}'
    data_path = os.path.join(dataset_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)

    # ------------------ Compressed Sensing ------------------
    D = wavelet_basis(n, basis, level=2)

    # # Sensing matrix
    # if mode == 'rakeness':
    #     corr_path = os.path.join(dataset_dir, 'correlation', f'{corr_name}.pkl')
    #     with open(corr_path, 'rb') as f:
    #         C = pickle.load(f)
    # else:
    #     C = None
    # A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orthogonal, correlation=C, loc=loc, seed=seed_matrix)
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

    results_name = f'bpdn-n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                 f'_isnr={isnr}_mode={mode}_src={source}_ort={orthogonal}_seedmat={seed_matrix}_sigma={sigma}'
    if mode == 'rakeness':
        results_name = f'{results_name}_corr={corr_name}_loc={loc}'
    results_path = os.path.join(results_folder, f'{results_name}.pkl')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if os.path.exists(results_path):
        print(f'\n{results_name}\nhas already been evaluated')
        sys.exit(0)

    # args = [(A @ D, Y[i], sigma) for i in range(len(Y))]
    # with mp.Pool(processes=processes) as pool:
    #     output = pool.starmap(spg_bpdn, args)

    # Xhat = np.empty(X.shape)
    # for i in len(output):
    #     Xhat[i] = D @ output[0]

    B = A @ D
    bpdn = BasisPursuitDenoise(B, D, sigma=sigma, processes=processes)
    Xhat = bpdn.decode(Y)
    results = pd.Series(index=['RSNR'], dtype=np.float64)
    results['RSNR'] = np.mean(compute_rsnr(X, Xhat))
    pd.to_pickle(results, results_path)

    # ------------------ Parser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate various detectors for anomaly detection tasks.")

    # Core arguments
    parser.add_argument('-n', '--n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('-m', '--m', type=int, required=True, help="Number of measurements")
    parser.add_argument('-md', '--mode', type=str, choices=['standard', 'rakeness'], required=True, help="Sensing matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('-idx', '--seed_matrix', type=int, required=True, help="Seed of random sensing matrix")
    parser.add_argument('-i', '--isnr', type=int, required=True, help="Signal-to-noise ratio (SNR) in dB")
    parser.add_argument('-M', '--N_test', type=int, default=10000, help="Number of test samples")
    parser.add_argument('-B', '--basis', type=str, default='sym6', help="Wavelet basis function")
    parser.add_argument('-f', '--fs', type=int, default=256, help="Sampling frequency")
    parser.add_argument('-r', '--heart_rate', type=int, nargs=2, default=(60, 100), help="Heart rate range")
    parser.add_argument('-o', '--orthogonal', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--source', '-src', type=str, choices=['best', 'random'], default='best', help="Sensing matrix type: genereated randomly or leading to best performance")
    parser.add_argument('-p', '--processes', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('-s', '--sigma', type=float, default=1e-4, help="BPDN threshold")

    return parser.parse_args()

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()
    # Call the test function with parsed arguments
    test(
        n=args.n,
        m=args.m,
        basis=args.basis,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
        mode=args.mode,
        orthogonal=args.orthogonal,
        source=args.source,
        seed_matrix=args.seed_matrix,
        processes=args.processes,
        N_test=args.N_test,
        sigma=args.sigma
    )




    