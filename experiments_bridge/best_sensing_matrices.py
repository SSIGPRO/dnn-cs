import os
import sys
import pickle
import logging

import argparse

# limit number of parallel threads numpy spawns
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import numpy as np
from scipy.fft import dct
from spgl1 import spg_bpdn

from tqdm import tqdm
from itertools import product

import multiprocessing as mp

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir
from cs import CompressedSensing, generate_sensing_matrix
from cs.decoders import BasisPursuitDenoise
from cs.utils import compute_rsnr


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

def generate(n, m, fs, window_size, mode, orthogonal, N_try,
              N_save, seed, seed_data, processes):

    # ------------------ Reproducibility ------------------
    rng = np.random.default_rng(seed)
    seeds_A = rng.choice(100000, size=N_try, replace=False)

    # ------------------ Signal ------------------
    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    subfolder = f'bridge_n={n}_fs={fs}_seed={seed_data}'
    files_dir = os.path.join(dataset_dir, subfolder)
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_train = pickle.load(f)

    # ------------------ Path ------------------
    fname =f'matrices_ort={orthogonal}_mode={mode}_M={N_try}_m={m}_mode={mode}_seed={seed}'
    path = os.path.join(dataset_dir, subfolder, f'Best_sensing_matrices', fname+'.pkl')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # ------------------ Compressed Sensing ------------------
    D = dct(np.eye(n), axis=0, norm='ortho')
    C = (X_train.T @ X_train) / len(X_train)
    loc = np.trace(C @ C) / np.trace(C)**2 - 1 / n

    best_rsnr = -np.inf * np.ones(N_save)
    seed_A_best = np.empty((N_save), dtype=int)
    keep_A = np.empty((N_save, m, n), dtype=np.float64)
    
    for i, seed_A in enumerate(seeds_A):
        A = generate_sensing_matrix(
            (m, n),
              mode=mode,
                orthogonal=orthogonal,
                  correlation=C,
                    loc=loc,
                      seed=seed_A
                      )
        B = A @ D
        cs = CompressedSensing(A, D)
        Y = cs.encode(X_train)  # measurements
        bpdn = BasisPursuitDenoise(B, D, sigma=1e-4, processes=processes)
        xhat = bpdn.decode(Y)
        rsnr = np.mean(compute_rsnr(X_train, xhat))

        idx_min = np.argmin(best_rsnr)
        if rsnr > best_rsnr[idx_min]:
            best_rsnr[idx_min] = rsnr
            seed_A_best[idx_min] = seed_A
            keep_A[idx_min] = A

        print(f'[{i}/{N_try-1}] rsnr={np.round(rsnr, 8)} --- best: {[np.round(i, 2) for i in best_rsnr]}')

    srt = np.argsort(best_rsnr)
    best_rsnr, seed_A_best, keep_A = best_rsnr[srt][::-1], seed_A_best[srt][::-1], keep_A[srt][::-1]
    
    list_save = []
    for A, rsnr, seed in zip(keep_A, best_rsnr, seed_A_best):
        list_save += [{'rsnr': rsnr, 'seed': seed, 'matrix': A}]

    with open(path, 'wb') as f:
        pickle.dump(list_save, f)

    print(f'best {N_save} (over {N_try}) have been saved for m = {m} in {path}')

    return

# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for TSOC Model with Compressed Sensing")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--window_size', '-w', type=int, default=20, help="Number of consecutive windows")
    parser.add_argument('--fs', '-f', type=int, default=50, help="Sampling frequency")
    parser.add_argument('--mode', '-M', type=str, choices=['standard', 'rakeness'], required=True, help="Measurement matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('--orthogonal', '-o', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--seed_data', '-s', type=int, required=True, help="Data-related random seed for reproducibility")
    parser.add_argument('--processes', '-p', type=int, default=48, help="Number of CPU processes")

    parser.add_argument('--seed', '-S', type=int, default=0, help="Seed")
    parser.add_argument('--M', '-t', type=int, default=100, help="Matrices to try")
    parser.add_argument('--K', '-k', type=int, default=8, help="Matrices to save")
    
    return parser.parse_args()

# ------------------ Main call ------------------
if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()
    
    # Call the training function with parsed arguments
    generate(
        n=args.n,
          m=args.m,
            fs=args.fs,
              window_size=args.window_size,
                mode=args.mode,
                  orthogonal=args.orthogonal,
                    N_try=args.M,
                      N_save=args.K,
                        seed=args.seed,
                          seed_data=args.seed_data,
                            processes=args.processes
                          )