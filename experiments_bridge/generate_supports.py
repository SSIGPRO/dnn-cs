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

from tqdm import tqdm
from itertools import product

import multiprocessing as mp

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir
from cs import CompressedSensing, generate_sensing_matrix
from cs.supports import find_support_TSOC


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

method = 'TSOC'
M = 100 # number of evaluated sensing matrices

def generate(n, fs, fraction, window_size, seed_matrix, seed_data, mode, source, orth, m, processes):
    # Signal
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')
    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_train = pickle.load(f)  

    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_fraction={fraction}_seed={seed_data}'
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_train_augmented = pickle.load(f)  

    data_name = f'bridge_ok_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_test = pickle.load(f)

    # Generate Sensing Matrix and set Compressed Sensing system
    
    # Generate a random sensing matrix
    if mode == 'rakeness':
        C = (X_train.T @ X_train) / len(X_train)
        loc = np.trace(C @ C) / np.trace(C)**2 - 1 / n
    else:
        C = None
        loc = None

    supports_name_ = f'supports_method={method}_mode={mode}_m={m}_orth={orth}'

    if source == 'best':
        fname =f'matrices_ort={orth}_mode={mode}_M={M}_m={m}_mode={mode}_seed={seed_data}'
        path = os.path.join(files_dir, f'Best_sensing_matrices', fname+'.pkl')
        with open(path, 'rb') as f:
            A_dict = pickle.load(f)
        A = A_dict[seed_matrix]['matrix']
        seed_matrix = A_dict[seed_matrix]['seed']
    logger.debug(f'generating sensing matrix ({m}, {n}), seed={seed_matrix}')
    A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orth, correlation=C, loc=loc, seed=seed_matrix)

    D = dct(np.eye(n), axis=0, norm='ortho')
    cs = CompressedSensing(A, D=D, decoder='SO')

    # compute suports for training data
    supports_path = os.path.join(files_dir,
                                  f'bridge_train_{supports_name_}_ws={window_size}_seed_matrix={seed_matrix}_seed_data={seed_data}.pkl')
    if os.path.exists(supports_path):
        logger.info(f'supports {supports_path} already exists')
    else:
        logger.debug(f'computing supports for training data')
        if processes is None:
            S = [find_support_TSOC(x, cs) for x in tqdm(X_train, desc=f'{m}, {seed_matrix}')]
        else:
            with mp.Pool(processes=processes) as pool:
                S = pool.starmap(find_support_TSOC, product(X_train, [cs]), chunksize=50)
        S = np.stack(S)

        logger.debug(f'storing supports')
        with open(supports_path, 'wb') as f:
            pickle.dump(S, f)

    # compute suports for training data
    supports_path = os.path.join(files_dir,
                                  f'bridge_train_{supports_name_}_ws={window_size}_fraction={fraction}_seed_matrix={seed_matrix}_seed_data={seed_data}.pkl')
    if os.path.exists(supports_path):
        logger.info(f'supports {supports_path} already exists')
    else:
        logger.debug(f'computing supports for training data')
        if processes is None:
            S = [find_support_TSOC(x, cs) for x in tqdm(X_train_augmented, desc=f'{m}, {seed_matrix}')]
        else:
            with mp.Pool(processes=processes) as pool:
                S = pool.starmap(find_support_TSOC, product(X_train_augmented, [cs]), chunksize=50)
        S = np.stack(S)

        logger.debug(f'storing supports')
        with open(supports_path, 'wb') as f:
            pickle.dump(S, f)

    # compute suports for test data
    supports_path = os.path.join(files_dir,
                                  f'bridge_ok_{supports_name_}_seed_matrix={seed_matrix}_seed_data={seed_data}.pkl')
    if os.path.exists(supports_path):
        logger.info(f'supports {supports_path} already exists')
    else:
        logger.debug(f'computing supports for test data')
        if processes is None:
            S = [find_support_TSOC(x, cs) for x in tqdm(X_test, desc=f'{m}, {seed_matrix}')]
        else:
            with mp.Pool(processes=processes) as pool:
                S = pool.starmap(find_support_TSOC, product(X_test, [cs]), chunksize=50)
        S = np.stack(S)

        logger.debug(f'storing supports')
        with open(supports_path, 'wb') as f:
            pickle.dump(S, f)

# ------------------ Parser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for AE Model")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--seed_matrix', '-s', type=int, default=0, help="Seed for random or index for (one of) the best Measurement matrix")
    parser.add_argument('--seed_data', '-D', type=int, default=0, help="Seed for training data")
    parser.add_argument('--window_size', '-w', type=int, default=20, help="Number of consecutive windows")
    parser.add_argument('--fs', '-f', type=int, default=50, help="Sampling frequency")
    parser.add_argument('--fraction', '-F', type=int, default=40, help="Fraction of overlapping windows used for training")
    parser.add_argument('--mode', '-e', choices=('standard', 'rakeness'), default='standard', 
                        help="compressed sensing encoding mode (default: %(default)s)")
    parser.add_argument('--source','-S', type=str, choices=('best', 'random'), default='random',
        help="Measurement matrix type: genereated randomly or leading to best performance (default: %(default)s)")
    parser.add_argument('--orthogonal', '-o', action='store_true', help="weather sensing matrix is orthogonal")
    parser.add_argument('--measurements','-m', type=int, default=64, help="number of measurements (default: %(default)s)")
    parser.add_argument('--processes', '-p', type=int, help="number of parallell processes")
    
    return parser.parse_args()

# ------------------ Main call ------------------
if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()
    
    # Call the training function with parsed arguments
    generate(
        n=args.n,
        fs=args.fs,
        fraction=args.fraction,
        window_size=args.window_size,
        seed_matrix=args.seed_matrix,
        seed_data=args.seed_data,
        mode=args.mode,
        source=args.source,
        orth=args.orthogonal,
        m=args.measurements,
        processes=args.processes     
    )