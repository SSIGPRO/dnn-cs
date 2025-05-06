import os
import sys
import numpy as np
from scipy.signal import decimate
from scipy.io import loadmat
import pickle
import argparse
from ponte_utils import sliding_window, select_random_windows
bridge_dir = '/srv/newpenny/dataset/Ponte_S101/Data/'

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))
from dataset import dataset_dir

def generate(n, fs, fraction, window_size, seed):
    np.random.seed(seed)  # Ensure reproducibility

    # Load normal and anomalous data
    Xok = loadmat(os.path.join(bridge_dir, 'downsampled_undamage.mat'))
    Xok= Xok['downsampled_undamage'][0]
    Xko = loadmat(os.path.join(bridge_dir, 'downsampled_dam_B.mat'))
    Xko = Xko['downsampled_dam_B'][0]

    # Decimate if fs is set to half of fs
    if fs == 50:
        Xok = decimate(Xok, q=2, ftype='iir')
        Xko = decimate(Xko, q=2, ftype='iir')
        n = n // 2

    print(len(Xko))
    print(len(Xok))

    # generate batches of window_size consecutive windows
    Xok = sliding_window(Xok, window_size*n, window_size*n)

    ##############################################################
    indices = np.arange(len(Xok))
    indices = np.random.permutation(indices)  # Shuffle indices
    

    split_idx = len(indices) // 2  # 50% split for train/test
    indices_train, indices_test = indices[:split_idx], indices[split_idx:]
    ################################################################

    # # ################################################################
    # indices = np.arange(len(Xok))
    # indices_test = indices[::2]
    # indices_train = np.setdiff1d(indices, indices_test)
    # # ################################################################

    print(indices_test)
    print(indices_train)

    # training data
    Xok_train_ = Xok[indices_train]
    Xok_train = sliding_window(Xok_train_.flatten(), n, n)
    # generate overlapping windows of length n for each selected batch
    Xok_train_augmented = np.concatenate([sliding_window(X, n, 1) for X in Xok_train_])
    N = len(Xok_train_augmented) // 100 * fraction
    Xok_train_augmented, _ = select_random_windows(Xok_train_augmented, N, seed)

    # test data
    Xok_test = Xok[indices_test]
    Xok_test = sliding_window(Xok_test.flatten(), n, n)
    Xko_test = sliding_window(Xko, n, n)

    P = len(Xok_test) // window_size * window_size
    Q = len(Xko_test) // window_size * window_size
    R = min(P, Q)
    Xok_test = Xok_test[:R]
    Xko_test = Xko_test[:R]

    # scaling
    mean = Xok_train.mean()
    std = Xok_train.std()
    # mean = Xok_train_augmented.mean()
    # std = Xok_train_augmented.std()
    
    Xok_train = (Xok_train - mean) / std
    Xok_test = (Xok_test - mean) / std
    Xko_test = (Xko_test - mean) / std
    Xok_train_augmented = (Xok_train_augmented - mean) / std

    print(Xok_train.std(), Xok_train.mean())
    print(Xok_test.std(), Xok_test.mean())
    print(Xko_test.std(), Xko_test.mean())
    print(Xok_train_augmented.std(), Xok_train_augmented.mean())

    # saving the datasets as pkl files
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed}')
    data_path = os.path.join(files_dir, f'bridge_train_n={n}_fs={fs}_ws={window_size}_seed={seed}.pkl')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, 'wb') as f:
        pickle.dump(Xok_train, f)
    data_path = os.path.join(files_dir, f'bridge_train_n={n}_fs={fs}_ws={window_size}_fraction={fraction}_seed={seed}.pkl')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, 'wb') as f:
        pickle.dump(Xok_train_augmented, f)
    data_path = os.path.join(files_dir, f'bridge_ok_n={n}_fs={fs}_ws={window_size}_seed={seed}.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(Xok_test, f)
    data_path = os.path.join(files_dir, f'bridge_ko_n={n}_fs={fs}_ws={window_size}_seed={seed}.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(Xko_test, f)

# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for AE Model")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--seed', '-s', type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument('--window_size', '-w', type=int, default=20, help="Number of consecutive windows")
    parser.add_argument('--fs', '-f', type=int, default=50, help="Sampling frequency")
    parser.add_argument('--fraction', '-F', type=int, default=40, help="Fraction of overlapping windows used for training")
    
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
        seed=args.seed     
    )