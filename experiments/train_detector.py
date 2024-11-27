import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import pickle
import tqdm
import argparse
import logging

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

from wombats.detectors.feature_based import *
from wombats.detectors.pca_based import *
from wombats.detectors.gaussian_distribution_based import *
from wombats.detectors.ml_based import *

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from cs.wavelet_basis import wavelet_basis
from cs import CompressedSensing, generate_sensing_matrix
from detectors import detectors_dir
from dataset import dataset_dir


logging.basicConfig(level=logging.DEBUG)

def test(
    n, m, N_train, basis, fs, heart_rate, isnr, mode, train_fraction,
    orthogonal, source, index, seed_detector, detector_type, 
    k, order, kernel, nu, neighbors, estimators
):
    # ------------------ Constants ------------------
    corr_name = '96af96a7ddfcb2f6059092c250e18f2a.pkl'
    seed_train_data = 11
    seed_training = 0 # seed for training data split
    seed_data_matrix = 0
    seed_selection = 0
    M = 1_000

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running test with parameters: {params_str}")

    
    # ------------------ Seeds ------------------
    np.random.seed(seed_detector)

    # ------------------ Signal ------------------
    # load training data
    data_name = f'ecg_N={N_train}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_train_data}'
    data_path = os.path.join(dataset_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_train = pickle.load(f)
    

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
    Y_train = cs.encode(X_train) # measurements

    # ------------------ Init and fit the detector ------------------
    if detector_type == 'SPE':
        detector = SPE(k)
        detector_label = f'SPE_k={k}'
    elif detector_type == 'T2':
        detector = T2(k)
        detector_label = f'T2_k={k}'
    elif detector_type == 'AR':
        detector = AR(order)
        detector_label = f'AR_order={order}'
    elif detector_type == 'OCSVM':
        detector = OCSVM(kernel, nu)
        detector_label = f'OCSVM_kernel={kernel}_nu={nu}'
    elif detector_type == 'LOF':
        detector = LOF(neighbors)
        detector_label = f'LOF_neighbors={neighbors}'
    elif detector_type == 'IF':
        detector = IF(estimators)
        detector_label = f'IF_estimators={estimators}'
    elif detector_type == 'MD':
        detector = MD()
        detector_label = detector_type
    elif detector_type == 'energy':
        detector = energy()
        detector_label = detector_type
    elif detector_type == 'TV':
        detector = TV()
        detector_label = detector_type
    elif detector_type == 'ZC':
        detector = ZC()
        detector_label = detector_type
    elif detector_type == 'pk-pk':
        detector = pk_pk()
        detector_label = detector_type
        
     # fit the detector
    model_name = f'{detector_label}_N={N_train}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_mode={mode}_src={source}_ort={orthogonal}_seedmat={index}_tf={train_fraction}_seeddet={seed_detector}'\
                f'_seeddata={seed_train_data}_seedtrain={seed_training}_seedselect={seed_selection}'
    if mode == 'rakeness':
        model_name = f'_{model_name}_corr={corr_name}'
    if source == 'best':
        model_name = f'_{model_name}_seeddatamat={seed_data_matrix}_M={M}'
    model_path = os.path.join(detectors_dir, f'{model_name}.pkl')
    model_path = os.path.join(detectors_dir, f'{model_name}.pkl')
    # stop if already trained
    if os.path.exists(model_path):
        print(f'{detector_label} already trained')
        sys.exit(0)
    # fit
    else:
        train_size = int(train_fraction * len(Y_train))  # 90% for training
        val_size = len(Y_train) - train_size  # 10% for validation

        # Split the dataset into training and validation
        generator = torch.Generator()
        generator.manual_seed(seed_training)
        train_dataset, val_dataset = random_split(Y_train, [train_size, val_size], generator=generator)
        Y_train, _ = train_dataset.dataset, val_dataset.dataset
        detector = detector.fit(Y_train)

        # save
        with open(model_path,'wb') as f:
            pickle.dump(detector, f)

# ------------------ Parser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train various detectors.")

    # Core arguments
    parser.add_argument('-n', '--n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('-m', '--m', type=int, required=True, help="Number of measurements")
    parser.add_argument('-s', '--seed_detector', type=int, required=True, help="Random seed associated to the detector")
    parser.add_argument('-md', '--mode', type=str, choices=['standard', 'rakeness'], required=True, help="Sensing matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('-idx', '--index', type=int, required=True, help="Seed for random or index for (one of) the best sensing matrix")
    parser.add_argument('-i', '--isnr', type=int, required=True, help="Signal-to-noise ratio (SNR) in dB")
    parser.add_argument('-dt', '--detector_type', type=str, required=True, help="Type of detector to evaluate (e.g., TSOC, SPE, OCSVM, LOF)")
    parser.add_argument('-N', '--N_train', type=int, default=2000000, help="Number of training samples")
    parser.add_argument('-tf', '--train_fraction', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('-B', '--basis', type=str, default='sym6', help="Wavelet basis function")
    parser.add_argument('-f', '--fs', type=int, default=256, help="Sampling frequency")
    parser.add_argument('--heart_rate', '-hr', type=int, nargs=2, default=(60, 100), help="Heart rate range")
    parser.add_argument('-o', '--orthogonal', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--source', '-src', type=str, choices=['best', 'random'], default='best', help="Sensing matrix type: genereated randomly or leading to best performance")

    # Standard detector-related arguments (optional for specific detectors)
    parser.add_argument('-k', '--k', type=int, default=5, help="Parameter k for SPE, T2, LOF, or related detectors")
    parser.add_argument('-ord', '--order', type=int, default=1, help="Order parameter for AR detector")
    parser.add_argument('-krn', '--kernel', type=str, default='rbf', help="Kernel type for OCSVM (e.g., linear, rbf, poly)")
    parser.add_argument('-nu', '--nu', type=float, default=0.5, help="Anomaly fraction for OCSVM")
    parser.add_argument('-nn', '--neighbors', type=int, default=10, help="Number of neighbors for LOF detector")
    parser.add_argument('-est', '--estimators', type=int, default=100, help="Number of estimators for IF detector")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()
    # Call the training function with parsed arguments
    test(
        n=args.n,
        m=args.m,
        N_train=args.N_train,
        basis=args.basis,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
        mode=args.mode,
        train_fraction = args.train_fraction,
        orthogonal=args.orthogonal,
        source=args.source,
        index=args.index,
        seed_detector=args.seed_detector,
        detector_type=args.detector_type,
        k=args.k,
        order=args.order,
        kernel=args.kernel,
        nu=args.nu,
        neighbors=args.neighbors,
        estimators=args.estimators
    )



    
