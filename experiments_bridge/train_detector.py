import os

os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
from scipy.fft import dct
import pandas as pd
import pickle
import tqdm
import argparse
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
from detectors.kde import KDE
from detectors import detectors_dir
from dataset import dataset_dir


logging.basicConfig(level=logging.DEBUG)

def test(
    n, m, fs, window_size, fraction, mode, train_fraction,
    orthogonal, source, seed_matrix, seed_detector, detector_type, 
    k, order, kernel, kernel2, nu, neighbors, estimators, bandwidth
):
    # ------------------ Constants ------------------
    M = 100 # number of evaluated sensing matrices
    seed_data = 0
    seed_training = 0
    if bandwidth == 0:
        bandwidth = 'silverman'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running test with parameters: {params_str}")

    
    # ------------------ Seeds ------------------
    np.random.seed(seed_detector)

    # ------------------ Signal ------------------
    # load training data
    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_train = pickle.load(f)
    

    # ------------------ Compressed Sensing ------------------
    D = dct(np.eye(n), axis=0, norm='ortho')

    # Sensing matrix
    C = (X_train.T @ X_train) / len(X_train)
    loc = np.trace(C @ C) / np.trace(C)**2 - 1 / n
    if source == 'random':
        A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orthogonal, correlation=C, loc=loc, seed=seed_matrix)
    elif source == 'best':
        fname =f'matrices_ort={orthogonal}_mode={mode}_M={M}_m={m}_mode={mode}_seed={seed_data}'
        path = os.path.join(files_dir, f'Best_sensing_matrices', fname+'.pkl')
        with open(path, 'rb') as f:
            A_dict = pickle.load(f)
        A = A_dict[seed_matrix]['matrix']
        seed_matrix = A_dict[seed_matrix]['seed']

    cs = CompressedSensing(A, D)
    Y_train = cs.encode(X_train)  # measurements

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
    elif detector_type == 'OC':
        detector = OCSVM(kernel, nu)
        detector_label = f'OC_krn={kernel}_nu={nu}'
    elif detector_type == 'LOF':
        detector = LOF(neighbors)
        detector_label = f'LOF_neighbors={neighbors}'
    elif detector_type == 'IF':
        detector = IF(estimators)
        detector_label = f'IF_estimators={estimators}'
    elif detector_type == 'KDE':
        detector = KDE(kernel2, bandwidth)
        detector_label = f'KDE_krn={kernel2}_bw={bandwidth}'
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

    # define the name sufix
    name = f'N={len(X_train)}_n={n}_ws={window_size}_fraction={fraction}_m={m}_fs={fs}'\
                f'_mode={mode}_src={source}_ort={orthogonal}_seedmat={seed_matrix}_tf={train_fraction}_seeddet={seed_detector}'\
                f'_seeddata={seed_data}_seedtrain={seed_training}'

    # scale the data for non-machine learning-based detectors
    if detector_type not in ['LOF', 'OC', 'IF', 'KDE']:
        scaler_name = f'bridge_scaler_{name}'
        scaler_path = os.path.join(detectors_dir, f'{scaler_name}.pkl')
        # load scaler
        if os.path.exists(scaler_path):
            print(f'scaler already trained')
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            # train scaler
            scaler = StandardScaler(with_std=False).fit(Y_train)
            # save scaler
            with open(scaler_path,'wb') as f:
                pickle.dump(scaler, f)
        
        # center training data
        Y_train = scaler.transform(Y_train)
        
    # fit the detector
    model_name = f'bridge_{detector_label}_{name}'
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
    parser.add_argument('--window_size', '-w', type=int, default=20, help="Number of consecutive windows")
    parser.add_argument('--fraction', '-F', type=int, default=40, help="Fraction of overlapping windows used for training")
    parser.add_argument('-s', '--seed_detector', type=int, required=True, help="Random seed associated to the detector")
    parser.add_argument('-md', '--mode', type=str, choices=['standard', 'rakeness'], required=True, help="Sensing matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('-S', '--seed_matrix', type=int, required=True, help="Seed for random or index for (one of) the best sensing matrix")
    parser.add_argument('-dt', '--detector_type', type=str, required=True, help="Type of detector to evaluate (e.g., TSOC, SPE, OC, LOF)")
    parser.add_argument('-tf', '--train_fraction', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('-f', '--fs', type=int, default=256, help="Sampling frequency")
    parser.add_argument('-o', '--orthogonal', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--source', '-src', type=str, choices=['best', 'random'], default='best', help="Sensing matrix type: genereated randomly or leading to best performance")

    # Standard detector-related arguments (optional for specific detectors)
    parser.add_argument('-k', '--k', type=int, default=5, help="Parameter k for SPE, T2, LOF, or related detectors")
    parser.add_argument('-ord', '--order', type=int, default=1, help="Order parameter for AR detector")
    parser.add_argument('-krn', '--kernel', type=str, default='rbf', help="Kernel type for OC (e.g., linear, rbf, poly)")
    parser.add_argument('-krn2', '--kernel2', type=str, default='gaussian', help="Kernel type for KDE")
    parser.add_argument('-nu', '--nu', type=float, default=0.5, help="Anomaly fraction for OC")
    parser.add_argument('-bw', '--bandwidth', type=float, default=0.1, help="Bandwidth for KDE")
    parser.add_argument('-nn', '--neighbors', type=int, default=20, help="Number of neighbors for LOF detector")
    parser.add_argument('-est', '--estimators', type=int, default=100, help="Number of estimators for IF detector")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()
    # Call the training function with parsed arguments
    test(
        n=args.n,
        m=args.m,
        fs=args.fs,
        window_size=args.window_size,
        fraction=args.fraction,
        mode=args.mode,
        train_fraction = args.train_fraction,
        orthogonal=args.orthogonal,
        source=args.source,
        seed_matrix=args.seed_matrix,
        seed_detector=args.seed_detector,
        detector_type=args.detector_type,
        k=args.k,
        order=args.order,
        kernel=args.kernel,
        kernel2=args.kernel2,
        nu=args.nu,
        neighbors=args.neighbors,
        estimators=args.estimators,
        bandwidth=args.bandwidth
    )



    
