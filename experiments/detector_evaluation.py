import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from scipy import linalg
import pickle
import tqdm
import argparse

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

from wombats.detectors.feature_based import *
from wombats.detectors.pca_based import *
from wombats.detectors.gaussian_distribution_based import *
from wombats.detectors.ml_based import *
from wombats.detectors._base import AUC

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset.synthetic_ecg import generate_ecg
from cs.wavelet_basis import wavelet_basis
from models.tsoc import TSOC
from cs import CompressedSensing, generate_sensing_matrix
from detectors.tsoc import TSOCDetector
from detectors import detectors_folder

def test(
    n, m, epochs, lr, batch_size, N_train, basis, fs, heart_rate, isnr, mode, 
    orthogonal, seed, processes, threshold, gpu, train_fraction, factor, min_lr, 
    min_delta, patience, detector_type, delta, N_test, detector_mode, 
    k, order, kernel, nu, neighbors, estimators
):
    # ------------------ Show parameter values ------------------
    print(
        f"Running with the following parameters:\n"
        f"  n={n}, m={m}, epochs={epochs}, lr={lr}, batch_size={batch_size}\n"
        f"  N_train={N_train}, N_test={N_test}, train_fraction={train_fraction}\n"
        f"  basis={basis}, fs={fs}, heart_rate={heart_rate}\n"
        f"  isnr={isnr}, mode={mode}, orthogonal={orthogonal}\n"
        f"  seed={seed}, processes={processes}, gpu={gpu}\n"
        f"  threshold={threshold}, factor={factor}, min_lr={min_lr}, min_delta={min_delta}, patience={patience}\n"
        f"  detector_type={detector_type}, detector_mode={detector_mode}, delta={delta}"
    )

    # ------------------ Folders ------------------
    data_folder = '/srv/newpenny/dnn-cs/JETCAS2020/data/'
    results_folder = '/srv/newpenny/dnn-cs/tsoc/results/TSOC/detection'

    # ------------------ Seeds ------------------
    np.random.seed(seed)

    # Set the seed for PyTorch (CPU)
    torch.manual_seed(seed)

    # Set the seed for PyTorch (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # ------------------ GPU ------------------
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')


    # ------------------ Signal ------------------
    # load test data
    data_path = os.path.join(data_folder, f'n{n}_ISNR', f'tsSet_n={n}_isnr={isnr}_no-sparse.h5')
    with pd.HDFStore(data_path, mode='r') as store:
        X = store.select('X').values.squeeze()
    X = X[:N_test]
    
    # standarize the data
    std, mean = X.std(), X.mean()
    Xstd = (X - mean) / std

    # ------------------ Compressed Sensing ------------------
    D = wavelet_basis(n, basis, level=2)
    A = generate_sensing_matrix((m, n), mode='standard', orthogonal=orthogonal, loc=.25, seed=seed)
    cs = CompressedSensing(A, D)
    Y = cs.encode(X)  # measurements

    # ------------------ Anomalies generation ------------------
    # initialize anomalies
    delta_nonstd = std**2 * delta

    anomalies_labels = [
        'GWN', 
        'Impulse', 
        'Step', 
        'Constant',  
        'GNN',
        'MixingGWN', 
        'MixingConstant',
        'SpectralAlteration', 
        'PrincipalSubspaceAlteration',
        'TimeWarping',
        'Clipping',
        'Dead-Zone'
    ]

    # create anomalies class instance
    anomalies = [
        GWN(delta),
        Impulse(delta),
        Step(delta),
        Constant(delta),  
        GNN(delta),
        MixingGWN(delta),
        MixingConstant(delta),
        SpectralAlteration(delta),
        PrincipalSubspaceAlteration(delta),
        TimeWarping(delta), 
        Clipping(delta),
        DeadZone(delta)
    ]

    anomalies_dict = dict(zip(anomalies_labels, anomalies))

    # initialize the dataframe for anomalous data
    Xko_df = pd.DataFrame(
        index=np.arange(N_test),
        columns=pd.MultiIndex.from_product([anomalies_labels, np.arange(n)])
    )

    Yko_df = pd.DataFrame(
        index=np.arange(N_test),
        columns=pd.MultiIndex.from_product([anomalies_labels, np.arange(m)]),
        dtype=np.float64
    )

    # generate anomalous data for each anomaly
    for anomaly_label, anomaly in tqdm.tqdm(anomalies_dict.items()):
        if anomaly_label in ['SpectralAlteration']:
            anomaly.fit(Xstd, isnr)
        else:
            anomaly.fit(Xstd)
        Xko_std = anomaly.distort(Xstd)
        deltahat = np.mean( (Xstd - Xko_std)**2 )
        print(
            f"\t{anomaly_label} delta={deltahat}"
            )
        Xko = Xko_std*std + mean
        Xko_df[anomaly_label] = Xko
        deltahat = np.mean( (X - Xko)**2 )
        print(
            f"\t{anomaly_label} delta non-std={deltahat}\n"
            )
        Yko = cs.encode(Xko)
        Yko_df[anomaly_label] = Yko

    # ------------------ Init and fit the detector ------------------
    #
    standard_detectors = ['SPE', 'T2', 'AR', 'OCSVM', 'LOF', 'IF', 'MD', 'energy', 'TV', 'ZC', 'pk-pk']
    # init the results 
    result = pd.Series(index=anomalies_labels, dtype=np.float64)

    # evaluate TSOC-based detector
    if 'TSOC' in detector_type:
        file_model = f'TSOC-N={N_train}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_mode={mode}_ort={orthogonal}_epochs={epochs}_bs={batch_size}_opt=sgd_lr={lr}'\
                f'_th={threshold}_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                f'_mind={min_delta}_seed={seed}.pth'
        detector = TSOCDetector(n, m, file_model, seed, mode=detector_mode, gpu=device)
        detector = detector.fit()

    # evaluate a standard detector
    # init the detector      
    elif detector_type in standard_detectors:
        if detector_type == 'SPE':
            detector = SPE(k)
            detector_label = f'SPE_{k}'
        elif detector_type == 'T2':
            detector = T2(k)
            detector_label = f'T2_{k}'
        elif detector_type == 'AR':
            detector = AR(order)
            detector_label = f'AR_{order}'
        elif detector_type == 'OCSVM':
            detector = OCSVM(kernel, nu)
            detector_label = f'OCSVM_{kernel}_{nu}'
        elif detector_type == 'LOF':
            detector = LOF(h)
            detector_label = f'LOF_{neighbors}'
        elif detector_type == 'IF':
            detector = IF(l)
            detector_label = f'IF_{estimators}'
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
        f'_isnr={isnr}_seed={seed}.pkl'
        model_path = os.path.join(detectors_folder, model_name)
        # load if already trained
        if os.path.exists(model_path):
            
            with open(model_path, 'rb') as f:
                detector = pickle.load(f)
        # fit
        else:
            data_path = os.path.join(data_folder, f'n{n}_ISNR', f'trSet_n={n}_isnr={isnr}_no-sparse.h5')
            with pd.HDFStore(data_path, mode='r') as store:
                X_train = store.select('X').values.squeeze()
            Y_train = cs.encode(X_train)

            train_size = int(train_fraction * len(Y_train))  # 80% for training
            val_size = len(Y_train) - train_size  # 20% for validation

            # Split the dataset into training and validation
            generator = torch.Generator()
            generator.manual_seed(seed)
            train_dataset, val_dataset = random_split(Y_train, [train_size, val_size], generator=generator)
            Y_train, Y_val = train_dataset.dataset, val_dataset.dataset
            detector = detector.fit(Y_train)

            # save
            with open(model_path,'wb') as f:
                pickle.dump(detector, f)
        
    # ------------------ Evaluate the detector for each anomaly ------------------
    for anomaly_label in tqdm.tqdm(anomalies_labels):
        # define the anomalous dataset
        Xko = Xko_df[anomaly_label].values
        if 'TSOC' in detector_type:
            Zanom = np.concatenate([X, Xko])
        elif detector_type in standard_detectors:
            Yko = cs.encode(Xko)
            Zanom = np.concatenate([Y, Yko])
        metric_value = detector.test(Zanom, metric='AUC')
        result.loc[anomaly_label] = metric_value

    results_path = os.path.join(results_folder, f'AUC-{detector_label}_N={N_train}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
        f'_isnr={isnr}_seed={seed}.pkl')
    pd.to_pickle(result, results_path)

# ------------------ Parser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate various detectors for anomaly detection tasks.")

    # Core arguments
    parser.add_argument('-n', '--n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('-m', '--m', type=int, required=True, help="Number of measurements")
    parser.add_argument('-s', '--seed', type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument('-md', '--mode', type=str, choices=['standard', 'rakeness'], required=True, help="Measurement matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('-i', '--isnr', type=int, required=True, help="Signal-to-noise ratio (SNR) in dB")
    parser.add_argument('-dt', '--detector_type', type=str, required=True, help="Type of detector to evaluate (e.g., TSOC, SPE, OCSVM, LOF)")
    parser.add_argument('-dlt', '--delta', type=float, required=True, help="Anomaly intensity parameter")
    parser.add_argument('-N', '--N_train', type=int, default=2_000_000, help="Number of training samples")
    parser.add_argument('-tf', '--train_fraction', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('-M', '--N_test', type=int, default=10_000, help="Number of test samples")
    parser.add_argument('-B', '--basis', type=str, default='sym6', help="Wavelet basis function")
    parser.add_argument('-f', '--fs', type=int, default=256, help="Sampling frequency")
    parser.add_argument('-hr', '--heart_rate', type=str, default='60,100', help="Heart rate range (comma-separated, e.g., 60,100)")
    parser.add_argument('-o', '--orthogonal', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('-p', '--processes', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('-g', '--gpu', type=int, default=3, help="GPU index to use for evaluation")

    # TSOC-related arguments
    parser.add_argument('-dmd', '--detector_mode', type=str, choices=['self-assessment', 'autoencoder'], help="Mode of operation of TSOC-based detector")
    parser.add_argument('-fct', '--factor', type=float, required=True, help="Factor for augmentation or scheduling")
    parser.add_argument('-minlr', '--min_lr', type=float, required=True, help="Minimum learning rate for optimizers")
    parser.add_argument('-mind', '--min_delta', type=float, required=True, help="Minimum change in monitored metric for early stopping")
    parser.add_argument('-pt', '--patience', type=int, required=True, help="Patience for early stopping in epochs")
    parser.add_argument('-b', '--batch_size', type=int, required=True, help="Batch size for training")
    parser.add_argument('-t', '--threshold', type=float, required=True, help="Threshold for TSOC detector")
    parser.add_argument('-e', '--epochs', type=int, required=True, help="Number of training epochs")
    parser.add_argument('-l', '--lr', type=float, required=True, help="Learning rate")

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
        seed=args.seed,
        processes=args.processes,
        threshold=args.threshold,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        factor=args.factor,
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience,
        detector_type=args.detector_type,
        delta=args.delta,
        N_test=args.N_test,
        detector_mode=args.detector_mode,
        k=args.k,
        order=args.order,
        kernel=args.kernel,
        nu=args.nu,
        neighbors=args.neighbors,
        estimators=args.estimators
    )



    
