import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6"


import sys
import torch
import numpy as np
from scipy.fft import dct
import pandas as pd
import pickle
import tqdm
import argparse
import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir
from cs import CompressedSensing, generate_sensing_matrix
from detectors.tsoc_bridge import TSOCDetector
from detectors.bpdn import BPDNDetector
from detectors import detectors_dir
from models import models_dir



logging.basicConfig(level=logging.DEBUG)

def test(
    n, m, window_size, fraction, epochs, lr, batch_size, fs, mode, 
    orthogonal, source, seed_matrix, seed_detector, processes, threshold, gpu, train_fraction, factor, alpha, min_lr, 
    min_delta, patience, opt, detector_type, detector_mode, 
    k, order, kernel, kernel2, nu, neighbors, estimators, bandwidth, sigma, version
):
       # ------------------ Constants ------------------
    M = 100 # number of evaluated sensing matrices
    seed_data = 0
    seed_training = 0
    if bandwidth == 0:
        bandwidth = 'silverman'
    encoder_type = 'CS'
    radius = 1.0
    standard_detectors = ['SPE', 'T2', 'AR', 'OC', 'LOF', 'KDE', 'IF', 'MD', 'energy', 'TV', 'ZC', 'pk-pk']
    estimators_list = ['random_forest', 'logistic_regression', 'SVM', 'naive_bayes', 'Sparse SVM', 'k_neighbors', 'radius_neighbors']

    # ------------------ Folders ------------------
    model_folder = f'{models_dir}{detector_type}'
    results_folder = '/srv/newpenny/dnn-cs/tsoc/results/TSOC/detection'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Evaluating {detector_type} detector with parameters: {params_str}")

    # ------------------ GPU ------------------
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # ------------------ Signal ------------------
    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_fraction={fraction}_seed={seed_data}'
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_augmented = pickle.load(f)

     # load training data
    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_train = pickle.load(f)

    # load test data
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')
    data_name = f'bridge_ok_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_ok = pickle.load(f)

    data_name = f'bridge_ko_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_ko = pickle.load(f)

    N = min(len(X_ok), len(X_ko))
    X_ok = X_ok[:N]
    X_ko = X_ko[:N]
    
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

    if detector_type=='BPDN':
        cs = CompressedSensing(A, D, decoder=detector_type, processes=processes, sigma=sigma)
    elif detector_type=='TSOC' or detector_type in estimators_list:
        cs = CompressedSensing(A, D, decoder='SO')
    else:
        cs = CompressedSensing(A, D)
    Y_ok = cs.encode(X_ok)  # measurements
    Y_ko = cs.encode(X_ko)  # measurements

    # ------------------ Init and fit the detector ------------------
    
    # evaluate TSOC-based detector
    if detector_type =='TSOC':
        alpha_string = f'alpha={alpha}' if detector_type == 'TSOC' else ''
        if version == 'v1':
            architecture = 'TSOC'
        elif version == 'v2':
            architecture = 'TSOCv2'
        model_name = f'{architecture}-N={len(X_augmented)}_n={n}_ws={window_size}_fraction={fraction}_m={m}_fs={fs}_mode={mode}'\
                 f'_src={source}_ort={orthogonal}_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                 f'_th={threshold}_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                 f'_mind={min_delta}_seedtraining={seed_training}_seedmatrix={seed_matrix}_seeddata={seed_data}'
        file_name = f'bridge-AUC_detector={model_name}.pkl'
        subfolder = f'{detector_type}_{mode}'   

        model_path = os.path.join(model_folder, alpha_string, f'bridge-{model_name}.pth')
        results_path = os.path.join(results_folder, subfolder, alpha_string, detector_mode, file_name)

        detector_label = model_name

        detector_path = os.path.join(detectors_dir, detector_type, f'bridge-{model_name}.pkl')
        if os.path.exists(detector_path):
            with open(detector_path, 'rb') as f:
                detector = pickle.load(f)
        else:
            print(f'Detector\n{model_name}\nhas not been trained')
            sys.exit(0)
        detector.mode = detector_mode

    elif detector_type in estimators_list: 
        if detector_type=='random_forest': estimator_label = f'{detector_type}-ne_{estimators}'
        elif detector_type=='k_neighbors': estimator_label = f'{detector_type}-nn_{neighbors}'
        elif detector_type=='radius_neighbors': estimator_label = f'{detector_type}-radius={radius}'
        elif detector_type=='SVM': estimator_label = f'{detector_type}-krn={kernel}'
        model_name = f'{estimator_label}-N={len(X_train)}_n={n}_ws={window_size}_fraction={fraction}_m={m}_fs={fs}_mode={mode}'\
                    f'_src={source}_ort={orthogonal}_th={threshold}_seedtraining={seed_training}_seedmatrix={seed_matrix}_seeddata={seed_data}'
        model_path = os.path.join(models_dir, detector_type, f'{model_name}.pkl')

        file_name = f'bridge-AUC_detector={model_name}.pkl'
        subfolder = f'{detector_type}_{mode}' 

        results_path = os.path.join(results_folder, subfolder, detector_mode, file_name)

        detector_label = model_name
        detector_path = os.path.join(detectors_dir, detector_type, f'bridge-{model_name}.pkl')
        if os.path.exists(detector_path):
            with open(detector_path, 'rb') as f:
                detector = pickle.load(f)
        else:
            print(f'Detector\n{model_name}\nhas not been trained')
            sys.exit(0)
        detector.mode = detector_mode

    # elif detector_type == 'BPDN':
    #     name = f'{detector_type}-n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
    #              f'_isnr={isnr}_mode={mode}_src={source}_ort={orthogonal}_seedmat={seed_matrix}_sigma={sigma}'
    #     if mode == 'rakeness':
    #         name = f'{name}_corr={corr_name}_loc={loc}'
    #     file_name = f'AUC_detector={name}_delta={delta}_seedko={seed_ko}.pkl'
    #     results_path = os.path.join(results_folder, detector_type, detector_mode, file_name)
    #     detector = BPDNDetector(cs, mode=detector_mode)
    #     detector_label = name
        
    # evaluate a standard detector
    # init the detector


    elif detector_type in standard_detectors:
        if detector_type == 'SPE':
            detector_label = f'{detector_type}_k={k}'
        elif detector_type == 'T2':
            detector_label = f'{detector_type}_k={k}'
        elif detector_type == 'AR':
            detector_label = f'{detector_type}_order={order}'
        elif detector_type == 'OC':
            detector_label = f'{detector_type}_krn={kernel}_nu={nu}'
        elif detector_type == 'LOF':
            detector_label = f'{detector_type}_neighbors={neighbors}'
        elif detector_type == 'IF':
            detector_label = f'{detector_type}_estimators={estimators}'
        elif detector_type == 'KDE':
            detector_label = f'{detector_type}_krn={kernel2}_bw={bandwidth}'
        else:
            detector_label = detector_type
        
        name= f'N={len(X_train)}_n={n}_ws={window_size}_fraction={fraction}_m={m}_fs={fs}'\
                f'_mode={mode}_src={source}_ort={orthogonal}_seedmat={seed_matrix}_tf={train_fraction}_seeddet={seed_detector}'\
                f'_seeddata={seed_data}_seedtrain={seed_training}'

        # load the detector
        model_name = f'{detector_label}_{name}'
        print(model_name)
        model_path = os.path.join(detectors_dir, f'bridge_{model_name}.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                detector = pickle.load(f)
        else:
            print(f'\ndetector {detector_label} has not been trained')
            sys.exit(0)

        # load the scaler
        if (detector_type in standard_detectors) and (detector_type not in ['LOF', 'OC', 'IF', 'KDE']):
            scaler_name = f'bridge_scaler_{name}'
            scaler_path = os.path.join(detectors_dir, f'{scaler_name}.pkl')
            # load scaler
            if os.path.exists(scaler_path):
                print(f'scaler already trained')
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                print(f'\nscaler has not been trained')
                sys.exit(0)

        results_path = os.path.join(results_folder, encoder_type, f'bridge_AUC_detector={model_name}.pkl')

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if os.path.exists(results_path):
        print(f'\ndetector {detector_label} has been already evaluated')
        sys.exit(0)
    
    # ------------------ Evaluate the detector for each anomaly ------------------
    result = pd.Series(index=['real'], dtype=np.float64)
    if detector_type in ['TSOC', 'NNdecoder', 'AE', 'BPDN', *estimators_list] :
        Zanom = np.concatenate([X_ok, X_ko])
    elif detector_type in standard_detectors:
        Zanom = np.concatenate([Y_ok, Y_ko])
        # center data for not machine learning based detectors
        if detector_type not in ['LOF', 'OC', 'IF', 'KDE']:
            Zanom = scaler.transform(Zanom)
    metric_value = detector.test(Zanom, metric='AUC')
    result.loc['real'] = metric_value

    pd.to_pickle(result, results_path)

# ------------------ Parser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate various detectors for anomaly detection tasks.")

    # Core arguments
    parser.add_argument('-n', '--n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('-m', '--m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--window_size', '-w', type=int, default=20, help="Number of consecutive windows")
    parser.add_argument('--fraction', '-F', type=int, default=40, help="Fraction of overlapping windows used for training")
    parser.add_argument('-md', '--mode', type=str, choices=['standard', 'rakeness'], required=True, help="Sensing matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('-S', '--seed_matrix', type=int, required=True, help="Seed for random or index for (one of) the best sensing matrix")
    parser.add_argument('-dt', '--detector_type', type=str, required=True, help="Type of detector to evaluate (e.g., TSOC, SPE, OC, LOF)")
    parser.add_argument('-tf', '--train_fraction', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('-f', '--fs', type=int, default=256, help="Sampling frequency")
    parser.add_argument('-o', '--orthogonal', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--source', '-src', type=str, choices=['best', 'random'], default='best', help="Sensing matrix type: genereated randomly or leading to best performance")
    parser.add_argument('-p', '--processes', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('-g', '--gpu', type=int, default=3, help="GPU index to use for evaluation")

    # NNdecoder/TSOC-related arguments
    parser.add_argument('-dmd', '--detector_mode', type=str,
                         choices=['self-assessment', 'autoencoder', 'sparsity', 'sparsity-threshold',
                                   'self-assessment-complement', 'complement', 'MD', 'norm', 'OC',
                                     'self-assessment2', 'self-assessment3', 'MDhat',
                                     'self-assessment-MD', 'complement-MD', 'self-assessment-complement-MD',
                                     'self-assessment-complement-MD-CDF', 'self-assessment-complement-MDy'],
                         help="Mode of operation of NNdecoder/TSOC-based detector")
    parser.add_argument('-fct', '--factor', type=float, default=0.2, help="Factor for augmentation or scheduling")
    parser.add_argument('-minlr', '--min_lr', type=float, default=0.001, help="Minimum learning rate for optimizers")
    parser.add_argument('-mind', '--min_delta', type=float, default=1e-4, help="Minimum change in monitored metric for early stopping")
    parser.add_argument('-pt', '--patience', type=int, default=40, help="Patience for early stopping in epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=50, help="Batch size for training")
    parser.add_argument('-e', '--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('-l', '--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer used for training")
    # TSOC-related arguments  
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help="Threshold for TSOC detector")
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help="Training loss weight for TSOC detector")
    # BPDN-related arguments 
    parser.add_argument('-z', '--sigma', type=float, default=1e-4, help="BPDN threshold")

    # Standard detector-related arguments (optional for specific detectors)
    parser.add_argument('-Ss', '--seed_detector', type=int, default=0, help="Random seed associated to the detector")
    parser.add_argument('-k', '--k', type=int, default=5, help="Parameter k for SPE, T2, LOF, or related detectors")
    parser.add_argument('-ord', '--order', type=int, default=1, help="Order parameter for AR detector")
    parser.add_argument('-krn', '--kernel', type=str, default='rbf', help="Kernel type for OC (e.g., linear, rbf, poly)")
    parser.add_argument('-nu', '--nu', type=float, default=0.5, help="Anomaly fraction for OC")
    parser.add_argument('-nn', '--neighbors', type=int, default=10, help="Number of neighbors for LOF detector")
    parser.add_argument('-est', '--estimators', type=int, default=100, help="Number of estimators for IF detector")
    parser.add_argument('-krn2', '--kernel2', type=str, default='gaussian', help="Kernel type for KDE")
    parser.add_argument('-bw', '--bandwidth', type=float, default=0.1, help="Bandwidth for KDE")
    parser.add_argument('--version', '-V', type=str, default='v1', help="TSOC version")
    


    return parser.parse_args()


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()
    # Call the training function with parsed arguments
    test(
        n=args.n,
        m=args.m,
        window_size=args.window_size,
        fraction=args.fraction,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        detector_type=args.detector_type,
        fs=args.fs,
        mode=args.mode,
        orthogonal=args.orthogonal,
        source=args.source,
        seed_matrix=args.seed_matrix,
        seed_detector=args.seed_detector,
        processes=args.processes,
        threshold=args.threshold,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        factor=args.factor,
        alpha=args.alpha,
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience,
        opt=args.optimizer,
        detector_mode=args.detector_mode,
        k=args.k,
        order=args.order,
        kernel=args.kernel,
        kernel2=args.kernel2,
        nu=args.nu,
        neighbors=args.neighbors,
        estimators=args.estimators,
        bandwidth=args.bandwidth,
        sigma=args.sigma,
        version=args.version
    )



    
