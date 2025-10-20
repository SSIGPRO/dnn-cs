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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import tqdm
import argparse
import logging


from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

from sklearn.metrics import roc_auc_score 

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir


logging.basicConfig(level=logging.DEBUG)

def test(
    n, m, N_train, N_test, fs, heart_rate, isnr, delta, seed_ko
):
    # ------------------ Constants ------------------
    seed_train_data = 11
    seed_test_data = 66
    detector_type = 'PCA'
    
    results_folder = '/srv/newpenny/dnn-cs/tsoc/results/TSOC/detection'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Evaluating {detector_type} detector with parameters: {params_str}")

    # ------------------ Seeds ------------------
    np.random.seed(seed_ko)

    # ------------------ Signal ------------------
    # load test data
    data_name_train = f'ecg_N={N_train}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_train_data}'
    data_path = os.path.join(dataset_dir, data_name_train+'.pkl')
    with open(data_path, 'rb') as f:
        X_train = pickle.load(f)
    data_name = f'ecg_N={N_test}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_test_data}'
    data_path = os.path.join(dataset_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)
    
    # standarize the data
    std, mean = X.std(), X.mean()
    Xstd = (X - mean) / std

    # ------------------ Train PCA ------------------
    mean = np.mean(X_train, axis=0)
    X_train_ = X_train - mean
    Cx = 1 / (N_train -1 ) * X_train_.T @ X_train_
    v, V = np.linalg.eigh(Cx)
    v = v[::-1]
    V = V[:, ::-1]
    w = v[:m]
    W = V[:, :m]

    # ------------------ Define T2 score ------------------
    T2 = lambda X_: np.sqrt(np.sum(((X_ - mean) @ W)**2 / w, axis=-1))
    
    results_name = f'pca-T2-N={N_train}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}_isnr={isnr}_delta={delta}_seedko={seed_ko}.pkl'
    results_path = os.path.join(results_folder, detector_type, 'T2', results_name)

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if os.path.exists(results_path):
        print(f'\ndetector has been already evaluated')
        sys.exit(0)

       # ------------------ Anomalies generation ------------------
    # initialize anomalies

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
        columns=pd.MultiIndex.from_product([anomalies_labels, np.arange(n)]),
        dtype=np.float64
    )

    # generate anomalous data for each anomaly
    logging.info(f"\nGenerating anomalies")
    for anomaly_label, anomaly in anomalies_dict.items():
        if anomaly_label in ['SpectralAlteration']:
            anomaly.fit(Xstd, isnr)
        else:
            anomaly.fit(Xstd)
        Xko_std = anomaly.distort(Xstd)
        Xko = Xko_std*std + mean
        Xko_df[anomaly_label] = Xko

    # init the results 
    result = pd.Series(index=anomalies_labels, dtype=np.float64)
    logging.info(f"\nEvaluating performance")
    for anomaly_label in tqdm.tqdm(anomalies_labels):
        Xko = Xko_df[anomaly_label].values
        scoreok = T2(X)
        scoreko = T2(Xko)
        label = np.concatenate([np.zeros(len(X)), np.ones(len(X))])
        score = np.concatenate([scoreok, scoreko])
        AUC = roc_auc_score(label, score)
        result.loc[anomaly_label] = AUC

    pd.to_pickle(result, results_path)

# ------------------ Parser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly detection Script for AE Unet Model")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--isnr', '-i', type=int, required=True, help="Signal-to-noise ratio (SNR)")
    parser.add_argument('-N', '--N_train', type=int, default=2000000, help="Number of training samples")
    parser.add_argument('-M', '--N_test', type=int, default=10000, help="Number of test samples")
    parser.add_argument('--fs', '-f', type=int, default=256, help="Sampling frequency")
    parser.add_argument('--heart_rate', '-r', type=int, nargs=2, default=(60, 100), help="Heart rate range")
    parser.add_argument('--delta', '-k', type=float, required=True, help="Anomaly intensity parameter")
    parser.add_argument('--seed_ko', '-z', type=int, required=True, help="Random seed for reproducibility")
    
    
    return parser.parse_args()

# ------------------ Main call ------------------
if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()
    
    # Call the training function with parsed arguments
    test(
        n=args.n,
        m=args.m,
        N_train=args.N_train,
        N_test=args.N_test,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
        delta=args.delta,
        seed_ko=args.seed_ko,
    )



    
