import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np
from scipy.fft import dct
import pickle
import argparse
import logging

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from models import models_dir
from dataset import dataset_dir
from detectors.support_estimator_detector import SupportEstimatorDetector
from detectors import detectors_dir
from cs import CompressedSensing, generate_sensing_matrix



logging.basicConfig(level=logging.DEBUG)

def training(
    n, m, window_size, fraction, fs, mode, orthogonal, source, seed_training,
      seed_matrix, seed_data, processes, threshold, estimator_type, n_estimators, version
):

    M = 100 # number of evaluated sensing matrices
    kernel = 'rbf'
    n_neighbors = 5
    radius = 1.0

    # ------------------ Reproducibility ------------------
    np.random.seed(seed_training) 

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running training with parameters: {params_str}")

    # ------------------ Signal ------------------
    data_name = f'bridge_train_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)

    # ------------------ Compressed Sensing ------------------
    D = dct(np.eye(n), axis=0, norm='ortho')

    # Sensing matrix
    C = (X.T @ X) / len(X)
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

    cs = CompressedSensing(A, D, decoder='SO')

    # ------------------ Model path ------------------
    if estimator_type=='random_forest': estimator_label = f'{estimator_type}-ne_{n_estimators}'
    elif estimator_type=='k_neighbors': estimator_label = f'{estimator_type}-nn_{n_neighbors}'
    elif estimator_type=='radius_neighbors': estimator_label = f'{estimator_type}-radius={radius}'
    elif estimator_type=='SVM': estimator_label = f'{estimator_type}-krn={kernel}'
    model_name = f'bridge-{estimator_label}-N={len(X)}_n={n}_ws={window_size}_fraction={fraction}_m={m}_fs={fs}_mode={mode}'\
                 f'_src={source}_ort={orthogonal}_th={threshold}_seedtraining={seed_training}_seedmatrix={seed_matrix}_seeddata={seed_data}'
    model_path = os.path.join(models_dir, estimator_type, f'{model_name}.pkl')

    detector_path = os.path.join(detectors_dir, estimator_type, f'{model_name}.pkl')
    os.makedirs(os.path.dirname(detector_path), exist_ok=True)
    
    # ------------------ Train detector ------------------
    # stop if already trained
    if os.path.exists(detector_path):
        print(f'The detectors has been already trained')
        sys.exit(0)
    # train the detector
    else:
        # init
        detector = SupportEstimatorDetector(cs, model_path, threshold=threshold, filter=True, version=version)
        # fit
        detector = detector.fit(X_train=X)
 
        # save
        with open(detector_path,'wb') as f:
            pickle.dump(detector, f)

# ------------------ Perser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for Support Estimator-based Detector")

    # Short and long argument versions
    parser.add_argument('--n', '-n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('--m', '-m', type=int, required=True, help="Number of measurements")
    parser.add_argument('--window_size', '-w', type=int, default=20, help="Number of consecutive windows")
    parser.add_argument('--fraction', '-F', type=int, default=40, help="Fraction of overlapping windows used for training")
    parser.add_argument('--fs', '-f', type=int, default=50, help="Sampling frequency")
    parser.add_argument('--mode', '-M', type=str, choices=['standard', 'rakeness'], required=True, help="Measurement matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('--orthogonal', '-o', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--source', '-c', type=str, choices=['best', 'random'], default='best', help="Measurement matrix type: genereated randomly or leading to best performance")
    parser.add_argument('--seed_training', '-s', type=int, required=True, help="Training-related random seed for reproducibility")
    parser.add_argument('--seed_matrix', '-S', type=int, default=0, help="Seed for random or index for (one of) the best Measurement matrix")
    parser.add_argument('--seed_data', '-D', type=int, default=0, help="Seed for training data")
    parser.add_argument('--processes', '-p', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help="Threshold for metrics")
    parser.add_argument('--estimator_type', '-E', type=str, default='random_forest', help="Machine learning-based classifier for support estimation")
    parser.add_argument('--n_estimators', '-e', type=int, default=100, help="Number of estimators for Random Forest Classifier")
    parser.add_argument('--version', '-V', type=str, default='v1', help="Estimator version")

    return parser.parse_args()

# ------------------ Main call ------------------
if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()
    
    # Call the training function with parsed arguments
    training(
        n=args.n,
        m=args.m,
        window_size=args.window_size,
        fraction=args.fraction,
        fs=args.fs,
        mode=args.mode,  
        orthogonal=args.orthogonal,
        source=args.source,
        seed_training=args.seed_training,
        seed_matrix = args.seed_matrix,
        seed_data =args.seed_data,
        processes=args.processes,
        threshold=args.threshold,
        estimator_type=args.estimator_type,
        n_estimators = args.n_estimators,
        version = args.version
    )
