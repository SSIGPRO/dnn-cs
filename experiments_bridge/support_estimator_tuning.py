import os

import sys
import numpy as np
import pandas as pd
import pickle
import argparse
import logging

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))
from dataset import dataset_dir

from wombats.detectors.gaussian_distribution_based import *


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ray import tune, init

from scipy.fft import dct
from scipy import linalg

from cs import CompressedSensing, generate_sensing_matrix

def compute_rsnr2(x, x_hat, axis=-1):
    ''' Compute Reconstruction Signal-to-Noise Ratio '''
    norm_signal = linalg.norm(x, axis=axis)
    norm_noise = linalg.norm(x - x_hat, axis=axis)
    rsnr = 20 * np.log10(norm_signal / norm_noise)
    return rsnr

def reconstructor(dnn_output, y, A, D, th=0.5):
    ''' Recover the singal based on the oracle prior '''
    B = A @ D
    zhat = dnn_output > th
    BTS = linalg.pinv(B[:, zhat])
    xihat = BTS @ y
    xhat = D[:, zhat] @ xihat
    return xhat

logging.basicConfig(level=logging.DEBUG)

def tuning(
    n, m, window_size, fs, mode, orthogonal, source, seed_training,
      seed_matrix, seed_data, processes, threshold, estimator_type,
      num_samples
):
    M = 100 # number of evaluated sensing matrices
    method = 'TSOC'
    
        # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Running training with parameters: {params_str}")


    files_dir = os.path.join(dataset_dir, f'bridge_n={n}_fs={fs}_seed={seed_data}')

    data_path = os.path.join(files_dir, f'bridge_train_n={n}_fs={fs}_ws={window_size}_seed={seed_data}.pkl')
    with open(data_path, 'rb') as f:
        X_train = pickle.load(f) 

    data_name = f'bridge_ok_n={n}_fs={fs}_ws={window_size}_seed={seed_data}'
    data_path = os.path.join(files_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X_test = pickle.load(f)

    if source == 'best':
        fname =f'matrices_ort={orthogonal}_mode={mode}_M={M}_m={m}_mode={mode}_seed={seed_data}'
        path = os.path.join(files_dir, f'Best_sensing_matrices', fname+'.pkl')
        with open(path, 'rb') as f:
            A_dict = pickle.load(f)
        A = A_dict[seed_matrix]['matrix']
        seed_matrix = A_dict[seed_matrix]['seed']

    C = (X_train.T @ X_train) / len(X_train)
    loc = np.trace(C @ C) / np.trace(C)**2 - 1 / n
    D = dct(np.eye(n), axis=0, norm='ortho')
    A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orthogonal, correlation=C, loc=loc, seed=seed_matrix)
    B = A @ D
    cs = CompressedSensing(A, D=D, decoder='SO')
    Y_train = cs.encode(X_train)
    Y_test = cs.encode(X_test)

    supports_name_ = f'supports_method={method}_mode={mode}_m={m}_orth={orthogonal}'
    supports_path = os.path.join(files_dir, f'bridge_train_{supports_name_}_ws={window_size}_seed_matrix={seed_matrix}_seed_data={seed_data}.pkl')
    with open(supports_path, 'rb') as f:
        Z_train = pickle.load(f)  


    def objective(config, estimator_type):
        
        if estimator_type == 'random_forest':
            # Extract hyperparameters
            n_estimators = int(config["n_estimators"])
            max_depth = config["max_depth"]
            min_samples_split = config["min_samples_split"]
            min_samples_leaf = config["min_samples_leaf"]
            
            # Create the Random Forest classifier
            estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=seed_training
            )

        elif estimator_type == 'svm':
            # Extract hyperparameters
            C = config["C"]
            kernel = config["kernel"]
            gamma = config["gamma"]

            # Create the SVM classifier
            estimator = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                probability=True,
                random_state=seed_training
            )

        estimator = MultiOutputClassifier(
            make_pipeline(
            StandardScaler(),
            estimator
            )
        )

        # Fit the model
        estimator.fit(Y_train @ B, Z_train.astype(int))
        
        # Make predictions
        Z_hat = estimator.predict(Y_test @ B)
        
        # Decode predictions
        Xhat = np.zeros(Z_hat.shape)
        idx = Z_hat.sum(axis=-1) > 0
        Xhat[idx] = np.array([reconstructor(z, y, A, D, th=threshold) for z, y in zip(Z_hat[idx].astype(bool), Y_test[idx])])
        
        # Compute the custom RSNR metric (higher is better)
        metric = np.mean(compute_rsnr2(X_test, Xhat))
        
        # Ray Tune maximizes the metric, so return the result
        tune.report(rsnr=metric)

    # Define the search space for the hyperparameters
    search_space ={'random_forest': {
                            "n_estimators": tune.qloguniform(50, 1000, 1), 
                            "max_depth": tune.randint(5, 50),
                            "min_samples_split": tune.choice([2, 5, 10, 20]),
                            "min_samples_leaf": tune.choice([1, 2, 4, 8]),
                            "max_features": tune.choice([None, "sqrt", "log2"]),
                            "bootstrap": tune.choice([True, False]),
                            "criterion": tune.choice(["gini", "entropy"]),
                            },
                    'svm': {
                            "C": tune.loguniform(0.1, 1000),
                            "kernel": tune.choice(["linear", "poly", "rbf", "sigmoid"]),
                            "gamma": tune.choice(["scale", "auto"]),
                            }
    }

    init(num_cpus=processes)

    # Run the Ray Tune experiment
    analysis = tune.run(
        lambda config: objective(config, estimator_type),
        config=search_space[estimator_type], 
        num_samples=num_samples,  # Number of different configurations to try
        resources_per_trial={"cpu": 1}, 
        metric="rsnr",  # Metric to optimize
        mode="max",  # maximize RSNR
        verbose=1
    )

    # Get the best configuration
    print("Best hyperparameters found were: ", analysis.best_config)


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
    parser.add_argument('--source', '-c', type=str, choices=['best', 'random'], default='best', help="Measurement matrix type: genereated randomly or leading to best performance")
    parser.add_argument('--seed_training', '-s', type=int, required=True, help="Training-related random seed for reproducibility")
    parser.add_argument('--seed_matrix', '-S', type=int, default=0, help="Seed for random or index for (one of) the best Measurement matrix")
    parser.add_argument('--seed_data', '-D', type=int, default=0, help="Seed for training data")
    parser.add_argument('--processes', '-p', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('--num_samples', '-N', type=int, default=100, help="Number of samples evaluated during tuning")
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help="Threshold for metrics")
    parser.add_argument('--estimator_type', '-E', type=str, default='random_forest', help="Machine learning-based classifier for support estimation")

    return parser.parse_args()

# ------------------ Main call ------------------
if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()
    
    # Call the training function with parsed arguments
    tuning(
        n=args.n,
        m=args.m,
        window_size=args.window_size,
        fs=args.fs,
        mode=args.mode,  
        orthogonal=args.orthogonal,
        source=args.source,
        seed_training=args.seed_training,
        seed_matrix = args.seed_matrix,
        seed_data =args.seed_data,
        processes=args.processes,
        num_samples=args.num_samples,
        threshold=args.threshold,
        estimator_type=args.estimator_type,
    )

