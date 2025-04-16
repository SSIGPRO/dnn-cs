import os

# os.environ['MKL_THREADING_LAYER'] = 'GNU'
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# # The number of used threads is set here. Some detectors, such as LOF, can take huge advantage from parallelization 
# os.environ["OMP_NUM_THREADS"] = "20" 
# os.environ["OPENBLAS_NUM_THREADS"] = "20" 
# os.environ["MKL_NUM_THREADS"] = "20" 
# os.environ["VECLIB_MAXIMUM_THREADS"] = "20" 
# os.environ["NUMEXPR_NUM_THREADS"] = "20"


import sys
import torch
import numpy as np
import pandas as pd
import pickle
import tqdm
import argparse
import logging

from wombats.anomalies.invariant import SpectralAlteration, PrincipalSubspaceAlteration

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir
from cs.wavelet_basis import wavelet_basis
from cs import CompressedSensing, generate_sensing_matrix
from detectors.tsoc import TSOCDetector
from detectors.NNdecoder import NNdecoderDetector
from detectors.ae import AEDetector
from detectors.bpdn import BPDNDetector
from detectors import detectors_dir
from models import models_dir


logging.basicConfig(level=logging.DEBUG)

def test(
    n, m, epochs, lr, batch_size, N_train, num_anomalies, basis, fs, heart_rate, isnr, mode, 
    orthogonal, source, seed_matrix, seed_detector, seed_ko, processes, threshold, gpu, train_fraction, factor, alpha, min_lr, 
    min_delta, patience, opt, detector_type, beta, gamma, delta, N_test, detector_mode, 
    k, order, kernel, nu, neighbors, estimators, sigma
):
    # ------------------ Constants ------------------
    corr_name = '96af96a7ddfcb2f6059092c250e18f2a'
    seed_train_data = 11
    seed_test_data = 66
    seed_training = 0 # seed for training data split
    seed_data_matrix = 0 # seed used to generate data for sensing matrix selection
    seed_selection = 0 # seed used to select the best sensing matrix on data geneerated with seed_data_matrix
    # M = 1_000
    if m==48:
        M = 1_000 # number of evaluated sensing matrices
    else:
        M = 100
    loc = 0.25
    encoder_type = 'CS'

    # ------------------ Folders ------------------
    model_folder = f'{models_dir}{detector_type}'
    results_folder = '/srv/newpenny/dnn-cs/tsoc/results/TSOC/detection'

    # ------------------ Show parameter values ------------------
    params = locals()
    params_str = ", ".join(f"{key}={value}" for key, value in params.items())
    logging.info(f"Evaluating {detector_type} detector with parameters: {params_str}")

    
    # ------------------ Seeds ------------------
    np.random.seed(seed_ko)

    # ------------------ GPU ------------------
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # ------------------ Signal ------------------
    # load test data
    data_name = f'ecg_N={N_test}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed_test_data}'
    data_path = os.path.join(dataset_dir, data_name+'.pkl')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)
    
    # standarize the data
    std, mean = X.std(), X.mean()
    Xstd = (X - mean) / std

    # ------------------ Compressed Sensing ------------------
    D = wavelet_basis(n, basis, level=2)

    # Sensing matrix
    if source == 'random':
        # Generate a random sensing matrix
        if mode == 'rakeness':
            corr_path = os.path.join(dataset_dir, 'correlation', f'{corr_name}.pkl')
            with open(corr_path, 'rb') as f:
                C = pickle.load(f)
        else:
            C = None
        A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orthogonal, correlation=C, loc=loc, seed=seed_matrix)
    elif source == 'best':
        # Load the best sensing matrix

        A_folder = f'ecg_N=10000_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}_isnr={isnr}_seed={seed_data_matrix}'
        A_name = f'sensing_matrix_M={M}_m={m}_mode={mode}_seed={seed_selection}'
        if mode == 'rakeness':
            A_name = f'{A_name}_loc={loc}_corr={corr_name}'
        data_path = os.path.join(dataset_dir, A_folder, 'A_Filippo', f'{A_name}.pkl')
        with open(data_path, 'rb') as f:
            A_dict = pickle.load(f)
        A = A_dict[seed_matrix]['matrix']
        seed_matrix = A_dict[seed_matrix]['seed']
        logging.info(f'sensing matrix ({m}, {n}) with seed={seed_matrix} loaded')

    if detector_type=='BPDN':
        cs = CompressedSensing(A, D, decoder=detector_type, processes=processes, sigma=sigma)
    elif detector_type=='TSOC':
        cs = CompressedSensing(A, D, decoder='SO')
    else:
        cs = CompressedSensing(A, D)
    Y = cs.encode(X)  # measurements

    # ------------------ Init and fit the detector ------------------
    #
    standard_detectors = ['SPE', 'T2', 'AR', 'OC', 'LOF', 'IF', 'MD', 'energy', 'TV', 'ZC', 'pk-pk']

    # evaluate TSOC-based detector
    if detector_type in ['TSOC', 'NNdecoder']:
        th_string = f'_th={threshold}' if detector_type == 'TSOC' else ''
        alpha_string = f'alpha={alpha}' if detector_type == 'TSOC' else ''
        model_name = f'{detector_type}-N={N_train}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                     f'_isnr={isnr}_mode={mode}_src={source}_ort={orthogonal}_seedmat={seed_matrix}'\
                     f'_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                     f'{th_string}_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                     f'_mind={min_delta}_seeddata={seed_train_data}_seedtrain={seed_training}'
        file_name = f'AUC_cov_alt_detector={model_name}_delta={delta}_seedko={seed_ko}_na={num_anomalies}.pkl'
        subfolder = f'{detector_type}_{mode}'   
        if mode == 'rakeness':
            model_name = f'{model_name}_corr={corr_name}_loc={loc}'
            subfolder = f'{subfolder}_corr={corr_name}_loc={loc}'

        model_path = os.path.join(model_folder, alpha_string, f'{model_name}.pth')
        results_path = os.path.join(results_folder, subfolder, alpha_string, detector_mode, file_name)

        detector_label = model_name
        if detector_type == 'TSOC':
            if detector_mode in ['self-assessment-MD', 'complement-MD', 'self-assessment-complement-MD']:
                detector_path = os.path.join(detectors_dir, detector_type, f'{model_name}.pkl')
                if os.path.exists(detector_path):
                    with open(detector_path, 'rb') as f:
                        detector = pickle.load(f)
                else:
                    print(f'Detector\n{model_name}\nhas not been trained')
                    sys.exit(0)
                detector.mode = detector_mode
            else:
                detector = TSOCDetector(cs, model_path, mode=detector_mode,
                                        threshold=threshold, filter=True, gpu=device)
                detector = detector.fit()
        else:
            detector = NNdecoderDetector(cs, model_path, mode=detector_mode, gpu=device)
            detector = detector.fit()
    
    elif detector_type == 'AE':
        model_name = f'{detector_type}-N={N_train}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                     f'_isnr={isnr}_beta={beta}_gamma={gamma}_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                     f'_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                     f'_mind={min_delta}_seeddata={seed_train_data}_seedtrain={seed_training}'
        file_name = f'AUC_cov_alt_detector={model_name}_delta={delta}_seedko={seed_ko}_na={num_anomalies}.pkl'

        model_path = os.path.join(model_folder, f'{model_name}.pth')
        results_path = os.path.join(results_folder, detector_type, detector_mode, file_name)

        detector_label = model_name
        if detector_mode in ['MD', 'MDhat', 'MDhat2', 'MDhat3', 'norm', 'OC']:
            detector_path = os.path.join(detectors_dir, 'AE', detector_mode, f'{model_name}.pkl')
            if os.path.exists(detector_path):
                with open(detector_path, 'rb') as f:
                    detector = pickle.load(f)
            else:
                RuntimeError(f'Detector\n{model_name}\nhas not been trained')
                sys.exit(0)
        else:
            detector = AEDetector(n, m, model_path, mode=detector_mode, gpu=device)
            detector = detector.fit()
    
    elif detector_type == 'BPDN':
        name = f'{detector_type}-n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                 f'_isnr={isnr}_mode={mode}_src={source}_ort={orthogonal}_seedmat={seed_matrix}_sigma={sigma}'
        if mode == 'rakeness':
            name = f'{name}_corr={corr_name}_loc={loc}'
        file_name = f'AUC_cov_alt_detector={name}_delta={delta}_seedko={seed_ko}.pkl'
        results_path = os.path.join(results_folder, detector_type, detector_mode, file_name)
        detector = BPDNDetector(cs, mode=detector_mode)
        detector_label = name
        
    # evaluate a standard detector
    # init the detector
    # # TODO: optimize this part (I don't need to init the detectors and the detector_label can be define more efficiently)      
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
        else:
            detector_label = detector_type
                   
        name = f'N={N_train}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_mode={mode}_src={source}_ort={orthogonal}_seedmat={seed_matrix}_tf={train_fraction}_seeddet={seed_detector}'\
                f'_seeddata={seed_train_data}_seedtrain={seed_training}'
        if mode == 'rakeness':
            name = f'{name}_corr={corr_name}_loc={loc}'

        # load the detector
        model_name = f'{detector_label}_{name}'
        model_path = os.path.join(detectors_dir, f'{model_name}.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                detector = pickle.load(f)
        else:
            print(f'\ndetector {detector_label} has not been trained')
            sys.exit(0)

        # load the scaler
        if (detector_type in standard_detectors) and (detector_type not in ['LOF', 'OC', 'IF']):
            scaler_name = f'scaler_{name}'
            scaler_path = os.path.join(detectors_dir, f'{scaler_name}.pkl')
            # load scaler
            if os.path.exists(scaler_path):
                print(f'scaler already trained')
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                print(f'\nscaler has not been trained')
                sys.exit(0)

        results_path = os.path.join(
            results_folder, encoder_type,
              f'AUC_cov_alt_detector={model_name}_delta={delta}_seedko={seed_ko}_na={num_anomalies}.pkl'
              )

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if os.path.exists(results_path):
        print(f'\ndetector {detector_label} has been already evaluated')
        sys.exit(0)

    # ------------------ Anomalies generation ------------------
    # initialize anomalies accounting for covariance alterations
    SA = SpectralAlteration(delta)
    SA.fit(Xstd, isnr)
    PSA = PrincipalSubspaceAlteration(delta)
    PSA.fit(Xstd)

    # initialize the dataframe for anomalous data
    XSA_df = pd.DataFrame(
        index=np.arange(N_test),
        columns=pd.MultiIndex.from_product([np.arange(num_anomalies), np.arange(n)]),
        dtype=np.float64
    )

    XPSA_df = pd.DataFrame(
        index=np.arange(N_test),
        columns=pd.MultiIndex.from_product([np.arange(num_anomalies), np.arange(n)]),
        dtype=np.float64
    )
    # generate anomalous data 
    for anomaly_idx in np.arange(num_anomalies):
        # generate anomalous data for Spectral Alteration anomaly
        Xko_std = SA.distort(Xstd)
        Xko = Xko_std*std + mean
        XSA_df[anomaly_idx] = Xko
        # generate anomalous data for Principal Subspace Alteration anomaly
        Xko_std = PSA.distort(Xstd)
        Xko = Xko_std*std + mean
        XPSA_df[anomaly_idx] = Xko
        
    
    # ------------------ Evaluate the detector for each anomaly ------------------
    # init the results 
    result = pd.DataFrame(index=np.arange(num_anomalies),
                           columns=['SpectralAlteration', 'PrincipalSubspaceAlteration'],
                             dtype=np.float64)
    
    for anomaly_label in ['SpectralAlteration', 'PrincipalSubspaceAlteration']:
        logging.info(f"\nEvaluating performance of {anomaly_label}")
        for anomaly_idx in tqdm.tqdm(np.arange(num_anomalies)):
            # define the anomalous dataset
            if anomaly_label == 'SpectralAlteration':
                Xko = XSA_df[anomaly_idx].values
            else:
                Xko = XPSA_df[anomaly_idx].values
            if detector_type in ['TSOC', 'NNdecoder', 'AE', 'BPDN'] :
                Zanom = np.concatenate([X, Xko])
            elif detector_type in standard_detectors:
                Yko = cs.encode(Xko)
                Zanom = np.concatenate([Y, Yko])
                # center data for not machine learning based detectors
                if detector_type not in ['LOF', 'OC', 'IF']:
                    Zanom = scaler.transform(Zanom)
            metric_value = detector.test(Zanom, metric='AUC')
            result.loc[anomaly_idx, anomaly_label] = metric_value

    pd.to_pickle(result, results_path)

# ------------------ Parser definition ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate various detectors for anomaly detection tasks.")

    # Core arguments
    parser.add_argument('-n', '--n', type=int, required=True, help="Number of samples per signal")
    parser.add_argument('-m', '--m', type=int, required=True, help="Number of measurements")
    parser.add_argument('-s', '--seed_ko', type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument('-md', '--mode', type=str, choices=['standard', 'rakeness'], required=True, help="Sensing matrix mode: 'standard' or 'rakeness'")
    parser.add_argument('-S', '--seed_matrix', type=int, required=True, help="Seed for random or index for (one of) the best sensing matrix")
    parser.add_argument('-i', '--isnr', type=int, required=True, help="Signal-to-noise ratio (SNR) in dB")
    parser.add_argument('-dt', '--detector_type', type=str, required=True, help="Type of detector to evaluate (e.g., TSOC, SPE, OC, LOF)")
    parser.add_argument('-dlt', '--delta', type=float, required=True, help="Anomaly intensity parameter")
    parser.add_argument('-N', '--N_train', type=int, default=2000000, help="Number of training samples")
    parser.add_argument('-na', '--num_anomalies', type=int, default=100, help="Number of anomalies to generate")
    parser.add_argument('-tf', '--train_fraction', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('-M', '--N_test', type=int, default=10000, help="Number of test samples")
    parser.add_argument('-B', '--basis', type=str, default='sym6', help="Wavelet basis function")
    parser.add_argument('-f', '--fs', type=int, default=256, help="Sampling frequency")
    parser.add_argument('--heart_rate', '-hr', type=int, nargs=2, default=(60, 100), help="Heart rate range")
    parser.add_argument('-o', '--orthogonal', action='store_true', help="Use orthogonalized measurement matrix (default: False)")
    parser.add_argument('--source', '-src', type=str, choices=['best', 'random'], default='best', help="Sensing matrix type: genereated randomly or leading to best performance")
    parser.add_argument('-p', '--processes', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('-g', '--gpu', type=int, default=3, help="GPU index to use for evaluation")

    # NNdecoder/TSOC-related arguments
    parser.add_argument('-dmd', '--detector_mode', type=str,
                         choices=['self-assessment', 'autoencoder', 'sparsity', 'sparsity-threshold',
                                   'self-assessment-complement', 'complement', 'MD', 'norm', 'OC',
                                     'self-assessment2', 'self-assessment3', 'MDhat', 'MDhat2', 'MDhat3',
                                     'self-assessment-MD', 'complement-MD', 'self-assessment-complement-MD'],
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
    # AE-related arguments  
    parser.add_argument('--beta', '-w', type=float, default=0.0, help="Latent volume regularization weight")
    parser.add_argument('--gamma', '-q', type=float, default=0.0, help="Regularization of the latten space reconstruction")
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
        N_test=args.N_test,
        num_anomalies=args.num_anomalies,
        detector_type=args.detector_type,
        delta=args.delta,
        basis=args.basis,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
        mode=args.mode,
        orthogonal=args.orthogonal,
        source=args.source,
        seed_matrix=args.seed_matrix,
        seed_detector=args.seed_detector,
        seed_ko=args.seed_ko,
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
        beta=args.beta,
        gamma=args.gamma,
        k=args.k,
        order=args.order,
        kernel=args.kernel,
        nu=args.nu,
        neighbors=args.neighbors,
        estimators=args.estimators,
        sigma=args.sigma
    )



    
