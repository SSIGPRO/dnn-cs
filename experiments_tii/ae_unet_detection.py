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
from models import models_dir
from models.ae_unet import AEUnet


logging.basicConfig(level=logging.DEBUG)

def test(
    n, m, epochs, lr, opt, coarse, beta, gamma, eta, batch_size, N_train, N_test, fs, heart_rate, isnr,
    seed_training, processes, gpu, train_fraction, factor, 
    min_lr, min_delta, patience, delta, seed_ko, num_iter
):
    # ------------------ Constants ------------------
    corr_name = '96af96a7ddfcb2f6059092c250e18f2a'
    seed_train_data = 11
    seed_test_data = 66
    seed_training = 0 # seed for training data split
    loc = 0.25
    chunk_size = 100
    detector_type = 'AEUnet'
    detector_mode = 'self_assessment'
        # tiny
    # expanded_channels = 8
    # steps_num = 2

    # # small
    expanded_channels = 8
    steps_num = 3

    # large
    # expanded_channels = 64
    # steps_num = 4

    kernel_size = 3


    encoder_mode = 'orth' if coarse=='transpose' else 'default'
    decoder_mode_eval = 'pinv' if coarse == 'lstsq' else coarse

    # ------------------ Folders ------------------
    model_folder = f'{models_dir}AE'
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

    # ------------------ Autoencoder initialization ------------------
    ae = AEUnet(n, m, encoder_mode=encoder_mode, decode_mode=decoder_mode_eval, expanded_channels=expanded_channels, steps_num=steps_num)
    ae.to(device) # move the network to GPU
    model_name = f'exp_chn={expanded_channels}-stp_nm={steps_num}-krn_sz={kernel_size}-N={N_train}_n={n}_m={m}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                    f'_isnr={isnr}_beta={beta}_gamma={gamma}-eta={eta}-enc={encoder_mode}-coarse={coarse}'\
                    f'_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                    f'_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                    f'_mind={min_delta}_seeddata={seed_train_data}_seedtrain={seed_training}' 
    model_path = os.path.join(model_folder, f'AEUnet-{model_name}.pth')
    ae.load_state_dict(torch.load(model_path, weights_only=True))
    ae.eval()     # Set the model to evaluation mode

    if detector_mode == 'self_assessment':
        detector_mode = detector_mode + f'_{num_iter}'

    results_path = os.path.join(results_folder, 'AEUnet', detector_mode, f'{model_name}_delta={delta}_seedko={seed_ko}.pkl')

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if os.path.exists(results_path):
        print(f'\ndetector has been already evaluated')
        sys.exit(0)

    # ------------------ Define anomaly scores ------------------ 
    def iterative_self_assessment_score(X, num_iter=1):
        with torch.no_grad(): 
            Xhat_tensor = torch.from_numpy(X).float().to(device)
            Yhat_tensor = ae.encode(Xhat_tensor)
            Y = Yhat_tensor.cpu().numpy()
            for _ in range(num_iter): 
                Xhat_tensor = ae.decode(Yhat_tensor)
                Yhat_tensor = ae.encode(Xhat_tensor)
            Yhat = Yhat_tensor.cpu().numpy()
        return np.sqrt(np.sum((Y - Yhat)**2, axis=-1))

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
        scoreok = iterative_self_assessment_score(X, num_iter)
        scoreko = iterative_self_assessment_score(Xko, num_iter)
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
    parser.add_argument('--seed_training', '-s', type=int, required=True, help="Training-related random seed for reproducibility")
    parser.add_argument('--epochs', '-e', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=50, help="Batch size for training")
    parser.add_argument('-N', '--N_train', type=int, default=2000000, help="Number of training samples")
    parser.add_argument('-M', '--N_test', type=int, default=10000, help="Number of test samples")
    parser.add_argument('--optimizer', '-O', type=str, default='adam', help="Optimizer for training")
    parser.add_argument('--coarse', '-c', type=str, default='lstsq', help="Mode to compute the coarse reconstruction")
    parser.add_argument('--beta', '-w', type=float, default=0, help="Latent volume regularization weight")
    parser.add_argument('--gamma', '-q', type=float, default=0, help="Regularization of the latten space reconstruction")
    parser.add_argument('--eta', '-y', type=float, default=0, help="Regularization of the sensing matrix")
    parser.add_argument('--fs', '-f', type=int, default=256, help="Sampling frequency")
    parser.add_argument('--heart_rate', '-r', type=int, nargs=2, default=(60, 100), help="Heart rate range")
    parser.add_argument('--processes', '-p', type=int, default=48, help="Number of CPU processes")
    parser.add_argument('--gpu', '-g', type=int, default=3, help="GPU index to use for training")
    parser.add_argument('--train_fraction', '-T', type=float, default=0.9, help="Fraction of data used for training")
    parser.add_argument('--factor', '-F', type=float, default=0.2, help="Factor for ReduceLROnPlateau scheduler")
    parser.add_argument('--min_lr', '-L', type=float, default=0.001, help="Minimum learning rate")
    parser.add_argument('--min_delta', '-d', type=float, default=1e-4, help="Minimum delta for early stopping and ReduceLROnPlateau")
    parser.add_argument('--patience', '-P', type=int, default=40, help="Patience for early stopping")
    parser.add_argument('--delta', '-k', type=float, required=True, help="Anomaly intensity parameter")
    parser.add_argument('--seed_ko', '-z', type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument('--num_iter', '-j', type=int, default=0, help="Rounds of self-assessment")
    
    
    return parser.parse_args()

# ------------------ Main call ------------------
if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()
    
    # Call the training function with parsed arguments
    test(
        n=args.n,
        m=args.m,
        epochs=args.epochs,
        lr=args.lr,
        opt=args.optimizer,
        coarse=args.coarse,
        beta=args.beta,
        gamma=args.gamma,
        eta=args.eta,
        batch_size=args.batch_size,
        N_train=args.N_train,
        N_test=args.N_test,
        fs=args.fs,
        heart_rate=args.heart_rate,
        isnr=args.isnr,
        seed_training=args.seed_training,
        processes=args.processes,
        gpu=args.gpu,
        train_fraction=args.train_fraction,
        factor=args.factor,  
        min_lr=args.min_lr,
        min_delta=args.min_delta,
        patience=args.patience,
        delta=args.delta,
        seed_ko=args.seed_ko,
        num_iter = args.num_iter
    )



    
