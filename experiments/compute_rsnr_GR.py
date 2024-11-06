import os
import pickle
import logging

import multiprocessing as mp

import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product

from synthetic_ecg import generate_ecg
from wavelet_basis import wavelet_basis
from utils import compute_rsnr
from compressed_sensing import CompressedSensing
from compressed_sensing import generate_sensing_matrix, find_support_GR



logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

################################################################################
# PARAMETERS                                                                   #
################################################################################

# ---- parallel processing ----
processes = None        # number of processes for multiprocessing

# ---- data ----
N = 10_000              # number of ECG traces
n = 128                 # length of an ECG trace
fs = 256                # sampling rate
heart_rate = (60, 100)  # min and max heart rate
isnr = 45               # signal-to-noise ratio in dB (35)
ecg_seed = 0            # random seed for ECG generation
basis = 'sym6'          # sparsity basis

# ---- support ----
method = 'GR'
eta_list = (
    0.9, 
    0.93,    0.95,    0.97,    0.98,    0.985,    0.99, 
    0.993,   0.995,   0.997,   0.998,   0.9985,   0.999, 
    0.9993,  0.9995,  0.9997,  0.9998,  0.99985,  0.9999, 
    0.99993, 0.99995, 0.99997, 0.99998, 0.999985, 0.99999,
)

# ---- compressed sensing ----
m_list = (16, 32, 48, 64)
mode_list = ('standard', )
orth_list = (True, )
seed_list = np.arange(20)

logger.info(f'n={n}, isnr={isnr}, method={method}')

################################################################################
# PATHs                                                                        #
################################################################################

data_folder = '/srv/newpenny/dnn-cs/tsoc/data/'
if not os.path.exists(data_folder):
    os.mkdir(data_folder)

data_name = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
            f'_isnr={isnr}_seed={ecg_seed}'
data_path = os.path.join(data_folder, data_name + '.pkl')

folder = os.path.join(data_folder, data_name)
if not os.path.exists(folder):
    os.mkdir(folder)

rsnr_path = os.path.join(folder, f'rsnr_method={method}.pkl')
supports_path = lambda eta: os.path.join(
    folder, f'supports_method={method}_eta={eta}.pkl')


################################################################################
# Load/Generate Data                                                           #
################################################################################

# loading/generate data
if os.path.exists(data_path):
    logger.debug(f'loading data from {data_path}')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)
else:
    logger.debug(f'generating data')
    X = generate_ecg(
        length=n, 
        num_traces=N,
        heart_rate=heart_rate, 
        sampling_rate=fs, 
        snr=isnr, 
        random_state=ecg_seed,
        verbose=True,
        processes=processes,
    )
    logger.debug(f'storing data in {data_path}')
    with open(data_path, 'wb') as f:
        pickle.dump(X, f)

D = wavelet_basis(n, 'sym6', level=2)


################################################################################
# Compute supports and RSNR                                                    #
################################################################################

columns = pd.MultiIndex.from_product(
    (mode_list, m_list, orth_list, seed_list, eta_list), 
    names=('mode', 'm', 'orth', 'seed', 'eta'),
)
if os.path.exists(rsnr_path):
    logger.debug(f'loading RSNR from {rsnr_path}')
    df = pd.read_pickle(rsnr_path)
    todrop = [col for col in df.columns if col in columns]
    if len(todrop) > 0:
        columns = columns.drop(todrop)
else:
    df = None
rsnr = pd.DataFrame(columns=columns)


for mode, m, orth, seed in columns.droplevel('eta').unique():

    # Generate Sensing Matrix and set Compressed Sensing system
    logger.debug(f'generating Sensing Matrix ({m}, {n}), seed={seed}')
    A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orth, seed=seed)
    cs = CompressedSensing(A, D)

    # Compute measurements
    logger.debug(f'encoding data')
    Y = cs.encode(X)

    _desc = f'm={m}, seed={seed}'
    for eta in tqdm(rsnr.columns.unique(level='eta'), desc=_desc):

        # load/compute supports
        if os.path.exists(supports_path(eta)):
            logger.debug(f'loading supports')
            with open(supports_path(eta), 'rb') as f:
                S = pickle.load(f)
        else:
            logger.debug(f'computing supports')
            if processes is None:
                desc = f'supports m={m}, seed={seed}, eta={eta}'
                S = [find_support_GR(x, D, eta=eta) for x in tqdm(X, desc=desc)]
            else:
                args_list = product(X, [D], [eta])
                with mp.Pool(processes=processes) as pool:
                    S = pool.starmap(find_support_GR, args_list, chunksize=20)
                S = np.stack(S)
            S = np.stack(S)
            logger.debug(f'storing supports')
            with open(supports_path(eta), 'wb') as f:
                pickle.dump(S, f)

        # reconstruct signal
        logger.debug(f'reconstructing signals')
        X_hat = cs.decode(Y, s=S, processes=processes)

        # compute RSNR
        logger.debug(f'computing RSNR')
        rsnr[(mode, m, orth, seed, eta)] = compute_rsnr(X, X_hat)

    # store intermediate results
    logger.debug(f'storing RSNR')
    tmp = pd.concat((df, rsnr.dropna(axis=1)), axis=1).sort_index(axis=1)
    tmp.to_pickle(rsnr_path)
    