import os
import pickle
import logging

import multiprocessing as mp

import numpy as np
import pandas as pd

from scipy import linalg
from numpy import random

from tqdm import tqdm
from itertools import product

from synthetic_ecg import generate_ecg
from wavelet_basis import wavelet_basis
from utils import compute_rsnr
from compressed_sensing import CompressedSensing, generate_sensing_matrix
from compressed_sensing import find_support_TSOC, find_support_TSOC2



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
isnr = 35               # signal-to-noise ratio in dB (35)
ecg_seed = 0            # random seed for ECG generation
basis = 'sym6'          # sparsity basis

# ---- support ----
method = 'TSOC2'         # {'TSOC', 'TSOC2'}

# ---- compressed sensing ----
# m_list = (16, 32, 48, 64)
m_list = (64, )
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
supports_path = lambda mode, m, orth, seed: os.path.join(folder, 
    f'supports_method={method}_mode={mode}_m={m}_orth={orth}_seed={seed}.pkl')


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
    (mode_list, m_list, orth_list, seed_list), 
    names=('mode', 'm', 'orth', 'seed'),
)

if os.path.exists(rsnr_path):
    logger.debug(f'loading RSNR from {rsnr_path}')
    df = pd.read_pickle(rsnr_path)
    todrop = [col for col in df.columns if col in columns]
    if len(todrop) > 0:
        columns = columns.drop(todrop)

if method == 'TSOC':
    find_support = find_support_TSOC
elif method == 'TSOC2':
    find_support = find_support_TSOC2
else:
    find_support = None

for i, (mode, m, orth, seed) in tqdm(list(enumerate(columns))):

    # Generate Sensing Matrix and set Compressed Sensing system
    logger.debug(f'generating Sensing Matrix ({m}, {n}), seed={seed}')
    A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orth, seed=seed)
    cs = CompressedSensing(A, D)

    _supports_path = supports_path(mode, m, orth, seed)
    if os.path.exists(_supports_path):
        with open(_supports_path, 'rb') as f:
            S = pickle.load(f)
    else:
        logger.debug(f'computing supports')
        if processes is None:
             desc = f'supports m={m}, seed={seed}'
             S = [find_support(x, cs) for x in tqdm(X, desc=desc)]
        else:
            args_list = product(X, [cs])
            with mp.Pool(processes=processes) as pool:
                S = pool.starmap(find_support, args_list, chunksize=10)
        S = np.stack(S)
        logger.debug(f'storing supports')
        with open(_supports_path, 'wb') as f:
              pickle.dump(S, f)
    
    # Compute measurements
    logger.debug(f'reconstructing signals')
    Y = cs.encode(X)

    # reconstruct signal
    logger.debug(f'computing RSNR')
    X_hat = cs.decode(Y, S)

    # compute RSNR
    _rsnr = compute_rsnr(X, X_hat)
    rsnr = pd.DataFrame(data=_rsnr, columns=columns[[i]])

    # store intermediate results
    if os.path.exists(rsnr_path):
        # re-load stored RSNR as it might have been updated by another 
        # process running in parallel
        logger.debug(f'loading RSNR from {rsnr_path}')
        df = pd.read_pickle(rsnr_path)
        rsnr = pd.concat((df, rsnr), axis=1).sort_index(axis=1)

    logger.debug(f'storing RSNR')
    rsnr.to_pickle(rsnr_path)

    
    