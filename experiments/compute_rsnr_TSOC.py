import os
import sys
import pickle
import logging
import argparse

import multiprocessing as mp

import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product

# import of local modules
root = os.path.dirname(os.path.realpath('__file__'))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir
from dataset.synthetic_ecg import generate_ecg
from cs.wavelet_basis import wavelet_basis
from cs import CompressedSensing, generate_sensing_matrix
from cs.utils import compute_rsnr



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
# isnr = 35               # signal-to-noise ratio in dB (35)
ecg_seed = 0            # random seed for ECG generation
basis = 'sym6'          # sparsity basis

# # ---- support ----
# method = 'TSOC'         # {'TSOC', 'TSOC2'}

# # ---- compressed sensing ----
# m_list = (16, 32, 48, 64)
# mode = 'rakeness' # standard, rakeness
# orth = True
# seed_list = np.arange(20)

# # ---- rakeness ----
# loc = .25
# corr_name = '96af96a7ddfcb2f6059092c250e18f2a'


def cmdline_args():
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-m", "--measurements", type=int, nargs='+', 
        help="number of measurements",
    )
    parser.add_argument(
        "-s", "--seed", type=int, nargs='+',
        help="random seed"
    )
    parser.add_argument(
        "-i", "--isnr", type=int, default=None,
        help="intrinsic signal-to-noise ration (ISNR) (default: %(default)s)"
    )
    parser.add_argument(
        "-a", "--algorithm", choices=('TSOC', 'TSOC2'), default='TSOC',
        help="support identification algorithm (default: %(default)s)",
    )
    parser.add_argument(
        "-e", "--encoder", choices=('standard', 'rakeness'), default='standard',
        help="compressed sensing encoding mode (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--orthogonal", action='store_true',
        help="weather sensing matrix is orthogonal",
    )
    parser.add_argument(
        "-c", "--correlation", default='aa851c0819ff9aaacf4099aa5b84696d',
        help="Correlation name (default: %(default)s)",
    )
    parser.add_argument(
        "-l", "--localization", type=float, default=.25,
        help="rakeness localization factor (default: %(default)s)",
    )
    parser.add_argument(
        "-p", "--processes", type=int,
        help="number of parallell processes",
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="increase output verbosity (default: %(default)s)",
    )
    
    return parser.parse_args()


def main(m_list, seed_list, isnr, method, mode, orth, corr, loc, seed, processes):
    ############################################################################
    # PATHs                                                                    #
    ############################################################################

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    data_name = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={ecg_seed}'
    data_path = os.path.join(dataset_dir, data_name + '.pkl')

    folder = os.path.join(dataset_dir, data_name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    rsnr_path = os.path.join(folder, f'rsnr_method={method}.pkl')

    if mode == 'standard':
        supports_path = lambda m, seed: os.path.join(
            folder, f'supports_method={method}_mode={mode}_m={m}_orth={orth}'\
                    f'_seed={seed}.pkl')
        
    elif mode == 'rakeness':
        supports_path = lambda m, seed: os.path.join(
            folder, f'supports_method={method}_mode={mode}_m={m}'\
                    f'_corr={corr}_loc={loc}_orth={orth}_seed={seed}.pkl')


    ############################################################################
    # Load Data                                                                #
    ############################################################################

    # loading data
    if not os.path.exists(data_path):
        raise RuntimeError(f'dataset {data_path} not available')

    logger.debug(f'loading data from {data_path}')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)

    # generate sparsity basis
    D = wavelet_basis(n, 'sym6', level=2)

    # load ECG correlation matrix
    C = None  # if mode is not rakeness, corr is set to None
    if mode == 'rakeness':
        corr_path = os.path.join(dataset_dir, 'correlation', corr + '.pkl')
        if not os.path.exists(corr_path):
            raise RuntimeError(f'correlation {corr} not available')
        with open(corr_path, 'rb') as f:
            C = pickle.load(f)


    ############################################################################
    # Compute supports and RSNR                                                #
    ############################################################################

    columns = pd.MultiIndex.from_product(
        ([mode], m_list, [orth], seed_list), 
        names=('mode', 'm', 'orth', 'seed'),
    )

    if os.path.exists(rsnr_path):
        logger.debug(f'loading RSNR from {rsnr_path}')
        df = pd.read_pickle(rsnr_path)
        todrop = [col for col in df.columns if col in columns]
        if len(todrop) > 0:
            columns = columns.drop(todrop)

    for i, (_, m, _, seed) in tqdm(list(enumerate(columns))):

        # Generate Sensing Matrix and set Compressed Sensing system
        logger.debug(f'generating sensing matrix ({m}, {n}), seed={seed}')
        A = generate_sensing_matrix(
            (m, n), mode=mode, orthogonal=orth, 
            correlation=C, loc=loc, seed=seed)
        cs = CompressedSensing(A, D)

        _supports_path = supports_path(m, seed)
        if not os.path.exists(_supports_path):
            raise RuntimeError(f'supports {_supports_path} not available')
        logger.debug(f'loading supports {_supports_path}')
        with open(_supports_path, 'rb') as f:
            S = pickle.load(f)
        
        # Compute measurements
        logger.debug(f'encoding signals')
        Y = cs.encode(X)

        # reconstruct signal
        logger.debug(f'reconstructing signals')
        X_hat = cs.decode(Y, S)

        # compute RSNR
        logger.debug(f'computing RSNR')
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


if __name__ == '__main__':

    args = cmdline_args()

    if args.verbose == 0:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose > 1:
        logger.setLevel(logging.DEBUG)

    logger.debug(str(args))

    main(
        args.measurements,
        args.seed,
        args.isnr,
        args.algorithm,
        args.encoder,
        args.orthogonal,
        args.correlation,
        args.localization,
        args.seed,
        args.processes,
    )
    
    