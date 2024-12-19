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
        "-a", "--algorithm", choices=('GR', 'TSOC', 'TSOC2'), default='TSOC',
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
        "-c", "--correlation",
        help="Correlation name",
    )
    parser.add_argument(
        "-l", "--localization", type=float, default=.25,
        help="rakeness localization factor (default: %(default)s)",
    )
    parser.add_argument(
        "--eta", type=float, nargs='+', 
        help="energy fraction the support must retain (GR algorithm only)",
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


def main(
    m_list, 
    seed_list, 
    isnr, 
    method, 
    mode, 
    orth, 
    corr, 
    loc, 
    eta_list, 
    processes
):
    ############################################################################
    # PATHs                                                                    #
    ############################################################################

    # dataset
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    data_name = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={ecg_seed}'
    data_path = os.path.join(dataset_dir, data_name + '.pkl')
    data_dir = os.path.join(dataset_dir, data_name)

    # RSNR
    rsnr_path = os.path.join(data_dir, f'rsnr_method={method}_mode={mode}.pkl')

    # supports
    if method == 'GR':
        _name = lambda eta: f'supports_method={method}_eta={eta}.pkl'
        supports_path = lambda eta: os.path.join(data_dir, _name(eta))

    elif method in ('TSOC', 'TSOC2'):
        if mode == 'standard':
            _name = lambda m, seed: \
                f'supports_method={method}_mode={mode}_m={m}_orth={orth}'\
                f'_seed={seed}.pkl'
        elif mode == 'rakeness':
            _name = lambda m, seed: \
                f'supports_method={method}_mode={mode}_m={m}_corr={corr}'\
                f'_loc={loc}_orth={orth}_seed={seed}.pkl'
        supports_path = lambda m, seed: os.path.join(data_dir, _name(m, seed))


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
    if mode == 'rakeness':
        corr_path = os.path.join(dataset_dir, 'correlation', corr + '.pkl')
        if not os.path.exists(corr_path):
            raise RuntimeError(f'correlation {corr} not available')
        with open(corr_path, 'rb') as f:
            C = pickle.load(f)


    ############################################################################
    # Compute supports and RSNR                                                #
    ############################################################################

    # build columns (pandas.MultiIndex) that contains all configurations for
    # which we compute RSNR. Each multiindex level corresponds to a parameter.
    columns = {'m': m_list, 'orth': [orth], 'seed': seed_list, }
    if mode == 'rakeness': columns = {**columns, 'corr': [corr], 'loc': [loc]}
    if method == 'GR': columns = {**columns, 'eta': eta_list}
    columns = pd.MultiIndex.from_product(columns.values(), names=columns.keys())

    # load already computed RSNR and remove those configurations for which
    # RSNR has been already computed
    if os.path.exists(rsnr_path):
        logger.debug(f'loading RSNR from {rsnr_path}')
        df = pd.read_pickle(rsnr_path)
        todrop = [col for col in df.columns if col in columns]
        if len(todrop) > 0:
            columns = columns.drop(todrop)

    # RSNR computation differs depending on the supports
    if method == 'GR':

        # iterate over m and seed. 
        # eta does not depend on compression therefore all different eta are 
        # considered in an internal loop after setting up compression
        cols = columns.droplevel('eta').unique()
        for i, _col in enumerate(cols):
            m = _col[cols.names.index('m')]
            seed = _col[cols.names.index('seed')]

            # generate sensing matrix and set up Compressed Sensing
            logger.debug(f'generating sensing matrix ({m}, {n}), seed={seed}')
            kwargs = {'mode': mode, 'orthogonal': orth, 'seed': seed}
            if mode == 'rakeness':
                kwargs = {**kwargs, 'correlation': C, 'loc': loc}
            A = generate_sensing_matrix((m, n), **kwargs)
            cs = CompressedSensing(A, D, decoder='SO', processes=processes)

            # Compute measurements
            logger.debug(f'encoding data')
            Y = cs.encode(X)

            rsnr = []  # store rsnr columns to be concatenated
            # iterate over etas corresponding to selected m and seed
            eta_cols = columns.get_loc_level((m, seed), level=('m', 'seed'))[1]
            desc = f'{m}, {seed}'
            for eta in tqdm(eta_cols.get_level_values('eta'), desc=desc):

                # load supports
                if not os.path.exists(supports_path(eta)):
                    logger.warning(f'{supports_path(eta)} not available')
                    continue
                logger.debug(f'loading supports')
                with open(supports_path(eta), 'rb') as f:
                    S = pickle.load(f)

                # reconstruct signal
                logger.debug(f'reconstructing signals')
                X_hat = cs.decode(list(zip(Y, S)))

                # compute RSNR
                logger.debug(f'computing RSNR')
                _rsnr = compute_rsnr(X, X_hat)
                col = pd.MultiIndex.from_tuples(
                    tuples=[(*_col, eta)], names=(*cols.names, 'eta'))
                rsnr.append(pd.DataFrame(data=_rsnr, columns=col))
            
            rsnr = pd.concat(rsnr, axis=1)
            
            # store intermediate results
            if os.path.exists(rsnr_path):
                # re-load stored RSNR as it might have been updated by another 
                # process running in parallel
                logger.debug(f'loading RSNR from {rsnr_path}')
                df = pd.read_pickle(rsnr_path)
                rsnr = pd.concat((df, rsnr), axis=1).sort_index(axis=1)
                rsnr = rsnr.loc[:,~rsnr.columns.duplicated()].copy()

            # store intermediate results
            logger.debug(f'storing RSNR')
            rsnr.to_pickle(rsnr_path)
    
    elif method in ('TSOC', 'TSOC2'):

        # iterate over all configurations
        for i, _col in tqdm(list(enumerate(columns))):
            m = _col[columns.names.index('m')]
            seed = _col[columns.names.index('seed')]

            # generate sensing matrix and set up Compressed Sensing
            logger.debug(f'generating sensing matrix ({m}, {n}), seed={seed}')
            kwargs = {'mode': mode, 'orthogonal': orth, 'seed': seed}
            if mode == 'rakeness':
                kwargs = {**kwargs, 'correlation': C, 'loc': loc}
            A = generate_sensing_matrix((m, n), **kwargs)
            cs = CompressedSensing(A, D, decoder='SO', processes=processes)

            # load supports
            _path = supports_path(m, seed)
            if not os.path.exists(_path):
                logger.warning(f'supports {_path} not available')
                continue
            logger.debug(f'loading supports {_path}')
            with open(_path, 'rb') as f:
                S = pickle.load(f)
            
            # Compute measurements
            logger.debug(f'encoding signals')
            Y = cs.encode(X)

            # reconstruct signal
            logger.debug(f'reconstructing signals')
            X_hat = cs.decode(list(zip(Y, S)))

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
                rsnr = rsnr.loc[:,~rsnr.columns.duplicated()].copy()

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
        args.eta,
        args.processes,
    )
    
    