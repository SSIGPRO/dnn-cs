"""
Generate ECG data.
"""


import os
import sys
import pickle
import logging

import argparse

# limit number of parallel threads numpy spawns
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import numpy as np

from tqdm import tqdm
from itertools import product

import multiprocessing as mp


# import of local modules
root = os.path.dirname(os.path.realpath('__file__'))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir
from cs import CompressedSensing, generate_sensing_matrix
from cs.wavelet_basis import wavelet_basis
from cs.supports import find_support_GR, find_support_TSOC, find_support_TSOC2


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# ---- data ----
n = 128                 # length of an ECG trace
fs = 256                # sampling rate
heart_rate = (60, 100)  # min and max heart rate
basis = 'sym6'          # sparsity basis
seed_data_matrix = 0    # seed used to generate data for sensing matrix selection
seed_selection = 0      # seed used to select the best sensing matrix on data geneerated with seed_data_matrix
M = 1_000               # number of evaluated sensing matrices to choose the 8 best 


def cmdline_args():
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-s", "--size",  type=int, default=10000,
        help="number of ECG examples (default: %(default)s)"
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
        '--source', type=str, choices=('best', 'random'), default='random',
        help="Measurement matrix type: genereated randomly or leading to best performance (default: %(default)s)",
        )
    parser.add_argument(
        "-o", "--orthogonal", action='store_true',
        help="weather sensing matrix is orthogonal",
    )
    parser.add_argument(
        "-m", "--measurements",  type=int, default=64,
        help="number of measurements (default: %(default)s)",
    )
    parser.add_argument(
        "-c", "--correlation", default='aa851c0819ff9aaacf4099aa5b84696d',
        help="Correlation name (rakeness mode only, default: %(default)s)",
    )
    parser.add_argument(
        "-l", "--localization", type=float, default=.25,
        help="rakeness localization factor (rakeness mode only, default: %(default)s)",
    )
    parser.add_argument(
        "--ecg_seed", type=int, default=0,
        help="Data random seed, default: %(default)s"
    )
    parser.add_argument(
        "--seed", type=int,
        help="either random seed or index in case of random or best sensing matrix respectively"
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


def main(N, isnr, method, mode, source, orth, m, corr, loc, ecg_seed, seed, eta_list, processes):
    
    ############################################################################
    # PATHs                                                                    #
    ############################################################################

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    data_name = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={ecg_seed}'
    data_path = os.path.join(dataset_dir, data_name + '.pkl')
    if not os.path.exists(data_path):
        raise RuntimeError(f'dataset {data_name} not available')

    supports_dir = os.path.join(dataset_dir, data_name)
    if not os.path.exists(supports_dir):
        os.mkdir(supports_dir)

    if method == 'GR':
        supports_path = lambda eta: os.path.join(
            dataset_dir, f'supports_method={method}_eta={eta}.pkl')
        
    if method in ('TSOC', 'TSOC2'):
        if mode == 'standard':
            supports_name_ = f'supports_method={method}_mode={mode}_m={m}'\
                f'_orth={orth}'
            
        elif mode == 'rakeness':
            supports_name_ = f'supports_method={method}_mode={mode}_m={m}'\
                f'_corr={corr}_loc={loc}_orth={orth}'
            
            corr_path = os.path.join(dataset_dir, 'correlation', corr + '.pkl')
            if not os.path.exists(corr_path):
                raise RuntimeError(f'correlation {corr} not available')

        supports_path = lambda seed: os.path.join(
            supports_dir, f'{supports_name_}_seed={seed}.pkl')
    
    ############################################################################
    # Load Data                                                                #
    ############################################################################
    logger.debug(f'loading dataset {data_name}')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)
        
    # generate Sparsity Basis
    D = wavelet_basis(n, 'sym6', level=2)

    ############################################################################
    # Compute supports                                                         #
    ############################################################################
    if method == 'GR':
        
        logger.debug(f'computing supports')
        for eta in tqdm(eta_list):
            if os.path.exists(supports_path(eta)):
                continue
            
            if processes is None:
                desc = f'{m}, {eta}'
                S = [find_support_GR(x, D, eta=eta) for x in tqdm(X, desc=desc)]
            else:
                args_list = product(X, [D], [eta])
                with mp.Pool(processes=processes) as pool:
                    S = pool.starmap(find_support_GR, args_list, chunksize=20)
            S = np.stack(S)

            logger.debug(f'storing supports eta={eta}')
            with open(supports_path(eta), 'wb') as f:
                    pickle.dump(S, f)

    elif method in ('TSOC', 'TSOC2'):

        # Generate Sensing Matrix and set Compressed Sensing system
        
        # Sensing matrix
        if source == 'random':
            # Generate a random sensing matrix
            if mode == 'rakeness':
                logger.debug(f'loading correlation {corr}')
                with open(corr_path, 'rb') as f:
                    C = pickle.load(f)
            else:
                C = None
            logger.debug(f'generating sensing matrix ({m}, {n}), seed={seed}')
            A = generate_sensing_matrix((m, n), mode=mode, orthogonal=orth, correlation=C, loc=.25, seed=seed)
        elif source == 'best':
            # Load the best sensing matrix

            A_folder = f'ecg_N=10000_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}_isnr={isnr}_seed={seed_data_matrix}'
            A_name = f'sensing_matrix_M={M}_m={m}_mode={mode}_seed={seed_selection}'
            if mode == 'rakeness':
                A_name = f'{A_name}_loc={.25}_corr={corr}'
            data_path = os.path.join(dataset_dir, A_folder, 'A_Filippo', f'{A_name}.pkl')
            logger.debug(f'loading {seed+1}-th best sensing matrix')
            with open(data_path, 'rb') as f:
                A_dict = pickle.load(f)
            A = A_dict[seed]['matrix']
            seed = A_dict[seed]['seed']
            logger.debug(f'sensing matrix ({m}, {n}) with seed={seed} loaded')

        if os.path.exists(supports_path(seed)):
            logger.info(f'supports {supports_path(seed)} already exists')
            return
        
        cs = CompressedSensing(A, D=D, decoder='SO')
        
        if method == 'TSOC':
            find_support = find_support_TSOC
        elif method == 'TSOC2':
            find_support = find_support_TSOC2

        logger.debug(f'computing supports')
        if processes is None:
            S = [find_support(x, cs) for x in tqdm(X, desc=f'{m}, {seed}')]
        else:
            with mp.Pool(processes=processes) as pool:
                S = pool.starmap(find_support, product(X, [cs]), chunksize=50)
        S = np.stack(S)

        logger.debug(f'storing supports')
        with open(supports_path(seed), 'wb') as f:
                pickle.dump(S, f)


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
        args.size,
        args.isnr,
        args.algorithm,
        args.encoder,
        args.source,
        args.orthogonal,
        args.measurements,
        args.correlation,
        args.localization,
        args.ecg_seed,
        args.seed,
        args.eta,
        args.processes,
    )