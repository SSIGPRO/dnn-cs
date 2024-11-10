"""
Compute ECG correlation matrix
"""


import os
import sys
import pickle
import logging

import argparse
import hashlib


# import of local modules
root = os.path.dirname(os.path.realpath('__file__'))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset.synthetic_ecg import generate_ecg
from dataset import dataset_dir


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

def cmdline_args():
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-s", "--size",  type=int, default=10_000,
        help="number of ECG examples (default: %(default)s)"
    )
    parser.add_argument(
        "-l", "--length",  type=int, default=256,
        help="number of sample in each ECG example (default: %(default)s)"
    )
    parser.add_argument(
        "-f", "--sample-freq",  type=int, default=256,
        help="sample frequency (default: %(default)s)"
    )
    parser.add_argument(
        "-r", "--heart-rate", type=int, nargs=2, default=(60, 100),
        help="hearte rate range (2 arguments, default: %(default)s)"
    )
    parser.add_argument(
        "-i", "--isnr", type=int, default=None,
        help="intrinsic signal-to-noise ration (ISNR) (default: %(default)s)"
    )
    parser.add_argument(
        "--seed", type=int,
        help="random seed"
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="increase output verbosity (default: %(default)s)"
    )
    
    return parser.parse_args()


def main(N, n, fs, heart_rate, isnr, seed):

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    data_name = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                f'_isnr={isnr}_seed={seed}'
    data_path = os.path.join(dataset_dir, data_name + '.pkl')

    corr_dir = os.path.join(dataset_dir, 'correlation')
    if not os.path.exists(corr_dir):
        os.mkdir(corr_dir)
    
    corr_name = hashlib.md5(data_name.encode()).hexdigest()
    corr_path = os.path.join(corr_dir, corr_name  + '.pkl')
    if os.path.exists(corr_path):
        logger.info(f'correlation {corr_name} already exists')
        return

    data_path = os.path.join(dataset_dir, data_name + '.pkl')
    if not os.path.exists(data_path):
        raise RuntimeError(f'dataset {data_name} not available')

    logger.debug(f'loading dataset {data_name}')
    with open(data_path, 'rb') as f:
        X = pickle.load(f)

    logger.debug(f'computing correlation')
    corr = (X.T @ X) / len(X)

    logger.debug(f'storing correlation {corr_name}')
    with open(corr_path, 'wb') as f:
        pickle.dump(corr, f)

    

if __name__ == '__main__':

    args = cmdline_args()

    if args.verbose == 0:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose > 1:
        logger.setLevel(logging.DEBUG)

    print(args)
    logger.debug(str(args))

    main(
        args.size, 
        args.length, 
        args.sample_freq, 
        args.heart_rate,
        args.isnr,
        args.seed,
    )