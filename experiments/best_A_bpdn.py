import os
import numpy as np
from spgl1 import spg_bpdn
from tqdm import tqdm
import pickle as pkl


import multiprocessing as mp
import itertools as it

import logging
import argparse


from matplotlib import pyplot as plt


import sys ; sys.path.append('..')  # useful if you're running locally

# import of local modules
root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset import dataset_dir

from cs.wavelet_basis import wavelet_basis
from cs.base import generate_sensing_matrix
from cs.utils import compute_rsnr


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

os.environ['SCIPY_USE_PROPACK'] = "True"

threads = "64"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads



logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

###### STANDARD PARAMETERS ######

corr_str = '96af96a7ddfcb2f6059092c250e18f2a' # name of corr matrix (file)
N_ecg = 10_000 # number of ecg samples in dataset
fs = 256 # sampling freq
hr = (60, 100) # heart rate
isnr = 35 # dB (intrinsic snr)
orth = True # orthonormal rows of A
loc = 0.25 # localization for rakeness

###### IMPORTANT PARAMETERS #####

n_default = 128 # length of ecg
m_default = 48 # length of compressed ecg
mode_A_default = 'standard' # A: standard or rakeness
N_try_A_default = 1_000 # how many A to try
N_keep_A_default = 8 # best As to save

seed_A_default = 0
seed_ecg_default = 0


##### TOP FUNCTIONS #####

def enc_dec_bpdn(A,D,B,x):
    y = A@x
    rsnr = compute_rsnr(x,D@spg_bpdn(B,y,10**-4)[0])
    return rsnr 

def cmdline_args():
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-n", "--ecglength",  type=int, default=n_default,
        help="number of samples in each ecg example (default: %(default)s)"
    )
    parser.add_argument(
        "-m", "--compressedlength",  type=int, default=m_default,
        help="number of sample in each COMPRESSED ECG example (default: %(default)s)"
    )
    parser.add_argument(
        '--modeA', type=str, default=mode_A_default,
        help="whether A is normal or rakeness (default: %(default)s)"
    )
    parser.add_argument(
        "--Ntry",  type=int, default=N_try_A_default,
        help="number of A composing the dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--Nkeep", type=int, default=N_keep_A_default,
        help="number of A to save (the first Nkeep best) (default: %(default)s)"
    )
    parser.add_argument(
        "--seedA", type=int, default=seed_A_default,
        help="seed for the generation of A (default: %(default)s)"
    )
    parser.add_argument(
        "--seedecg", type=int, default=seed_ecg_default,
        help="seed for the generation of ecg dataset (default: %(default)s)"
    )

    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="increase output verbosity (default: %(default)s)"
    )
    
    return parser.parse_args()


def main(n, m, mode_A, N_try_A, N_keep_A, seed_A, seed_ecg):

    str_ecg_setting = f'ecg_N={N_ecg}_n={n}_fs={fs}_hr={hr[0]}-{hr[1]}_isnr={isnr}_seed={seed_ecg}'
    path_ecg = os.path.join(dataset_dir, str_ecg_setting + '.pkl')

    str_A_setting = f'sensing_matrix_M={N_try_A}_m={m}_mode={mode_A}_seed={seed_A}'
    if mode_A=='rakeness':
        str_A_setting = str_A_setting + f'_loc={loc}_corr={corr_str}'

    save_A_folder = os.path.join(dataset_dir, str_ecg_setting, 'A_Filippo')
    save_A_dest = os.path.join(save_A_folder, str_A_setting+'.pkl')
    print(N_ecg)
    print(corr_str)
    # standard correlation matrix computed from ecg dataset with isnr=0dB
    path_corr = os.path.join(dataset_dir, 'correlation', corr_str + '.pkl')

    with open(path_ecg, 'rb') as f:
        ecg = pkl.load(f)[:, np.newaxis]
    with open(path_corr, 'rb') as f:
        corr = pkl.load(f)

    D = wavelet_basis(n, 'sym6', level=2)

    keep_A = {}

    pool = mp.Pool(processes=20)


    best_rsnr = np.array([-1000 - i for i in range(N_keep_A)], dtype=np.float64)
    seed_A_best = np.empty((N_keep_A), dtype=int)
    keep_A = np.empty((N_keep_A, m, n), dtype=np.float64)

    
    rng = np.random.default_rng(seed_A)
    rn_seeds = rng.choice(100000, size=N_try_A, replace=False)
    
    for i, seed_trial in enumerate(rn_seeds):
        A_trial = generate_sensing_matrix(
                        shape = (m, n), 
                        mode=mode_A, 
                        antipodal=False, 
                        orthogonal=orth,
                        correlation=corr,
                        loc=loc, 
                        seed=seed_trial)
        
        B = A_trial @ D

        rsnr = pool.starmap(enc_dec_bpdn, 
                            it.product([A_trial],[D],[B],ecg[:, 0, :]))
                    
        rsnr = np.mean(rsnr)
        if rsnr>np.min(best_rsnr):
            ind_best = best_rsnr==np.min(best_rsnr)
            best_rsnr[ind_best] = rsnr
            seed_A_best[ind_best] = seed_trial
            keep_A[ind_best] = A_trial


        print(f'[{i}/{N_try_A-1}] rsnr={np.round(rsnr, 8)} --- best: {[np.round(i, 2) for i in best_rsnr]}')
    
    pool.close()
    pool.join() 

    srt = np.argsort(best_rsnr)
    best_rsnr, seed_A_best, keep_A = best_rsnr[srt][::-1], seed_A_best[srt][::-1], keep_A[srt][::-1]
    
    list_save = []
    for A, rsnr, seed in zip(keep_A, best_rsnr, seed_A_best):

        list_save += [{'rsnr': rsnr,
                      'seed': seed,
                      'matrix': A}]


     

    if not os.path.isdir(save_A_folder):
        os.mkdir(save_A_folder)

    with open(save_A_dest, 'wb') as f:
        pkl.dump(list_save, f)

    print(f'best {N_keep_A} (over {N_try_A}) have been saved for m = {m} in',
          f'{save_A_dest}')

    return

    



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
        args.ecglength, 
        args.compressedlength,
        args.modeA,
        args.Ntry,
        args.Nkeep,
        args.seedA,
        args.seedecg,
    )