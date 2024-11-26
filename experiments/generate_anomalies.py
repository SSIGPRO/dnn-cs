
import os
import sys
import numpy as np
import pickle
import argparse
import tqdm

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

# import of local modules
# root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
# sys.path.insert(0, os.path.join(root, 'src'))

# import of local modules
root = os.path.dirname(os.path.realpath('__file__'))
sys.path.insert(0, os.path.join(root, 'src'))

from dataset.synthetic_ecg import generate_ecg
from dataset import dataset_dir

def generate(N, n, fs, heart_rate, isnr, seed_ok, seed_ko, delta, processes):

    # ------------------ Loard normal ECG data ------------------
    ok_name = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                    f'_isnr={isnr}_seed={seed_ok}.pkl'
    ok_path = os.path.join(dataset_dir, ok_name)
    with open(ok_path, 'rb') as f:
        Xok = pickle.load(f)

    # standarize the data
    std, mean = Xok.std(), Xok.mean()
    Xok = (Xok - mean) / std
    print(std, mean)

    # ------------------ Seeds ------------------
    np.random.seed(seed_ko)

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
    
    # generate anomalous data for each anomaly
    print("Generation KO ECG data...")
    for anomaly_label, anomaly in tqdm.tqdm(anomalies_dict.items()):
        ko_name = f'ecg_anomaly_{anomaly_label}_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                    f'_isnr={isnr}_delta={delta}_seedok={seed_ok}_seedko={seed_ko}.pkl'
        ok_folder = f'ecg_N={N}_n={n}_fs={fs}_hr={heart_rate[0]}-{heart_rate[1]}'\
                    f'_isnr={isnr}_seed={seed_ok}'

        ko_path = os.path.join(dataset_dir, ok_folder, 'anomalies', f'delta={delta}', f'seed={seed_ko}', ko_name)
        # ensure the directories exists
        os.makedirs(os.path.dirname(ko_path), exist_ok=True)

        # check if the file already exists
        if os.path.exists(ko_path):
            print(f"{anomaly_label} anomaly already exists.")
            continue  # skip to the next anomaly label
        
        if anomaly_label in ['SpectralAlteration']:
            anomaly.fit(Xok, isnr)
        else:
            anomaly.fit(Xok)
        Xko = anomaly.distort(Xok)
        deltahat = np.mean( (Xok - Xko)**2 )
        print(
            f"\t{anomaly_label} delta={np.round(deltahat, 4)}"
            )
        
        # rescale back the data
        Xko = Xko*std + mean

        # save data        
        with open(ko_path, 'wb') as f:
            pickle.dump(Xko, f)


def parse_args():

    parser = argparse.ArgumentParser(description="Generate Anomalous data for ECG signal")
    
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
        "--seed_ok", type=int, help="random seed for normal data"
    )
    parser.add_argument(
        "--seed_ko", type=int, help="random seed for anomalous data"
    )
    parser.add_argument(
        "-p", "--processes", type=int,
        help="number of parallell processes"
    )
    parser.add_argument(
        "-d", "--delta",  type=float, default=0.1,
        help="Intensity of the anomalies (default: %(default)s)"
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="increase output verbosity (default: %(default)s)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    generate(
        args.size, 
        args.length, 
        args.sample_freq, 
        args.heart_rate,
        args.isnr,
        args.seed_ok,
        args.seed_ko,
        args.delta,
        args.processes,
    )