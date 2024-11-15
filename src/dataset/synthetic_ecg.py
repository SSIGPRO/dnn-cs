import numpy as np
from numpy import random
from scipy import io

import neurokit2 as nk

import multiprocessing as mp

from tqdm import tqdm
from .utils import add_noise


def load_synthetic_ecg(num_samples=None, sample_dimension=None, seed=None):

    rng = np.random.default_rng(seed=seed)

    path1 = '/srv/newpenny/dataset/ECG/synthetic/ecgSyn_n512_Ny256_(HR 80 100).mat'
    path2 = '/srv/newpenny/dataset/ECG/synthetic/ecgSynB_n512_Ny256_(HR 60 80).mat'

    data1 = io.loadmat(path1)['sigs'].T
    data2 = io.loadmat(path2)['sigs'].T

    data = np.concatenate((data1, data2))

    rng.shuffle(data)


    if num_samples is None or num_samples > len(data):
        num_samples = len(data)
    if sample_dimension is None or sample_dimension > data.shape[1]:
        sample_dimension = data.shape[1]

    if num_samples < len(data):
        data = data[:num_samples]

    if sample_dimension < data.shape[1]:
        X = np.empty((num_samples, sample_dimension), dtype=float)
        for i in range(num_samples):
            j = rng.integers(data.shape[1] - sample_dimension)
            X[i, :] = data[i, j:j+sample_dimension]
    else:
        X = data

    return X

def generate_ecg(
    length, 
    num_traces=1,
    heart_rate=70, 
    sampling_rate=256, 
    snr=None, 
    random_state=None,
    verbose=False,
    processes=None,
):
    '''
    Generate ECG Dataset

    Generate a synthetic ECG dataset as a set of traces with a given length,
    sampling rate using ECGSYN dynamical model (McSharry et al., 2003).

    Parameters
    ----------
    length: int
        Desired traces length (in samples).
    num_traces: int, optional (default 1)
        number of traces.
    heart_rate: float or (float, float), optional (default 70)
        Desired simulated heart rate (in beats per minute). If tuple, the heart 
        rate of each trace is a random value uniformely distributed in the range
        define by the two values in the tuple.
    sampling_rate: int, optional (default 256)
        The desired sampling rate (samples/second).
    snr: None or float, optional (default None)
        desired Signal-to-Noise Ratio. If None, no noise is injected.
    noise : float
        Noise level (amplitude of the laplace noise).
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator.
    verbose: bool, optional (default False)
        Enable progress bar when multiprocessing is disabled.
    processes: int or None, optional (defaul None)
        Number of processes for multiprocessing. 
        If None, Multiprocessing is disabled.

    Returns
    -------
    1D or 2D numpy.ndarray
        vector containing a ECG trace or matrix containing ECG traces.
    '''
    rng = random.default_rng(random_state)

    seed_list = rng.integers(2**32, size=num_traces)
    if hasattr(heart_rate, "__len__"):
        hr_list = rng.uniform(*heart_rate, size=num_traces)
    else:
        hr_list = heart_rate * np.ones(num_traces)
    period_list = np.round(60/hr_list * sampling_rate).astype(int)
    shift_list = np.array([rng.integers(period) for period in period_list])

    args_list = zip(hr_list, shift_list, seed_list)

    if processes is None: 
        # single thread processing
        X = np.empty((num_traces, length), dtype=float)
        tqdm_kwargs = {'total': num_traces, 'disable': not verbose}
        for i, (hr, shift, seed) in tqdm(enumerate(args_list), **tqdm_kwargs):
            X[i, :] = _ecg_generate_trace(length, sampling_rate, hr, shift, seed)
    else:
        # multiprocessing
        _args_list = ((length, sampling_rate, *args) for args in args_list)
        with mp.Pool(processes) as pool:
            X = pool.starmap(_ecg_generate_trace, _args_list, chunksize=100)
        X = np.stack(X)

    if snr is not None:
        X = add_noise(X, snr=snr)

    return X



def _ecg_generate_trace(
    length, 
    sampling_rate=256, 
    heart_rate=70, 
    shift=0, 
    seed=None
    ):
    x = nk.ecg_simulate(
        sampling_rate=sampling_rate, 
        length=shift + length, 
        heart_rate=heart_rate, 
        noise=0., 
        random_state=seed,
    )
    return x[shift:]