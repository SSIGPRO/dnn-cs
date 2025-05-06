import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

from ponte_utils import *


import torch

root = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.insert(0, os.path.join(root, 'src'))

from models import models_dir
dataset_dir = '/srv/newpenny/dataset/Ponte_S101/Data/'
from models.ae import AE
model_folder = f'{models_dir}AEPonte'

# fs = 100
fs = 50
n = 600               # Number of samples per signal
m = 64                # Number of measurements
# n = 256               # Number of samples per signal
# m = 32                # Number of measurements
K = 4
method = 'burg'
seed_training=0
gpu=1               # GPU index

beta=0.0              # Latent volume regularization weight
gamma=0.0 

opt="adam"    # Optimizer for training
lr=0.001            # Learning rate
batch_size=64      # Batch size for training
min_lr=0.00001      # Minimum learning rate

train_fraction=0.9  # Fraction of data used for training
factor=0.2          # Factor for ReduceLROnPlateau scheduler

min_delta=1e-8      # Minimum delta for early stopping and ReduceLROnPlateau
patience=40         # Patience for early stopping
epochs=1000         # Number of training epochs
data_augmentation = True
max_batch = 20

# ------------------ CONSTANTS ------------------
natural_frequencies = [4.05, 6.3, 9.69, 13.29, 15.93]
margin = 0.075
subbands = [(f - margin * f, f + margin * f) for f in natural_frequencies]
eps = 0
eps2 = 0

if method == 'burg':
    resolution = K # per burg resolution 
else:
     resolution = 1

# ------------------ GPU ------------------
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

# ------------------ Signal ------------------

# load
Xko = loadmat(os.path.join(dataset_dir, 'downsampled_dam_B.mat'))
Xko = Xko['downsampled_dam_B'][0]
Xok = loadmat(os.path.join(dataset_dir, 'downsampled_undamage.mat'))
Xok = Xok['downsampled_undamage'][0]
# downsample
if fs == 50:
    Xok = decimate(Xok, q=2, ftype='iir')
    Xko = decimate(Xko, q=2, ftype='iir')
    n = n // 2
    m = m // 2
print(len(Xko))
nn = n * K

# Preprpcessing
X = sliding_window(Xok, max_batch*n, max_batch*n)
indices = np.arange(0, len(X))

# indices2 = indices.copy()
indices_test = indices[::2]
indices_train = np.setdiff1d(indices, indices_test)

# np.random.seed(0)
# indices = np.random.permutation(indices)  # Shuffle indices
# split_idx = len(indices) // 2  # 50% split for train/test
# indices_train, indices_test = indices[:split_idx], indices[split_idx:]

print(indices_test)
print(indices_train)

Xok_test = X[indices_test]
Xok_test = sliding_window(Xok_test.flatten(), n, n)
P = len(Xok_test) // K * K
Xok_test = Xok_test[:P]

Xko_test = sliding_window(Xko, n, n)
Q = len(Xko_test) // K * K
Xko_test = Xko_test[:Q]

R = min(P, Q)
Xok_test = Xok_test[:R]
Xko_test = Xko_test[:R]

Xok_train = X[indices_train]
Xok_train = np.concatenate([sliding_window(X_, n, 1) for X_ in Xok_train])
N_train = len(Xok_train) // 100 * 40
Xok_train, _ = select_random_windows(Xok_train, N_train, seed_training)

Xok_train2 = X[indices_train]
Xok_train2 = sliding_window(Xok_train2.flatten(), n, n)
Xok_train2 = Xok_train2[:len(Xok_train2) // K * K]

mean = Xok_train.mean()
std = Xok_train.std()
Xok_train = (Xok_train - mean) / std
Xok_train2 = (Xok_train2 - mean) / std
Xok_test = (Xok_test - mean) / std
Xko_test = (Xko_test - mean) / std

print(Xok_train.std(), Xok_train.mean())
print(Xok_test.std(), Xok_test.mean())
print(Xok_train2.std(), Xok_train2.mean())

# ------------------ Model ------------------
ae = AE(n, m)
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
ae.to(device) # move the network to GPU
model_name = f'AE-N={Xok_train.shape[0]}_n={n}_m={m}_fs={fs}-beta={beta}_gamma={gamma}'\
                f'_epochs={epochs}_bs={batch_size}_opt={opt}_lr={lr}'\
                f'_tf={train_fraction}_minlr={min_lr}_p={patience}'\
                f'_mind={min_delta}_seedtrain={seed_training}'    
model_path = os.path.join(model_folder, f'{model_name}.pth')
ae.to(device) # move the network to GPU
ae.load_state_dict(torch.load(model_path, weights_only=True))
ae.eval()

num_trainable_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")

# ------------------ PCA ------------------
Cx = 1/(len(Xok_train)-1) * Xok_train.T @ Xok_train
# A part from Xok_train2, 2 in names refers to PCA!

# ------------------ Reconstruction and Re-encoding ------------------
# Process training data
Xok_train_hat, Yok_train, Yok_train_hat = recon_reencode(Xok_train2, ae, device)
Xok_train_hat2 = pca(Cx, Xok_train2, m)
# Process normal test data
Xok_test_hat, Yok_test, Yok_test_hat = recon_reencode(Xok_test, ae, device)
Xok_test_hat2 = pca(Cx, Xok_test, m)
# Process anomalous test data
Xko_test_hat, Yko_test, Yko_test_hat = recon_reencode(Xko_test, ae, device)
Xko_test_hat2 = pca(Cx, Xko_test, m)

# ------------------ Reconstruction performance ------------------
rsnr_train = compute_rsnr(Xok_train2, Xok_train_hat)
rsnr_test = compute_rsnr(Xok_test, Xok_test_hat)
rsnr_train2 = compute_rsnr(Xok_train2, Xok_train_hat2)
rsnr_test2 = compute_rsnr(Xok_test, Xok_test_hat2)
RSNR = pd.Series(
    index=['AE', 'PCA'],
      data=[np.mean(rsnr_test), np.mean(rsnr_test2)],
      name = 'RSNR'
 )
print(RSNR)


# ------------------ Spectrum estimation ------------------
# freqs, coeff_train, cc_train = estimate_spectrum(
#     n, fs, Xok_train2, K, resolution, method
#     )
# freqs, coeff_train_hat, cc_train_hat = estimate_spectrum(
#     n, fs, Xok_train_hat, K, resolution, method
#     )
freqs, coeff_ok, cc_ok = estimate_spectrum(
    n, fs, Xok_test, K, resolution, method
    )
freqs, coeff_ok_hat, cc_ok_hat = estimate_spectrum(
    n, fs, Xok_test_hat, K, resolution, method
    )
freqs, coeff_ko, cc_ko = estimate_spectrum(
    n, fs, Xko_test, K, resolution, method
    )
freqs, coeff_ko_hat, cc_ko_hat = estimate_spectrum(
    n, fs, Xko_test_hat, K, resolution, method
    )
freqs, coeff_train_hat2, cc_train_hat2 = estimate_spectrum(
    n, fs, Xok_train_hat2, K, resolution, method
    )
freqs, coeff_ok_hat2, cc_ok_hat2 = estimate_spectrum(
    n, fs, Xok_test_hat2, K, resolution, method)
freqs, coeff_ko_hat2, cc_ko_hat2 = estimate_spectrum(
    n, fs, Xko_test_hat2, K, resolution, method
    )

# ------------------ Detection performance ------------------
DETECTION = pd.Series()
# Self-assessment score
score_ok = np.sum((Yok_test - Yok_test_hat)**2, axis=-1)
score_ok = np.mean(sliding_window(score_ok.flatten(), K, K), axis=-1)
score_ko = np.sum((Yko_test - Yko_test_hat)**2, axis=-1)
score_ko = np.mean(sliding_window(score_ko.flatten(), K, K), axis=-1)
score_ko = score_ko[:len(score_ok)]

label = np.concatenate([np.zeros(len(score_ko)), np.ones(len(score_ok))])
scores_ae = np.concatenate([score_ok, score_ko])


# average MSISSD score
score2ok = np.mean(
    [
        MSISSD(cc_train_hat2, cc, subbands, freqs, eps) 
        for cc in coeff_ok_hat
    ], axis=1
)

score2ko = np.mean(
    [
        MSISSD(cc_train_hat2, cc, subbands, freqs, eps) 
        for cc in coeff_ko_hat
    ], axis=1
)

score2ko = score2ko[:len(score2ok)]
scores_avmsissd = np.concatenate([score2ok, score2ko])


# MSISSD scores
score3ok = np.array(
    [
        MSISSD(cc_train_hat2, cc, subbands, freqs, eps) 
        for cc in coeff_ok_hat
    ]
)

score3ko = np.array(
    [
        MSISSD(cc_train_hat2, cc, subbands, freqs, eps) 
        for cc in coeff_ko_hat
    ]
)

scores_msissd = [
    np.concatenate([score3ok[:, i], score3ko[:, i]])
      for i in range(len(subbands))
      ]

# SISSD over whole spectrum
score4ok = np.array(
    [
        MSISSD(cc_train_hat2, cc, [(freqs.min(), freqs.max())], freqs, eps) 
        for cc in coeff_ok_hat
    ]
)[:, 0]

score4ko = np.array(
    [
        MSISSD(cc_train_hat2, cc, [(freqs.min(), freqs.max())], freqs, eps) 
        for cc in coeff_ko_hat
    ]
)[:, 0]
score4ko = score4ko[:len(score4ok)]
scores_allmsissd = np.concatenate([score4ok, score4ko])
print(scores_allmsissd.shape)

DETECTION['AE'] = roc_auc_score(label, scores_ae)
DETECTION['Avg MSISSD'] = roc_auc_score(label, scores_avmsissd)
for i in range(len(subbands)):
    DETECTION[f'SISSD_{i}'] = roc_auc_score(label, scores_msissd[i])
DETECTION['All MSISSD'] = roc_auc_score(label, scores_allmsissd)

print(DETECTION)





