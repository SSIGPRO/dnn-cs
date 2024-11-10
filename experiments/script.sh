#!/bin/bash

n=128
N_corr=100000
fs=256
heart_rate=(60 100)
ecg_corr_seed=42

python ./experiments/generate_ecg.py \
    --size $N_corr \
    --length $n \
    --sample-freq $fs \
    --heart-rate ${heart_rate[0]} ${heart_rate[1]} \
    --seed $ecg_corr_seed \
    --processes 32 \
    -vv
python ./experiments/compute_correlation.py \
    --size $N_corr \
    --length $n \
    --sample-freq $fs \
    --heart-rate ${heart_rate[0]} ${heart_rate[1]} \
    --seed $ecg_corr_seed \
    -vv



algorithm=TSOC
encoder=rakeness
m=64
corr=96af96a7ddfcb2f6059092c250e18f2a
loc=0.25
seed=0

for m in 16 32 48 64
do
    for seed in {0..19}
    do
        echo "supports m=$m, seed=$seed"
        python experiments/compute_supports_TSOC.py \
            --algorithm $algorithm \
            --encoder $encoder \
            --orthogonal \
            --measurements $m \
            --correlation $corr \
            --localization $loc \
            --seed $seed \
#             --processes \
#             --vv
    done
done

echo "RSNR"
python experiments/compute_rsnr_TSOC.py