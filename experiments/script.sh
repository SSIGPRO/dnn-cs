#!/bin/bash

n=128
N_corr=100000
fs=256
heart_rate=(60 100)
ecg_corr_seed=42

# python ./experiments/generate_ecg.py \
#     --size $N_corr \
#     --length $n \
#     --sample-freq $fs \
#     --heart-rate ${heart_rate[0]} ${heart_rate[1]} \
#     --seed $ecg_corr_seed \
#     --processes 32 \
#     -vv
# python ./experiments/compute_correlation.py \
#     --size $N_corr \
#     --length $n \
#     --sample-freq $fs \
#     --heart-rate ${heart_rate[0]} ${heart_rate[1]} \
#     --seed $ecg_corr_seed \
#     -vv


isnr=25
algorithm=TSOC
encoder=rakeness
corr=96af96a7ddfcb2f6059092c250e18f2a
loc=0.25

m_list=(16 32 48 64)
seed_list=($(seq 0 19))

for m in "${m_list[@]}"
do
    for seed in "${seed_list[@]}"
    do
        python experiments/compute_supports_TSOC.py \
            --isnr $isnr \
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
python experiments/compute_rsnr_TSOC.py \
    --measurements "${m_list[@]}" \
    --seed "${seed_list[@]}" \
    --isnr $isnr \
    --algorithm $algorithm \
    --encoder $encoder \
    --orthogonal \
    --correlation $corr \
    --localization $loc \
    # --processes \
    # --vv