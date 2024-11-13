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


# isnr=35           # 25, 35, 45
# algorithm=GR      # GR, TSOC, TSOC2
# encoder=rakeness  # standard, rakeness
# corr=96af96a7ddfcb2f6059092c250e18f2a
# loc=0.25

# m_list=(16 32 48 64)
# seed_list=($(seq 0 19))
# eta_list=(
#     0.9 
#     0.93    0.95    0.97    0.98    0.985    0.99 
#     0.993   0.995   0.997   0.998   0.9985   0.999 
#     0.9993  0.9995  0.9997  0.9998  0.99985  0.9999 
#     0.99993 0.99995 0.99997 0.99998 0.999985 0.99999
# )

# for m in "${m_list[@]}"
# do
#     for seed in "${seed_list[@]}"
#     do
#         python experiments/compute_supports.py \
#             --isnr $isnr \
#             --algorithm $algorithm \
#             --encoder $encoder \
#             --orthogonal \
#             --measurements $m \
#             --correlation $corr \
#             --localization $loc \
#             --seed $seed \
#             # --processes \
#             # --vv
#     done
# done

# echo "RSNR"
# python experiments/compute_rsnr.py \
#     --measurements "${m_list[@]}" \
#     --seed "${seed_list[@]}" \
#     --isnr $isnr \
#     --algorithm $algorithm \
#     --encoder $encoder \
#     --orthogonal \
#     --correlation $corr \
#     --localization $loc \
#     --eta "${eta_list[@]}" \
#     # --processes \
#     # --vv



###############################################################################

corr=96af96a7ddfcb2f6059092c250e18f2a
loc=0.25

m_list=(16 32 48 64)
seed_list=($(seq 0 19))
eta_list=(
    0.9 
    0.93    0.95    0.97    0.98    0.985    0.99 
    0.993   0.995   0.997   0.998   0.9985   0.999 
    0.9993  0.9995  0.9997  0.9998  0.99985  0.9999 
    0.99993 0.99995 0.99997 0.99998 0.999985 0.99999
)

for isnr in 35 25 45
do
    for encoder in "standard" "rakeness"
    do
        for algorithm in "GR" "TSOC"
        do
            echo $isnr $encoder $algorithm
            python experiments/compute_rsnr.py \
                --measurements "${m_list[@]}" \
                --seed "${seed_list[@]}" \
                --isnr $isnr \
                --algorithm $algorithm \
                --encoder $encoder \
                --orthogonal \
                --correlation $corr \
                --localization $loc \
                --eta "${eta_list[@]}" \
                # --processes \
                # --vv \
        done
    done
done

# encoder="standard"
# algorithm="TSOC"

# for isnr in 35 25 45
# do
#     echo $isnr $encoder $algorithm
#     python experiments/compute_rsnr.py \
#         --measurements "${m_list[@]}" \
#         --seed "${seed_list[@]}" \
#         --isnr $isnr \
#         --algorithm $algorithm \
#         --encoder $encoder \
#         --orthogonal \
#         --correlation $corr \
#         --localization $loc \
#         --eta "${eta_list[@]}" \
#         # -vv \
#         # --processes \
# done