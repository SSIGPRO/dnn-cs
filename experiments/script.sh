#!/bin/bash


################################################################################
# Computation of ECG Correlation matrix                                        #
################################################################################

# n=128
# N_corr=100000
# fs=256
# heart_rate=(60 100)
# ecg_corr_seed=42

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


################################################################################
# Computation of ECG Supports and RSNR (GR, TSOC                               #
################################################################################

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

# for isnr in 35 25 45
# do
#     for encoder in "standard" "rakeness"
#     do
#         # for algorithm in "GR" "TSOC"
#         # do
#             for m in "${m_list[@]}"
#             do
#                 for seed in "${seed_list[@]}"
#                 do
#                     echo $isnr $encoder $algorithm $m $seed
#                     python experiments/compute_supports.py \
#                         --isnr $isnr \
#                         --algorithm $algorithm \
#                         --encoder $encoder \
#                         --measurements $m \
#                         --correlation $corr \
#                         --localization $loc \
#                         --eta "${eta_list[@]}" \
#                         --seed $seed \
#                         # --orthogonal \
#                         # --processes \
#                         # --vv
#                 done
#             done
#         # done
#     done
# done

# for isnr in 35 25 45
# do
#     for encoder in "standard" "rakeness"
#     do
#         for algorithm in "GR" "TSOC"
#         do
#             echo $isnr $encoder $algorithm
#             python experiments/compute_rsnr.py \
#                 --measurements "${m_list[@]}" \
#                 --seed "${seed_list[@]}" \
#                 --isnr $isnr \
#                 --algorithm $algorithm \
#                 --encoder $encoder \
#                 --correlation $corr \
#                 --localization $loc \
#                 --eta "${eta_list[@]}" \
#                 # --orthogonal \
#                 # --processes \
#                 # --vv \
#         done
#     done
# done


################################################################################
# Computation of ECG Supports and RSNR (TSOC2)                                 #
################################################################################


# corr=96af96a7ddfcb2f6059092c250e18f2a
# loc=0.25

# m_list=(16 32 48 64)
# seed_list=($(seq 0 19))

# isnr=35
# encoder="standard"
# algorithm="TSOC2"

# for encoder in "standard" "rakeness"
# do
#     for m in "${m_list[@]}"
#     do
#         for seed in "${seed_list[@]}"
#         do
#             echo $isnr $encoder $algorithm $m $seed
#             python experiments/compute_supports.py \
#                 --isnr $isnr \
#                 --algorithm $algorithm \
#                 --encoder $encoder \
#                 --measurements $m \
#                 --correlation $corr \
#                 --localization $loc \
#                 --seed $seed \
#                 --orthogonal \
#                 # --processes \
#                 # --vv
#         done
#     done
# done

# echo $isnr $encoder $algorithm
# python experiments/compute_rsnr.py \
#     --measurements "${m_list[@]}" \
#     --seed "${seed_list[@]}" \
#     --isnr $isnr \
#     --algorithm $algorithm \
#     --encoder $encoder \
#     --orthogonal \
#     # --processes \
#     # --vv \

################################################################################
# TSOC Training Script for Multiple Configurations                             #
################################################################################

# Configuration Section
n=128               # Number of samples per signal
m=32                # Number of measurements
seed=0              # Random seed for reproducibility
isnr=35             # Signal-to-noise ratio (SNR)
mode="standard"     # Encoder mode, change to 'rakeness' if needed
gpu=3               # GPU index
train_fraction=0.9  # Fraction of data used for training
factor=0.2          # Factor for ReduceLROnPlateau scheduler
min_lr=0.001        # Minimum learning rate
min_delta=1e-4      # Minimum delta for early stopping and ReduceLROnPlateau
patience=40         # Patience for early stopping
epochs=500          # Number of training epochs
lr=0.1              # Learning rate
batch_size=50       # Batch size for training
N=2000000           # Number of training instances
basis="sym6"        # Wavelet basis function
fs=256              # Sampling frequency
heart_rate="60 100" # Heart rate range
threshold=0.5       # Threshold for metrics
orthogonal=True    # Whether to use orthogonalized matrix
processes=48        # Number of CPU processes for parallelism

echo "Running TSOC training with ISNR=$isnr, Mode=$mode, Orthogonalization=$orthogonal, Measurements=$m, \
Seed=$seed, TrainingInstances=$N, Basis=$basis, FS=$fs, HeartRate=($heart_rate), Processes=$processes, \
Threshold=$threshold, GPU=$gpu, TrainFraction=$train_fraction, \
Factor=$factor, MinLR=$min_lr, MinDelta=$min_delta, Patience=$patience"

if [ "$orthogonal" = "True" ]; then
    orthogonal_flag="--orthogonal"
else
    orthogonal_flag=""
fi

# Run the training script with the selected configuration
python tsoc_training.py \
    --n $n \
    --m $m \
    --epochs $epochs \
    --lr $lr \
    --batch_size $batch_size \
    --N $N \
    --basis $basis \
    --fs $fs \
    --heart_rate $heart_rate \
    --isnr $isnr \
    --mode $mode \
    $orthogonal_flag \
    --seed $seed \
    --processes $processes \
    --threshold $threshold \
    --gpu $gpu \
    --train_fraction $train_fraction \
    --factor $factor \
    --min_lr $min_lr \
    --min_delta $min_delta \
    --patience $patience

