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
# ECG Data Generation                                                          #
################################################################################

# # Parameters of the training data
# N_train=10000 # Number of ECG training examples
# n=128           # Length of each ECG signal
# fs=256          # Sampling frequency
# heart_rate=(60 100) # Heart rate range
# isnr=35         # Intrinsic signal-to-noise ratio
# ecg_seed=66     # Random seed for data generation
# processes=4     # Number of parallel processes

# # Parameters of the support of the training data
# # m=48
# # m_list=(16 32 48 64)
# m_list=(48)
# corr=96af96a7ddfcb2f6059092c250e18f2a
# loc=0.25
# # encoder="rakeness"
# source='random'    # Whether to use best or random sensing matrix
# seed_list=(0)          # "Random" seed for sensing matrix generation or index of the "best" sensing matrix according to "source"
# # source='best'
# # seed_list=(0 1 2 3 4 5 6 7)
# algorithm="TSOC"
# # orthogonal=True

# python ./experiments/generate_ecg.py \
#     --size $N_train \
#     --length $n \
#     --sample-freq $fs \
#     --heart-rate ${heart_rate[0]} ${heart_rate[1]} \
#     --isnr $isnr \
#     --seed $ecg_seed \
#     --processes $processes \
#     -vv

# for seed in "${seed_list[@]}"
# do
#     for orthogonal in "True" "False"
#     do
#         if [ "$orthogonal" = "True" ]; then
#                     orthogonal_flag="--orthogonal"
#                 else
#                     orthogonal_flag=""
#                 fi

#         for encoder in "standard" "rakeness" 
#         do
#             for m in "${m_list[@]}"
#             do
#                 python ./experiments/compute_supports.py \
#                     --size $N_train \
#                     --isnr $isnr \
#                     --algorithm $algorithm \
#                     --encoder $encoder \
#                     --source $source \
#                     --measurements $m \
#                     --correlation $corr \
#                     --localization $loc \
#                     --ecg_seed $ecg_seed\
#                     --seed $seed \
#                     $orthogonal_flag \
#                     --processes $processes\
#                     -vv
#             done
#         done
#     done
# done
# ###############################################################################
# TSOC Training for Multiple Configurations                                     #
# ###############################################################################

# Configuration Section
n=128               # Number of samples per signal
isnr=35             # Signal-to-noise ratio (SNR)

m=48                # Number of measurements
# m_list=(16 32 48 64)
# m_list=(48)
# seed_training=1     # Training-related random seed for reproducibility
seed_training_list=(0 1 2 3)
mode="rakeness"     # Encoder mode, change to 'rakeness' if needed
gpu=5               # GPU index

# optimizer="sgd"    # Optimizer for training
# lr=0.1              # Learning rate
# batch_size=50       # Batch size for training
# min_lr=0.001        # Minimum learning rate

optimizer="adam"    # Optimizer for training
lr=0.001            # Learning rate
batch_size=64       # Batch size for training
min_lr=0.00001      # Minimum learning rate

orthogonal=True     # Whether to use orthogonalized matrix
source='best'       # Whether to use best or random matrix
# seed_matrix=0       # Index or seed of the best or random matrix, respectivelly
seed_matrix_list=(0 1 2 3)

train_fraction=0.9  # Fraction of data used for training
factor=0.2          # Factor for ReduceLROnPlateau scheduler

min_delta=1e-4      # Minimum delta for early stopping and ReduceLROnPlateau
patience=40         # Patience for early stopping
epochs=1000         # Number of training epochs

N=2000000           # Number of training instances
basis="sym6"        # Wavelet basis function
fs=256              # Sampling frequency
heart_rate="60 100" # Heart rate range
threshold=0.5       # Threshold for metrics
processes=48        # Number of CPU processes for parallelism

for 
for m in "${m_list[@]}"
    do
        if [ "$orthogonal" = "True" ]; then
            orthogonal_flag="--orthogonal"
        else
            orthogonal_flag=""
        fi

        # Run the training script with the selected configuration
        python tsoc_training_norsnr.py \
            --n $n \
            --m $m \
            --epochs $epochs \
            --lr $lr \
            --optimizer $optimizer \
            --batch_size $batch_size \
            --N $N \
            --basis $basis \
            --fs $fs \
            --heart_rate $heart_rate \
            --isnr $isnr \
            --mode $mode \
            $orthogonal_flag \
            --source $source \
            --seed_matrix $seed_matrix \
            --seed_training $seed_training \
            --processes $processes \
            --threshold $threshold \
            --gpu $gpu \
            --train_fraction $train_fraction \
            --factor $factor \
            --min_lr $min_lr \
            --min_delta $min_delta \
            --patience $patience
    done

##############################################################################
# TSOC Reconstruction Performanve Evaluation for Multiple Configurations     #
##############################################################################

# # Core parameters
# n=128                # Number of samples per signal
# isnr=35              # Signal-to-noise ratio (SNR)
# m_list=(32 48 64)    # Number of measurements
# seed_training=1      # Training-related random seed for reproducibility
# mode="standard"      # Encoder mode, change to 'rakeness' if needed
# gpu=3                # GPU index

# # Optimizer configuration
# optimizer="adam"     # Optimizer for training
# lr=0.001             # Learning rate
# batch_size=64        # Batch size for training
# min_lr=0.00001       # Minimum learning rate

# # Other parameters
# orthogonal=False     # Whether to use orthogonalized matrix
# source="best"        # Whether to use best or random matrix
# seed_matrix=0        # Index or seed of the best or random matrix
# train_fraction=0.9   # Fraction of data used for training
# min_delta=1e-4       # Minimum delta for early stopping and ReduceLROnPlateau
# patience=40          # Patience for early stopping
# epochs=1000          # Number of training epochs
# N_train=2000000      # Number of training instances
# N_test=10000         # Number of test samples
# basis="sym6"         # Wavelet basis function
# fs=256               # Sampling frequency
# heart_rate="60,100"  # Heart rate range
# threshold=0.5        # Threshold for metrics
# processes=48         # Number of CPU processes for parallelism

# # Iterate over measurement configurations
# for source in "best" "random"
# do
#     for orthogonal in "True" "False"
#     do
#         for mode in in "standard" "rakeness"
#         do
#             for m in "${m_list[@]}"
#             do
#                 if [ "$orthogonal" = "True" ]; then
#                     orthogonal_flag="--orthogonal"
#                 else
#                     orthogonal_flag=""
#                 fi

#                 # Run the evaluation script with the selected configuration
#                 python detector_evaluation.py \
#                     --n $n \
#                     --m $m \
#                     --seed_training $seed_training \
#                     --mode $mode \
#                     --seed_matrix $seed_matrix \
#                     --isnr $isnr \
#                     --N_train $N_train \
#                     --train_fraction $train_fraction \
#                     --N_test $N_test \
#                     --basis $basis \
#                     --fs $fs \
#                     --heart_rate $heart_rate \
#                     $orthogonal_flag \
#                     --source $source \
#                     --processes $processes \
#                     --gpu $gpu \
#                     --min_lr $min_lr \
#                     --min_delta $min_delta \
#                     --patience $patience \
#                     --batch_size $batch_size \
#                     --threshold $threshold \
#                     --epochs $epochs \
#                     --lr $lr \
#                     --optimizer $optimizer
#             done
#         done
#     done
# done



################################################################################
# ECG Anomaly Data Generation #
################################################################################

# Parameters
# N_test=10000         # Number of ECG examples
# n=128           # Length of each ECG signal
# fs=256          # Sampling frequency
# heart_rate=(60 100) # Heart rate range
# isnr=35         # Intrinsic signal-to-noise ratio
# seed_ok=66         # Random seed for normal data
# seed_ko=2         # Random seed for anomalous data
# delta=0.005       # Intensity of anomalies
# processes=48    # Number of parallel processes

# python ./experiments/generate_ecg.py \
#     --size $N_test \
#     --length $n \
#     --sample-freq $fs \
#     --heart-rate ${heart_rate[0]} ${heart_rate[1]} \
#     --isnr $isnr \
#     --seed $seed_ok \
#     --processes $processes \
#     -vv

# python ./experiments/generate_anomalies.py \
#     --size $N_test \
#     --length $n \
#     --sample-freq $fs \
#     --heart-rate ${heart_rate[0]} ${heart_rate[1]} \
#     --isnr $isnr \
#     --seed_ok $seed_ok \
#     --seed_ko $seed_ko \
#     --delta $delta \
#     --processes $processes \


###############################################################################
# Standard detectors training                                                 #
###############################################################################

# # Configuration Section
# n=128               # Number of samples per signal
# m=48                # Number of measurements
# seed_detector=0     # Random seed associated to the detector
# isnr=35             # Signal-to-noise ratio (SNR)
# mode="standard"     # Sensing matrix mode, change to 'rakeness' if needed
# N_train=2000000     # Number of training instances
# train_fraction=0.9  # Fraction of data used for training
# basis="sym6"        # Wavelet basis function
# fs=256              # Sampling frequency
# heart_rate="60 100" # Heart rate range
# orthogonal=True     # Whether to use orthogonalized measurement matrix
# source='best'     # Sensing matrix source: 'best' or 'random'
# index=0             # Index or seed for the sensing matrix

# detector_type="LOF" # Detector type to evaluate (e.g., SPE, OCSVM, LOF)

# # Detector-specific configuration
# k=5                 # Parameter k for SPE, T2, or related detectors
# order=1             # Order parameter for AR detector
# kernel="rbf"        # Kernel type for OCSVM (e.g., linear, rbf, poly)
# nu=0.01              # Anomaly fraction for OCSVM
# nu_list=(0.001 0.01 0.1)
# neighbors=20        # Number of neighbors for LOF detector
# neighbors_list=(5 10 20 50)
# estimators=100      # Number of estimators for IF detector
# estimators_list=(5 10 20 50 100 200 500 1000)

# ks=(1 2 4 8 16 24)
# orders=(1 2 4 8 16 24)

# if [ "$orthogonal" = "True" ]; then
#     orthogonal_flag="--orthogonal"
# else
#     orthogonal_flag=""
# fi

# # for detector_type in "SPE" "T2" 
# # do
# #     for k in "${ks[@]}"
# # for order in "${orders[@]}"
# # for nu in "${nu_list[@]}"
# # for estimators in "${estimators_list[@]}"
# for neighbors in "${neighbors_list[@]}"
# do
#     python train_detector.py \
#         --n $n \
#         --m $m \
#         --seed_detector $seed_detector \
#         --mode $mode \
#         --isnr $isnr \
#         --detector_type $detector_type \
#         --N_train $N_train \
#         --train_fraction $train_fraction \
#         --basis $basis \
#         --fs $fs \
#         --heart_rate $heart_rate \
#         $orthogonal_flag \
#         --source $source \
#         --index $index \
#         --k $k \
#         --order $order \
#         --kernel $kernel \
#         --nu $nu \
#         --neighbors $neighbors \
#         --estimators $estimators
# done
# done


################################################################################
# Anomaly Detection Evaluation                                                 #
################################################################################

# # Configuration Section
# n=128               # Number of samples per signal
# m=32              # Number of measurements
# seed_ko=0           # Random seed for anomalies generation
# isnr=35             # Signal-to-noise ratio (SNR)
# mode="standard"     # Encoder mode, change to 'rakeness' if needed
# N_train=2000000     # Number of training instances
# N_test=10000        # Number of test instances
# train_fraction=0.9  # Fraction of data used for training
# basis="sym6"        # Wavelet basis function
# fs=256              # Sampling frequency
# heart_rate="60 100" # Heart rate range
# orthogonal=True     # Whether to use orthogonalized measurement matrix
# source='random'       # Whether to use best or random matrix
# index=0             # Index or seed of the best or random matrix, respectivelly
# processes=48        # Number of CPU processes
# gpu=3               # GPU index for evaluation

# detector_type="ZC"  # Detector type to evaluate (e.g., TSOC, SPE, OCSVM)
# delta=0.1           # Anomaly intensity parameter

# # TSOC-specific configuration
# detector_mode="self-assessment"  # Mode of operation for TSOC
# factor=0.2                # Augmentation/scheduling factor
# min_lr=0.001              # Minimum learning rate for optimizers
# min_delta=1e-4            # Minimum change in monitored metric for early stopping
# patience=40               # Patience for early stopping
# epochs=500                # Number of epochs for training
# lr=0.1                    # Learning rate
# batch_size=50             # Batch size for training
# threshold=0.5             # Threshold for TSOC detector

# # Standard detectors-related arguments
# # Parameter k for SPE, T2:
# k=5
# # Order parameter for AR detector:                       
# order=1
# # Parameters for OCSVM:                 
# kernel="rbf"              
# nu=0.5                    
# # Number of neighbors for LOF detector
# neighbors=10
# # Number of estimators for IF detector                      
# estimators=100

# ks=(1 2 4 8 16 24)
# orders=(1 2 4 8 16 24)

# detector_type="AR"


# if [ "$orthogonal" = "True" ]; then
#     orthogonal_flag="--orthogonal"
# else
#     orthogonal_flag=""
# fi

# for order in "${orders[@]}"
# do
# # Run the evaluation script with the selected configuration
# python detector_evaluation.py \
#     --n $n \
#     --m $m \
#     --seed_ko $seed_ko \
#     --mode $mode \
#     --isnr $isnr \
#     --detector_type $detector_type \
#     --delta $delta \
#     --N_train $N_train \
#     --N_test $N_test \
#     --train_fraction $train_fraction \
#     --basis $basis \
#     --fs $fs \
#     --heart_rate $heart_rate \
#     $orthogonal_flag \
#     --source $source \
#     --index $index \
#     --processes $processes \
#     --gpu $gpu \
#     --detector_mode $detector_mode \
#     --factor $factor \
#     --min_lr $min_lr \
#     --min_delta $min_delta \
#     --patience $patience \
#     --epochs $epochs \
#     --lr $lr \
#     --batch_size $batch_size \
#     --threshold $threshold \
#     --k $k \
#     --order $order \
#     --kernel $kernel \
#     --nu $nu \
#     --neighbors $neighbors \
#     --estimators $estimators
# done





