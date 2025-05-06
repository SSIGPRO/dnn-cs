# ###############################################################################
# Autoencoder Ponte Training for Multiple Configurations                        #
# ###############################################################################

# # Configuration Section
# n=512               # Number of samples per signal

# m=32                # Number of measurements
# # m_list=(16 32 48 64)
# m_list=(64)
# seed_training=2     # Training-related random seed for reproducibility
# # seed_training_list=(0 1 2 3)
# seed_training_list=(3)
# gpu=1               # GPU index

# beta=0.0              # Latent volume regularization weight
# gamma=0.0            # Latent reconstruction regularization weight

# beta_list=(0.0001 0.001 0.01 0.1 1 10 100 1000)
# beta_list=(0)

# optimizer="adam"    # Optimizer for training
# lr=0.001            # Learning rate
# batch_size=64      # Batch size for training
# min_lr=0.00001      # Minimum learning rate

# train_fraction=0.9  # Fraction of data used for training
# factor=0.2          # Factor for ReduceLROnPlateau scheduler

# min_delta=1e-8      # Minimum delta for early stopping and ReduceLROnPlateau
# patience=40         # Patience for early stopping
# epochs=1000         # Number of training epochs

# fs=50              # Sampling frequency
# processes=48        # Number of CPU processes for parallelism

# for seed_training in "${seed_training_list[@]}"
# do
#     # for gamma in "${gamma_list[@]}"
#     for beta in "${beta_list[@]}"
#     do
#         for m in "${m_list[@]}"
#         do

#             # Run the training script with the selected configuration
#             python ponte_ae_training.py \
#                 --n $n \
#                 --m $m \
#                 --epochs $epochs \
#                 --lr $lr \
#                 --optimizer $optimizer \
#                 --beta $beta \
#                 --gamma $gamma \
#                 --batch_size $batch_size \
#                 --fs $fs \
#                 --seed_training $seed_training \
#                 --processes $processes \
#                 --gpu $gpu \
#                 --train_fraction $train_fraction \
#                 --factor $factor \
#                 --min_lr $min_lr \
#                 --min_delta $min_delta \
#                 --patience $patience
#         done
#     done
# done

#!/bin/bash

###############################################################################
# TSOC Training with Compressed Sensing - Grid Search                         #
###############################################################################

# # Fixed Parameters
# n=300
# window_size=20
# fraction=40
# mode="rakeness"
# gpu=1
# optimizer="adam"
# lr=0.01
# batch_size=50
# min_lr=0.0001
# orthogonal=True
# source='best'
# alpha=0.9
# train_fraction=0.9
# factor=0.2
# min_delta=1e-4
# patience=30
# epochs=500
# fs=50
# threshold=0.5
# processes=4

# # Configuration Lists
# m_list=(75)
# seed_training_list=(0)

# # Handle orthogonal flag
# if [ "$orthogonal" = "True" ]; then
#     orthogonal_flag="--orthogonal"
# else
#     orthogonal_flag=""
# fi

# # Training Loop
# for seed_training in "${seed_training_list[@]}"
# do
#     for m in "${m_list[@]}"
#     do
#         echo "Training with m=$m, seed_matrix=$seed_matrix, seed_training=$seed_training"

#         python tsoc_training.py \
#             --n $n \
#             --m $m \
#             --window_size $window_size \
#             --fraction $fraction \
#             --epochs $epochs \
#             --lr $lr \
#             --optimizer $optimizer \
#             --batch_size $batch_size \
#             --fs $fs \
#             --mode $mode \
#             $orthogonal_flag \
#             --source $source \
#             --alpha $alpha \
#             --seed_training $seed_training \
#             --processes $processes \
#             --threshold $threshold \
#             --gpu $gpu \
#             --train_fraction $train_fraction \
#             --factor $factor \
#             --min_lr $min_lr \
#             --min_delta $min_delta \
#             --patience $patience
#         done
# done

###############################################################################
# TSOC V1/V2/V3 Training with Compressed Sensing - Grid Search                         #
###############################################################################

# # Fixed Parameters
# v=2
# n=300
# window_size=20
# fraction=100
# mode="rakeness"
# gpu=5
# optimizer="adam"
# # lr=0.01
# lr=0.001
# # batch_size=50
# batch_size=64
# # min_lr=0.0001
# min_lr=0.00001
# orthogonal=True
# source='best'
# alpha=0.5
# train_fraction=0.9
# factor=0.2
# min_delta=1e-5
# patience=40
# epochs=1000
# fs=50
# threshold=0.5
# processes=4

# # Configuration Lists
# m_list=(50)
# alpha_list=(0.6 0.7 0.8)
# alpha_list=(0.5 0.6 0.7 0.8 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.99)
# alpha_list=(0.5)
# seed_training_list=(0)

# # Handle orthogonal flag
# if [ "$orthogonal" = "True" ]; then
#     orthogonal_flag="--orthogonal"
# else
#     orthogonal_flag=""
# fi

# # Training Loop
# for seed_training in "${seed_training_list[@]}"
# do
#     for alpha in "${alpha_list[@]}"
#     do
#         for m in "${m_list[@]}"
#         do
#         echo "Training with m=$m, seed_matrix=$seed_matrix, seed_training=$seed_training"

#         python tsoc_training_v$v.py \
#             --n $n \
#             --m $m \
#             --window_size $window_size \
#             --fraction $fraction \
#             --epochs $epochs \
#             --lr $lr \
#             --optimizer $optimizer \
#             --batch_size $batch_size \
#             --fs $fs \
#             --mode $mode \
#             $orthogonal_flag \
#             --source $source \
#             --alpha $alpha \
#             --seed_training $seed_training \
#             --processes $processes \
#             --threshold $threshold \
#             --gpu $gpu \
#             --train_fraction $train_fraction \
#             --factor $factor \
#             --min_lr $min_lr \
#             --min_delta $min_delta \
#             --patience $patience
#         done
#     done
# done

###############################################################################
# ML - based Support Classifier Training                                      #
###############################################################################

# # Fixed Parameters
# n=300
# m=50
# window_size=20
# fraction=100
# mode="rakeness"
# orthogonal=True
# source='best'
# fs=50
# threshold=0.5
# processes=4
# estimator_type='random_forest'
# n_estimators=200
# seed_training=0
# seed_data=0
# seed_matrix=0

# # Configuration Lists
# m_list=(50 75)

# # Handle orthogonal flag
# if [ "$orthogonal" = "True" ]; then
#     orthogonal_flag="--orthogonal"
# else
#     orthogonal_flag=""
# fi

# # Training Loop

# for m in "${m_list[@]}"
# do
# echo "Training with m=$m, seed_matrix=$seed_matrix"

# python support_estimator_training.py \
#     --n $n \
#     --m $m \
#     --window_size $window_size \
#     --fraction $fraction \
#     --fs $fs \
#     --mode $mode \
#     $orthogonal_flag \
#     --source $source \
#     --seed_training $seed_training\
#     --seed_data $seed_data \
#     --seed_matrix $seed_matrix \
#     --processes $processes \
#     --threshold $threshold \
#     --estimator_type $estimator_type \
#     --n_estimators $n_estimators
# done

###############################################################################
# ML - based Support Classifier Tuning                                        #
###############################################################################

# Fixed Parameters
n=300
m=50
window_size=20
mode="rakeness"
orthogonal=True
source='best'
fs=50
threshold=0.5
processes=32
estimator_type='random_forest'
seed_training=0
seed_data=0
seed_matrix=0

num_samples=1000

# Configuration Lists
m_list=(75)

# Handle orthogonal flag
if [ "$orthogonal" = "True" ]; then
    orthogonal_flag="--orthogonal"
else
    orthogonal_flag=""
fi

# Training Loop

for m in "${m_list[@]}"
do
echo "Tuning with m=$m, seed_matrix=$seed_matrix"

python support_estimator_tuning.py \
    --n $n \
    --m $m \
    --window_size $window_size \
    --fs $fs \
    --mode $mode \
    $orthogonal_flag \
    --source $source \
    --seed_training $seed_training\
    --seed_data $seed_data \
    --seed_matrix $seed_matrix \
    --processes $processes \
    --num_samples $num_samples \
    --threshold $threshold \
    --estimator_type $estimator_type
done

###############################################################################
# Standard detectors training                                                 #
###############################################################################

# # Configuration Section
# n=300               # Number of samples per signal
# m=50                # Number of measurements
# m_list=(50 75)
# window_size=20
# fraction=100

# seed_detector=0     # Random seed associated to the detector
# mode="rakeness"     # Sensing matrix mode, change to 'rakeness' if needed
# train_fraction=0.9  # Fraction of data used for training
# fs=50               # Sampling frequency
# orthogonal=True     # Whether to use orthogonalized measurement matrix
# source='best'       # Sensing matrix source: 'best' or 'random'
# # seed_matrix=0       # Index or seed for the sensing matrix
# seed_matrix_list=(0)

# detector_type="OC" # Detector type to evaluate (e.g., SPE, OCSVM, LOF)

# # Detector-specific configuration
# k=5                 # Parameter k for SPE, T2, or related detectors
# order=1             # Order parameter for AR detector
# kernel="poly"      # Kernel type for OCSVM (e.g., linear, rbf, poly)
# kernel2="gaussian"      # Kernel type for OCSVM (e.g., linear, rbf, poly)
# # kernel="poly"       
# nu=0.01             # Anomaly fraction for OCSVM
# nu_list=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.6 0.7)
# neighbors=20        # Number of neighbors for LOF detector
# neighbors_list=(5 10 20 50)
# estimators=100      # Number of estimators for IF detector
# estimators_list=(5 10 20 50 100 200 500 1000)
# bandwidth=0.1
# bandwidth_list=(0 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0)
# ks=(1 2 4 8 16 24 32 46)
# orders=(1 2 4 8 16 24 32 46)


# if [ "$orthogonal" = "True" ]; then
#     orthogonal_flag="--orthogonal"
# else
#     orthogonal_flag=""
# fi

# for seed_matrix in "${seed_matrix_list[@]}"
# do
#     # for estimators in "${estimators_list[@]}"
#     # for bandwidth in "${bandwidth_list[@]}"
#     # for k in "${ks[@]}"
#     # for order in "${orders[@]}"
#     for nu in "${nu_list[@]}"
#     # for detector_type in "MD" "TV" "ZC" "pk-pk" "energy"
#     # for neighbors in "${neighbors_list[@]}"
#     # for k in "${ks[@]}"
#     do
#         # for detector_type in "SPE" "T2" 
#         for m in "${m_list[@]}" 
#         do
#         python train_detector.py \
#             --n $n \
#             --m $m \
#             --window_size $window_size \
#             --fraction $fraction \
#             --seed_detector $seed_detector \
#             --mode $mode \
#             --detector_type $detector_type \
#             --train_fraction $train_fraction \
#             --fs $fs \
#             $orthogonal_flag \
#             --source $source \
#             --seed_matrix $seed_matrix \
#             --k $k \
#             --order $order \
#             --kernel $kernel \
#             --kernel2 $kernel2 \
#             --nu $nu \
#             --neighbors $neighbors \
#             --estimators $estimators \
#             --bandwidth $bandwidth
#         done
#     done
# done

# ###############################################################################
# TSOC Detector Training for Multiple Configurations                            #
# ###############################################################################

# # Configuration Section
# n=300               # Number of samples per signal

# m=50                # Number of measurements
# version=v2
# version_list=(v1 v2)
# # m_list=(16 32 48 64)
# m_list=(50 75)
# # seed_training=1     # Training-related random seed for reproducibility
# seed_training_list=(0)
# # seed_training_list=(0)
# mode="rakeness"     # Encoder mode, change to 'rakeness' if needed
# gpu=5               # GPU index

# window_size=20
# fraction=100

# optimizer="adam"    # Optimizer for training
# lr=0.001            # Learning rate
# batch_size=64      # Batch size for training
# min_lr=0.00001      # Minimum learning rate

# orthogonal=True     # Whether to use orthogonalized matrix
# source='best'       # Whether to use best or random matrix
# # seed_matrix=0       # Index or seed of the best or random matrix, respectivelly
# # seed_matrix_list=(0 1 2 3)
# seed_matrix_list=(0)
# alpha=0.5           # training loss weight promoting either FN (>0.5) or FP (<0.5) reduction 
# # alpha_list=(0.5)

# train_fraction=0.9  # Fraction of data used for training
# factor=0.2          # Factor for ReduceLROnPlateau scheduler

# min_delta=1e-5      # Minimum delta for early stopping and ReduceLROnPlateau
# patience=40         # Patience for early stopping
# epochs=1000         # Number of training epochs

# fs=50              # Sampling frequency
# threshold=0.5       # Threshold for metrics
# processes=48        # Number of CPU processes for parallelism


# if [ "$orthogonal" = "True" ]; then
#     orthogonal_flag="--orthogonal"
# else
#     orthogonal_flag=""
# fi

# # for alpha in "${alpha_list[@]}"
# # do
#     for seed_training in "${seed_training_list[@]}"
#     do
#         for seed_matrix in "${seed_matrix_list[@]}"
#         do
#             for m in "${m_list[@]}"
#             do
#                 for version in "${version_list[@]}"
#                 do

#                 # Run the training script with the selected configuration
#                 python tsoc_training_detector.py \
#                     --n $n \
#                     --m $m \
#                     --epochs $epochs \
#                     --window_size $window_size \
#                     --fraction $fraction \
#                     --lr $lr \
#                     --optimizer $optimizer \
#                     --batch_size $batch_size \
#                     --fs $fs \
#                     --mode $mode \
#                     $orthogonal_flag \
#                     --source $source \
#                     --seed_matrix $seed_matrix \
#                     --alpha $alpha \
#                     --seed_training $seed_training \
#                     --processes $processes \
#                     --threshold $threshold \
#                     --gpu $gpu \
#                     --train_fraction $train_fraction \
#                     --factor $factor \
#                     --min_lr $min_lr \
#                     --min_delta $min_delta \
#                     --patience $patience \
#                     --version $version
#             done
#         done
#     done
# done

# ###############################################################################
# Support Estimator Detector Training for Multiple Configurations               #
# ###############################################################################

# # Fixed Parameters
# n=300
# m=75
# version=v2
# window_size=20
# fraction=100
# mode="rakeness"
# orthogonal=True
# source='best'
# fs=50
# threshold=0.5
# processes=4
# estimator_type='random_forest'
# n_estimators=200
# seed_training=0
# seed_data=0
# seed_matrix=0

# # Configuration Lists
# m_list=(50 75)

# # Handle orthogonal flag
# if [ "$orthogonal" = "True" ]; then
#     orthogonal_flag="--orthogonal"
# else
#     orthogonal_flag=""
# fi

# # Training Loop

# for m in "${m_list[@]}"
# do
# echo "Training with m=$m, seed_matrix=$seed_matrix"

# python support_estimator_training_detector.py \
#     --n $n \
#     --m $m \
#     --window_size $window_size \
#     --fraction $fraction \
#     --fs $fs \
#     --mode $mode \
#     $orthogonal_flag \
#     --source $source \
#     --seed_training $seed_training \
#     --seed_data $seed_data \
#     --seed_matrix $seed_matrix \
#     --processes $processes \
#     --threshold $threshold \
#     --estimator_type $estimator_type \
#     --n_estimators $n_estimators \
#     --version $version
# done

################################################################################
# Anomaly Detection Evaluation                                                 #
################################################################################

# # Configuration Section
# n=300               # Number of samples per signal
# m=50                # Number of measurements
# m_list=(50 75)
# train_fraction=0.9  # Fraction of data used for training
# fs=50              # Sampling frequency

# seed_matrix=0       # Index or seed of the best or random matrix, respectivelly
# processes=64        # Number of CPU processes
# gpu=2               # GPU index for evaluation

# mode="rakeness"     # Encoder mode, change to 'rakeness' if needed
# orthogonal=True     # Whether to use orthogonalized measurement matrix

# detector_type="random_forest"  # Detector type to evaluate (e.g., TSOC, SPE, OCSVM)
# source='best'       # Whether to use best or random matrix

# window_size=20
# fraction=100

# # TSOC-specific configuration
# # detector_mode="self-assessment"  # Mode of operation for TSOC
# # detector_mode='self-assessment', 'autoencoder', 'sparsity', 'sparsity-threshold', 'self-assessment-complement', 'complement', 'MD'
# # detector_mode="self-assessment3"
# detector_mode="sparsity"
# factor=0.2                # Augmentation/scheduling factor

# patience=40               # Patience for early stopping
# epochs=1000                # Number of epochs for training

# lr=0.001                    # Learning rate
# batch_size=64             # Batch size for training
# optimizer="adam"    # Optimizer for training
# min_lr=0.00001              # Minimum learning rate for optimizers

# # AE-related arguments
# # min_delta=1e-8            # Minimum change in monitored metric for early stopping
# gamma=0.0                 # Regularization of the lattent space reconstruction"
# beta=0.0                  # Latent volume regularization weight

# # TSOC-related arguments
# min_delta=1e-5            # Minimum change in monitored metric for early stopping
# threshold=0.5             # threshold 
# alpha=0.5                 # loss weight
# # BPDN-related arguments
# sigma=1e-4 # BPDN threshold
# # Standard detectors-related arguments
# seed_detector=0
# # Parameter k for SPE, T2:
# k=5
# # Order parameter for AR detector:                       
# order=1
# # Parameters for OCSVM:                 
# # kernel="rbf"    
# kernel="rbf"  
# kernel2="gaussian"        
# nu=0.5                    
# # Number of neighbors for LOF detector
# neighbors=10
# # Number of estimators for IF detector                      
# estimators=200
# bandwidth=0.1


# # alpha_list=(0.001 0.01 0.1 0.25 0.4 0.6 0.75 0.9 0.99)
# alpha_list=(0.25 0.6)
# gamma_list=(0.0 0.0001 0.001 0.01 0.1 1 10 100 1000)
# beta_list=(0.0 0.0001 0.001 0.01 0.1 1 10 100 1000 0.0005 0.005 0.05 0.5 5)
# # beta_list=(0.0 0.0001 0.001 0.01 0.1 1 2 5 10 20 50 100 1000)
# # gamma_list=(0.0 0.0001 0.001 0.01 0.1 1 2 5 10 20 50 100 1000)
# ks=(1 2 4 8 16 24 32 46)
# orders=(1 2 4 8 16 24 32 46)
# nu_list=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.6 0.7)
# neighbors_list=(5 10 20 50)
# estimators_list=(5 10 20 50 100 200 500 1000)
# # estimators_list=(100)
# bandwidth_list=(0 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0)
# sigma_list=(1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 0.2 0.4 0.6 0.8 1)
# version_list=(v2)


# if [ "$orthogonal" = "True" ]; then
#     orthogonal_flag="--orthogonal"
# else
#     orthogonal_flag=""
# fi



# # for k in "${ks[@]}"
# # for order in "${orders[@]}"
# # for estimators in "${estimators_list[@]}"
# # for neighbors in "${neighbors_list[@]}"
# # for nu in "${nu_list[@]}"
# # for bandwidth in "${bandwidth_list[@]}"
# # for batch_size in "${batch_size_list[@]}"
# # for beta in "${beta_list[@]}"
# # for gamma in "${gamma_list[@]}"
# # for sigma in "${sigma_list[@]}"
# # for beta in "${beta_list[@]}"
# # for alpha in "${alpha_list[@]}"
# # for detector_type in "MD" "TV" "ZC" "pk-pk" "energy"
# # do
#     # for detector_type in "SPE" "T2" 
#     for m in "${m_list[@]}"
#     do
#         for version in "${version_list[@]}"
#         do
#         # Run the evaluation script with the selected configuration
#         python detector_evaluation.py \
#             --n $n \
#             --m $m \
#             --window_size $window_size \
#             --fraction $fraction \
#             --mode $mode \
#             --detector_type $detector_type \
#             --train_fraction $train_fraction \
#             --fs $fs \
#             $orthogonal_flag \
#             --source $source \
#             --seed_matrix $seed_matrix \
#             --seed_detector $seed_detector \
#             --processes $processes \
#             --gpu $gpu \
#             --factor $factor \
#             --alpha $alpha \
#             --min_lr $min_lr \
#             --min_delta $min_delta \
#             --patience $patience \
#             --epochs $epochs \
#             --lr $lr \
#             --batch_size $batch_size \
#             --threshold $threshold \
#             --optimizer $optimizer \
#             --detector_mode $detector_mode \
#             --k $k \
#             --order $order \
#             --kernel $kernel \
#             --kernel2 $kernel2 \
#             --nu $nu \
#             --neighbors $neighbors \
#             --estimators $estimators \
#             --sigma $sigma \
#             --bandwidth $bandwidth \
#             --version $version
#         done
#     done
# done
