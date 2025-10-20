# ###############################################################################
# Autoencoder Training for Multiple Configurations                              #
# ###############################################################################

# # Configuration Section
# n=128               # Number of samples per signal
# isnr=35             # Signal-to-noise ratio (SNR)

# m=16                # Number of measurements
# # m_list=(16 32 48 64)
# m_list=(32)
# # seed_training=1     # Training-related random seed for reproducibility
# # seed_training_list=(0 1 2 3)
# seed_training_list=(0)
# gpu=5               # GPU index

# coarse="lstsq"    # lstsq, pinv, svd, transpose, learn

# # optimizer="sgd"    # Optimizer for training
# # lr=0.1              # Learning rate
# # batch_size=50       # Batch size for training
# # min_lr=0.001        # Minimum learning rate

# beta=0.0            # Latent volume regularization weight
# gamma=0.0         # Latent reconstruction regularization weight
# eta=0.0            # Sensing matrix regularization weight
# # gamma_list=(0.001 0.01 0.1 0.0001)
# # gamma_list=(1 10 100 1000)
# # beta_list=(0.0001 0.001 0.01 0.1 1 2 5 10 20 50 100 1000)
# # gamma_list=(0.0001 0.001 0.01 0.1 1 2 5 10 20 50 100 1000)
# # gamma_list=(0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000)
# # beta_list=(0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000)
# beta_list=(0.00005 0.0005 0.005 0.05 0.5 5 0.00002 0.0002 0.002 0.02 0.2 2)
# gamma_list=(0.00005 0.0005 0.005 0.05 0.5 5 0.00002 0.0002 0.002 0.02 0.2 2)
# # beta_list=(0.0002 0.002 0.02 0.2 2)
# # beta_list=(0.001 0.01 0.1 0.0001)

# optimizer="adam"    # Optimizer for training
# lr=0.001            # Learning rate
# batch_size=128      # Batch size for training
# min_lr=0.00001      # Minimum learning rate

# train_fraction=0.9  # Fraction of data used for training
# factor=0.2          # Factor for ReduceLROnPlateau scheduler

# min_delta=1e-8      # Minimum delta for early stopping and ReduceLROnPlateau
# patience=40         # Patience for early stopping
# epochs=1000         # Number of training epochs

# N=2000000           # Number of training instances
# fs=256              # Sampling frequency
# heart_rate="60 100" # Heart rate range
# processes=48        # Number of CPU processes for parallelism

# for seed_training in "${seed_training_list[@]}"
# do
#     # for gamma in "${gamma_list[@]}"
#     # for beta in "${beta_list[@]}"
#     # do
#         for m in "${m_list[@]}"
#         do

#             # Run the training script with the selected configuration
#             # python ae_unet_tiny_training.py \
#             python ae_unet_training.py \
#                 --n $n \
#                 --m $m \
#                 --epochs $epochs \
#                 --lr $lr \
#                 --optimizer $optimizer \
#                 --coarse $coarse \
#                 --beta $beta \
#                 --gamma $gamma \
#                 --eta $eta \
#                 --batch_size $batch_size \
#                 --N $N \
#                 --fs $fs \
#                 --heart_rate $heart_rate \
#                 --isnr $isnr \
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



# ###############################################################################
# Autoencoder Detection Evaluation                                              #
# ###############################################################################

# # Configuration Section
# n=128               # Number of samples per signal
# isnr=35             # Signal-to-noise ratio (SNR)

# m=16                # Number of measurements
# # m_list=(16 32 48 64)
# m_list=(32)
# # seed_training=1     # Training-related random seed for reproducibility
# # seed_training_list=(0 1 2 3)
# seed_training_list=(0)
# gpu=1               # GPU index

# coarse="transpose"    # lstsq, pinv, svd, transpose, learn

# # optimizer="sgd"    # Optimizer for training
# # lr=0.1              # Learning rate
# # batch_size=50       # Batch size for training
# # min_lr=0.001        # Minimum learning rate

# beta=0.0            # Latent volume regularization weight
# gamma=0.0         # Latent reconstruction regularization weight
# eta=0.0            # Sensing matrix regularization weight
# # gamma_list=(0.001 0.01 0.1 0.0001)
# # gamma_list=(1 10 100 1000)
# # beta_list=(0.0001 0.001 0.01 0.1 1 2 5 10 20 50 100 1000)
# # gamma_list=(0.0001 0.001 0.01 0.1 1 2 5 10 20 50 100 1000)
# # gamma_list=(0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000)
# # beta_list=(0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000)
# beta_list=(0.0 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 0.00005 0.0005 0.005 0.05 0.5 5 0.00002 0.0002 0.002 0.02 0.2 2)
# gamma_list=(0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 0.00005 0.0005 0.005 0.05 0.5 5 0.00002 0.0002 0.002 0.02 0.2 2)
# # beta_list=(0.0002 0.002 0.02 0.2 2)
# # beta_list=(0.001 0.01 0.1 0.0001)

# delta_list=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.8)

# num_iter=1
# num_iter_list=(2 3 4 5 10 20)

# seed_ko=0
# optimizer="adam"    # Optimizer for training
# lr=0.001            # Learning rate
# batch_size=128      # Batch size for training
# min_lr=0.00001      # Minimum learning rate

# train_fraction=0.9  # Fraction of data used for training
# factor=0.2          # Factor for ReduceLROnPlateau scheduler

# min_delta=1e-8      # Minimum delta for early stopping and ReduceLROnPlateau
# patience=40         # Patience for early stopping
# epochs=1000         # Number of training epochs

# N_train=2000000           # Number of training instances
# N_test=10000           # Number of training instances
# fs=256              # Sampling frequency
# heart_rate="60 100" # Heart rate range
# processes=48        # Number of CPU processes for parallelism

# for seed_training in "${seed_training_list[@]}"
# do
#     # for num_iter in "${num_iter_list[@]}"
#     # do 
#         for delta in "${delta_list[@]}"
#         do
#             # for gamma in "${gamma_list[@]}"
#             # for beta in "${beta_list[@]}"
#             # do
#                 for m in "${m_list[@]}"
#                 do

#                     # Run the training script with the selected configuration
#                     # python ae_unet_tiny_training.py \
#                     python ae_unet_detection.py \
#                         --n $n \
#                         --m $m \
#                         --epochs $epochs \
#                         --lr $lr \
#                         --optimizer $optimizer \
#                         --coarse $coarse \
#                         --beta $beta \
#                         --gamma $gamma \
#                         --eta $eta \
#                         --batch_size $batch_size \
#                         --N_train $N_train \
#                         --N_test $N_test \
#                         --fs $fs \
#                         --heart_rate $heart_rate \
#                         --isnr $isnr \
#                         --seed_training $seed_training \
#                         --processes $processes \
#                         --gpu $gpu \
#                         --train_fraction $train_fraction \
#                         --factor $factor \
#                         --min_lr $min_lr \
#                         --min_delta $min_delta \
#                         --patience $patience \
#                         --delta $delta \
#                         --seed_ko $seed_ko \
#                         --num_iter $num_iter
#                 done
#             done
#         done
#     done
# done

# ###############################################################################
# PCA Detection Evaluation                                                      #
# ###############################################################################

# Configuration Section
n=128               # Number of samples per signal
isnr=35             # Signal-to-noise ratio (SNR)

m=16                # Number of measurements
# m_list=(16 32 48 64)
m_list=(16)

delta_list=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.8)

seed_ko=0

N_train=2000000           # Number of training instances
N_test=10000           # Number of testing instances
fs=256              # Sampling frequency
heart_rate="60 100" # Heart rate range

for delta in "${delta_list[@]}"
do
    for m in "${m_list[@]}"
    do

        # Run the evaluation script with the selected configuration
        python pca_detection.py \
            --n $n \
            --m $m \
            --N_train $N_train \
            --N_test $N_test \
            --fs $fs \
            --heart_rate $heart_rate \
            --isnr $isnr \
            --delta $delta \
            --seed_ko $seed_ko
        
    done
done