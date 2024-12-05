## Environment
- Generate a .yml file for environment with minimal requirements quick setup.

### Sparse Dataset
- Look for a real-world sparse dataset.


## CS for anomaly detection

#### TSOC training
- 4 A matrices, 4 training seeds, bs={32, 64, 128} (fixed m=48, rakeness, ort=True)
- With the best bs: m={32, 64}. How to define the best batch size?
- 
- Check if TSOC trains better with the dataset used in JETCAS
- In spare time m=16 and standard.

#### Autoencoder
- Train an AE with the same architecture as TSOC

#### Detectors
- Train and test non machine learning-based detectors
- 
- PCA likelihood detector (???)
- Check if the seed works for the generation of anomalies

## Unet-based CS

#### Unet training
- Train with:
    channel = {8, 16, 32, 64}
    steps = {2, 3, 4}
    kernel_size = {3, 5}
    seed = {0, 1, 2, 3, ...}

#### A selection    
- Update bestAbpdn to improve speed

