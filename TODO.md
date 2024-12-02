## CS for anomaly detection

#### TSOC training
- 4 A matrices, 4 training seeds, bs={32, 64, 128} (fixed m=48, rakeness, ort=True)
- With the best bs: m={32, 64}
- In spare time m=16 and standard.

#### Detectors
- Script for hyperparameter tuning of detectors parameters 
- Notebook for results visualization of parameters tuning
- PCA likelihood detector (???)


## Unet-based CS

#### Unet training
- Train with:
    channel = {8, 16, 32, 64}
    steps = {2, 3, 4}
    kernel_size = {3, 5}
    seed = {0, 1, 2, 3, ...}
    
#### A selection    
- Update bestAbpdn to imrove speed
