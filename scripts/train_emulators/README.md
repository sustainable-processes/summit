# Train Emulators
<!-- This file is auto-generated. Do not edit directly.You can regenerate using python train_emulators.py --bypass-training -->
The `train_emulators.py` script will train emulators and create this report.
## Reizman Suzuki Cross coupling 
This is the data from training of the reizman suzuki benchmark for 1000 epochs with 5 cross-validation folds.
| case   |   avg_fit_time |   avg_val_r2 |   avg_val_RMSE |   avg_test_r2 |   avg_test_RMSE |
|:-------|---------------:|-------------:|---------------:|--------------:|----------------:|
| case_1 |           3.52 |         0.81 |          11.21 |          0.93 |            7.66 |
| case_2 |           3.58 |         0.59 |           5.54 |          0.67 |            4.91 |
| case_3 |           3.61 |         0.76 |          13.11 |          0.84 |           12.04 |
| case_4 |           3.6  |         0.7  |          15.99 |          0.74 |           13.8  |
## Baumgartner C-N Cross Cross Coupling 
This is the data from training of the Baumgartner C-N aniline cross-coupling benchmark for 1000 epochs with 5 cross-validation folds.
| case        |   avg_fit_time |   avg_val_r2 |   avg_val_RMSE |   avg_test_r2 |   avg_test_RMSE |
|:------------|---------------:|-------------:|---------------:|--------------:|----------------:|
| one-hot     |           3.54 |         0.79 |           0.18 |          0.88 |            0.13 |
| descriptors |           3.52 |         0.84 |           0.16 |          0.9  |            0.12 |
