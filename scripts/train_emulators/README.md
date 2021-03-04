# Train Emulators
<!-- This file is auto-generated. Do not edit directly.You can regenerate using python train_emulators.py --bypass-training -->
The `train_emulators.py` script will train emulators and create this report.
## Reizman Suzuki Cross coupling 
This is the data from training of the reizman suzuki benchmark for 1000 epochs with 5 cross-validation folds.
| case   |   avg_fit_time |   avg_val_r2 |   avg_val_RMSE |   avg_test_r2 |   avg_test_RMSE |
|:-------|---------------:|-------------:|---------------:|--------------:|----------------:|
| case_1 |           9.68 |         0.82 |          10.84 |          0.93 |            7.64 |
| case_2 |           9.16 |         0.57 |           5.65 |          0.66 |            5.04 |
| case_3 |           8.83 |         0.74 |          13.42 |          0.83 |           12.32 |
| case_4 |           9.08 |         0.69 |          16.31 |          0.75 |           13.57 |
## Baumgartner C-N Cross Cross Coupling 
This is the data from training of the Baumgartner C-N aniline cross-coupling benchmark for 1000 epochs with 5 cross-validation folds.
| case        |   avg_fit_time |   avg_val_r2 |   avg_val_RMSE |   avg_test_r2 |   avg_test_RMSE |
|:------------|---------------:|-------------:|---------------:|--------------:|----------------:|
| one-hot     |           9.33 |         0.81 |           0.18 |          0.87 |            0.14 |
| descriptors |           8.91 |        -0.09 |           0.43 |         -0.13 |            0.41 |
