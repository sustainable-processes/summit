# Train Emulators
<!-- This file is auto-generated. Do not edit directly.You can regenerate using python train_emulators.py --bypass-training -->
The `train_emulators.py` script will train emulators and create this report.
## Reizman Suzuki Cross coupling 
This is the data from training of the reizman suzuki benchmark for 1000 epochs with 5 cross-validation folds.
| case   |   avg_fit_time |   avg_val_r2 |   avg_val_RMSE |   avg_test_r2 |   avg_test_RMSE |
|:-------|---------------:|-------------:|---------------:|--------------:|----------------:|
| case_1 |           8.63 |         0.82 |          11.14 |          0.93 |            7.54 |
| case_2 |           8.8  |         0.61 |           5.38 |          0.66 |            4.99 |
| case_3 |           8.24 |         0.78 |          12.91 |          0.84 |           11.9  |
| case_4 |           8.31 |         0.7  |          15.67 |          0.73 |           14.06 |
## Baumgartner C-N Cross Cross Coupling 
This is the data from training of the Baumgartner C-N aniline cross-coupling benchmark for 1000 epochs with 5 cross-validation folds.
| case        |   avg_fit_time |   avg_val_r2 |   avg_val_RMSE |   avg_test_r2 |   avg_test_RMSE |
|:------------|---------------:|-------------:|---------------:|--------------:|----------------:|
| one-hot     |           8.17 |         0.8  |           0.18 |          0.89 |            0.13 |
| descriptors |           8.19 |         0.86 |           0.15 |          0.91 |            0.11 |
