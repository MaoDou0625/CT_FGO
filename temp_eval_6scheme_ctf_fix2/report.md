# 60-80s Experiment Evaluation

- Baseline: `schemeA_ct_main_allwheel`
- Target segment: `60-80`
- Gate: improve >= 15.0%, non-target degrade <= 5.0%, speed degrade <= 5.0%

## Runs
- `schemeA_ct_main_allwheel` | pos60-80=0.0358m | speed60-80=0.1228m/s | term=NO_LOG | pass=False
- `schemeB_ct_main_rear_right` | pos60-80=0.0358m | speed60-80=0.1226m/s | term=NO_LOG | pass=False
- `schemeC_ct_main_front_left` | pos60-80=0.0359m | speed60-80=0.1231m/s | term=NO_LOG | pass=False
- `schemeF_ct_main_only` | pos60-80=0.1042m | speed60-80=0.1333m/s | term=NO_LOG | pass=False
- `schemeD_kf_gins_main_imu` | pos60-80=1.2639m | speed60-80=1.4646m/s | term=NO_LOG | pass=False
- `schemeE_wheel_gins_main_rear` | pos60-80=0.8794m | speed60-80=0.4410m/s | term=NO_LOG | pass=False
