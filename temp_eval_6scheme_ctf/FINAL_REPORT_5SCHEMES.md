# Multi-Scheme Unified Comparison Report

## Inputs
- Manifest: `temp_eval_6scheme_ctf\manifest_multischemes.csv`
- Converted KF-GINS result: `D:\Code\dataset\WID\Datasets\transformedData2\transformedforKF_GINS_schemeD_trial01\ct_trajectory.txt`
- Converted Wheel-GINS result: `D:\Code\dataset\WID\Datasets\transformedData2\transformedforWheel_GINS_schemeE_trial01\ct_trajectory.txt`

## Results
- schemeA_ct_main_allwheel: pos60-80=0.0358 m, speed60-80=0.1228 m/s, posGlobalW=0.2294 m, overall_pass=False
- schemeB_ct_main_rear_right: pos60-80=0.0358 m, speed60-80=0.1226 m/s, posGlobalW=0.2234 m, overall_pass=False
- schemeC_ct_main_front_left: pos60-80=0.0359 m, speed60-80=0.1231 m/s, posGlobalW=0.2423 m, overall_pass=False
- schemeF_ct_main_only: pos60-80=0.3367 m, speed60-80=3.3797 m/s, posGlobalW=0.2847 m, overall_pass=False
- schemeD_kf_gins_main_imu: pos60-80=1.2639 m, speed60-80=1.4646 m/s, posGlobalW=1.7811 m, overall_pass=False
- schemeE_wheel_gins_main_rear: pos60-80=0.8794 m, speed60-80=0.4410 m/s, posGlobalW=1.9525 m, overall_pass=False

## Files
- `summary_metrics.csv`, `gate_results.csv`, `report.md` from unified evaluator
- `compare_60_80.png`, `compare_global_weighted.png`
- `detailed_plots/all_schemes_2d_vs_truth.png`
- `detailed_plots/all_schemes_enu_vs_truth.png`
- `detailed_plots/per_scheme/*_2d_vs_truth.png`
- `detailed_plots/per_scheme/*_enu_vs_truth.png`

## Notes
- Unified segments: `0-20,20-40,40-60,60-80,80-90`
- Baseline: `schemeA_ct_main_allwheel`
