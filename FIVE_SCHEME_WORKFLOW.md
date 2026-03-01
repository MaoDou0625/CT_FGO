# Five-Scheme Unified Workflow (trial01)

## 1. Run CT_FGO schemes (A/B/C)

```powershell
.\bin\Release\ob_gins_ct.exe config\ob_gins_ct_schemeA_main_allwheel.yaml
.\bin\Release\ob_gins_ct.exe config\ob_gins_ct_schemeB_main_rear_right.yaml
.\bin\Release\ob_gins_ct.exe config\ob_gins_ct_schemeC_main_front_left.yaml
```

## 2. Run KF-GINS scheme (D)

Use KF-GINS repo executable with its YAML:

```powershell
D:\Code\KF-GINS\bin\Release\KF-GINS.exe D:\Code\dataset\WID\Datasets\transformedData2\kf_gins_D_mainimu_trial01_direct_20260228_01\kf-gins-trial01-direct.yaml
```

## 3. Run Wheel-GINS scheme (E)

Use Wheel-GINS executable and config (produces `traj.txt`).

## 4. Unified conversion + evaluation + report

```powershell
python run_five_scheme_compare.py --out-dir temp_eval_5scheme
```

Outputs:
- `temp_eval_5scheme/manifest_5schemes.csv`
- `temp_eval_5scheme/summary_metrics.csv`
- `temp_eval_5scheme/gate_results.csv`
- `temp_eval_5scheme/compare_60_80.png`
- `temp_eval_5scheme/compare_global_weighted.png`
- `temp_eval_5scheme/FINAL_REPORT_5SCHEMES.md`

Converted files (required transformedfor***):
- `D:\Code\dataset\WID\Datasets\transformedData2\transformedforKF_GINS_schemeD_trial01\ct_trajectory.txt`
- `D:\Code\dataset\WID\Datasets\transformedData2\transformedforWheel_GINS_schemeE_trial01\ct_trajectory.txt`
