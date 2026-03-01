# Multi-Scheme Unified Workflow (all trials)

## 1. Run CT_FGO schemes (A/B/C/F)

```powershell
.\bin\Release\ob_gins_ct.exe config\ob_gins_ct_schemeA_main_allwheel.yaml
.\bin\Release\ob_gins_ct.exe config\ob_gins_ct_schemeB_main_rear_right.yaml
.\bin\Release\ob_gins_ct.exe config\ob_gins_ct_schemeC_main_front_left.yaml
.\bin\Release\ob_gins_ct.exe config\ob_gins_ct_schemeF_main_only.yaml
```

## 2. Run KF-GINS scheme (D)

Use KF-GINS repo executable with its YAML:

```powershell
D:\Code\KF-GINS\bin\Release\KF-GINS.exe D:\Code\dataset\WID\Datasets\transformedData2\kf_gins_D_mainimu_trial01_direct_20260228_01\kf-gins-trial01-direct.yaml
```

## 3. Run Wheel-GINS scheme (E)

Use Wheel-GINS executable and config (produces `traj.txt`).

## 4. Unified conversion + evaluation + report (all trials)

```powershell
python run_multi_scheme_compare.py --trials all --out-dir D:\Code\dataset\WID\Datasets\transformedData2\five_scheme_eval_20260301
```

If your multi-trial outputs are stored with explicit `{trial}` folders, use templates:

```powershell
python run_multi_scheme_compare.py --trials all `
  --scheme-a-ct-template "D:/Code/dataset/WID/Datasets/transformedData2/output_schemeA_main_allwheel_20260228/{trial}/ct_trajectory.txt" `
  --scheme-b-ct-template "D:/Code/dataset/WID/Datasets/transformedData2/output_subtask3_schemeB_main_plus_rear_right/{trial}/ct_trajectory.txt" `
  --scheme-c-ct-template "D:/Code/dataset/WID/Datasets/transformedData2/output_scheme_c_front_left_20260228/{trial}/ct_trajectory.txt" `
  --scheme-f-ct-template "D:/Code/dataset/WID/Datasets/transformedData2/output_schemeF_main_only_20260301/{trial}/ct_trajectory.txt" `
  --scheme-d-kf-nav-template "D:/Code/dataset/WID/Datasets/transformedData2/kf_gins_D_mainimu_{trial}_direct_20260228_01/KF_GINS_Navresult.nav" `
  --scheme-e-wheel-traj-template "D:/Code/dataset/WID/Datasets/transformedData2/wheel_gins_single_rear2_{trial}_20260228_2145/traj.txt" `
  --out-dir D:/Code/dataset/WID/Datasets/transformedData2/five_scheme_eval_20260301
```

Outputs:
- `D:\Code\dataset\WID\Datasets\transformedData2\five_scheme_eval_20260301\manifest_multischemes.csv`
- `D:\Code\dataset\WID\Datasets\transformedData2\five_scheme_eval_20260301\summary_metrics.csv`
- `D:\Code\dataset\WID\Datasets\transformedData2\five_scheme_eval_20260301\gate_results.csv`
- `D:\Code\dataset\WID\Datasets\transformedData2\five_scheme_eval_20260301\compare_60_80.png`
- `D:\Code\dataset\WID\Datasets\transformedData2\five_scheme_eval_20260301\compare_global_weighted.png`
- `D:\Code\dataset\WID\Datasets\transformedData2\five_scheme_eval_20260301\FINAL_REPORT_5SCHEMES.md`

Converted files (required transformedfor***):
- `D:\Code\dataset\WID\Datasets\transformedData2\transformedforKF_GINS_schemeD_trial01\ct_trajectory.txt` ... `trial12`
- `D:\Code\dataset\WID\Datasets\transformedData2\transformedforWheel_GINS_schemeE_trial01\ct_trajectory.txt` ... `trial12`
