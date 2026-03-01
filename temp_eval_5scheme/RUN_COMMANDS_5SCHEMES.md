# Commands For Reproduction

## 1) Build and run unified five-scheme comparison
python run_five_scheme_compare.py --out-dir "temp_eval_5scheme"

## 2) Re-run unified evaluate only
python experiment_60_80.py evaluate --manifest "temp_eval_5scheme\manifest_5schemes.csv" --out-dir "temp_eval_5scheme" --baseline-run schemeA_ct_main_allwheel --segments 0-20,20-40,40-60,60-80,80-90 --target-segment 60-80
