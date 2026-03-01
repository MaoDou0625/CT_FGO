# Commands For Reproduction

## 1) Build and run unified multi-scheme comparison
python run_multi_scheme_compare.py --out-dir "temp_eval_6scheme_ctf_fix2"

## 2) Re-run unified evaluate only
python experiment_60_80.py evaluate --manifest "temp_eval_6scheme_ctf_fix2\manifest_multischemes.csv" --out-dir "temp_eval_6scheme_ctf_fix2" --baseline-run schemeA_ct_main_allwheel --segments 0-20,20-40,40-60,60-80,80-90 --target-segment 60-80
