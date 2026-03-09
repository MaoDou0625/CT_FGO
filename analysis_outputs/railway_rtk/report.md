# Railway RTK Analysis

Source README: https://github.com/ETH-PBL/Railway-Precise-Localization/blob/main/README.md

## Key Findings

- Runs analyzed: 12
- Total RTK points (float + fix): 20984
- RTK fix points: 9743
- RTK float points: 11241
- IMU average rate across runs: 1688.8 Hz
- IMU median-dt implied rate across runs: 1776.2 Hz
- NAV-PVT average rate across runs: 1.00 Hz
- NAV-COV average rate across runs: 1.00 Hz
- Largest RTK altitude step: 22.54 m in 20221020_1217_R90121_Modena_SN2

## Repeatability

- Formigine: pairwise median nearest-neighbor distance 3.12 m, pairwise p95 566.69 m
- Modena: pairwise median nearest-neighbor distance 3.70 m, pairwise p95 23.57 m

## Output Files

- rtk_repeatability_map.png
- rtk_altitude_panels.png
- run_summary.csv
- repeatability_pairwise.csv
- raw_format_summary.csv
- rtk_points.csv
