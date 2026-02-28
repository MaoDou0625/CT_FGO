# 60-80s Experiment Evaluation

- Baseline: `baseline`
- Target segment: `60-80`
- Gate: improve >= 15.0%, non-target degrade <= 5.0%, speed degrade <= 5.0%

## Runs
- `baseline` | pos60-80=0.3381m | speed60-80=3.6817m/s | term=NO_LOG | pass=False
- `output2` | pos60-80=0.0303m | speed60-80=0.1916m/s | term=NO_LOG | pass=True
- `output3` | pos60-80=0.0351m | speed60-80=0.1234m/s | term=NO_LOG | pass=False
- `output4` | pos60-80=0.0351m | speed60-80=0.1241m/s | term=NO_LOG | pass=False
- `output_bottest` | pos60-80=0.0356m | speed60-80=0.1227m/s | term=NO_LOG | pass=True
