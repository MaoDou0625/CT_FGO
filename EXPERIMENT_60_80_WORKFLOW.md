# 60-80s 精度改进实验实施流程（可复现）

本流程对应 `EXPERIMENT_PLAN_60_80.md` 的分阶段执行，目标是：
- 提升 `60-80s` 过渡段位置/速度精度；
- 非目标时段不明显退化；
- 全流程可复现、可回放、可对比。

## 0. 统一入口

统一脚本：`experiment_60_80.py`

包含 3 个子命令：
- `gen-grid`：生成参数扫描配置（P1-1）
- `run`：批量运行求解器（P0/P1）
- `evaluate`：统一分段评估 + 基线对比 + 验收门槛判定（P0-1）

## 1. P0-1 分段评估自动化（先做）

若你已经有多组实验结果（`ct_trajectory.txt`），先准备 `manifest.csv`：

```csv
run_name,config_path,result_path,truth_path,log_path,status,runtime_sec,return_code
baseline,config/ob_gins_ct.yaml,D:/.../baseline/ct_trajectory.txt,D:/.../GNSS_use.txt,D:/.../baseline.log,DONE,12.3,0
trial_a,config/trial_a.yaml,D:/.../trial_a/ct_trajectory.txt,D:/.../GNSS_use.txt,D:/.../trial_a.log,DONE,13.1,0
```

执行评估：

```bash
python experiment_60_80.py evaluate \
  --manifest D:/exp60_80/manifest.csv \
  --out-dir D:/exp60_80/eval \
  --baseline-run baseline
```

输出：
- `summary_metrics.csv`：每个 run 的 60-80 与全局加权指标
- `gate_results.csv`：门槛通过/失败明细
- `report.md`：可读总结
- `per_run_segments/*.csv`：每个 run 的分段详细指标

默认门槛（与计划一致）：
- `60-80s Pos2D_RMSE` 改善 `>= 15%`
- 其他分段 `Pos2D_RMSE` 退化 `<= 5%`
- `60-80s Speed2D_RMSE` 退化 `< 5%`

## 2. P1-1 权重敏感性扫描（先粗后细）

生成网格配置 + 清单：

```bash
python experiment_60_80.py gen-grid \
  --base-config config/ob_gins_ct.yaml \
  --out-dir D:/exp60_80/scan_round1 \
  --truth-path D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/GNSS_use.txt \
  --speed-values 8,10,12 \
  --nhc-values 8,10,12 \
  --roll-values 80,100,120 \
  --yaw-front-values 0.5,1,2 \
  --yaw-rear-values 60,100,140
```

随后批跑：

```bash
python experiment_60_80.py run \
  --manifest D:/exp60_80/scan_round1/manifest.csv \
  --executable D:/path/to/ob_gins_ct.exe \
  --only-pending
```

再统一评估：

```bash
python experiment_60_80.py evaluate \
  --manifest D:/exp60_80/scan_round1/manifest.csv \
  --out-dir D:/exp60_80/scan_round1/eval \
  --baseline-run baseline
```

说明：
- 生成配置时会改写每个 run 的 `outputpath` 到独立目录，避免结果覆盖。
- `center_imu1/3` 使用前轮 yaw 权重，`center_imu2/4` 使用后轮 yaw 权重。

## 3. P0-2 收敛/失败诊断

`evaluate` 会自动解析每个 run 的日志：
- `termination`（若日志包含 `Termination: ...`）
- `NO_CONVERGENCE` 出现次数

可直接用于筛除不稳定参数区间。

## 4. 推荐执行顺序

1. 固定当前基线并完成 `P0-1` 自动评估
2. 进行 `P1-1` 粗网格扫描并筛掉不稳定区
3. 在候选区间做细网格扫描
4. 对通过门槛的方案再做 `P0-2` 诊断复核

