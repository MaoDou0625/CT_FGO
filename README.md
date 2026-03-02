# CT_FGO: GNSS/Multi-IMU Tightly-Coupled Continuous-Time Optimization Framework

**CT_FGO** is a state-of-the-art continuous-time state estimation framework designed for complex multi-sensor systems, specifically targeting vehicle platforms equipped with Global Navigation Satellite Systems (GNSS) and multiple Inertial Measurement Units (IMUs), including Wheel-mounted IMUs.

## 🚀 Key Features

*   **Continuous-Time Trajectory**: Powered by **B-Spline** on $SE(3)$, allowing asynchronous sensor fusion and analytical derivative computation (velocity/acceleration).
*   **Wheel IMU Specifics**:
    *   **Wheel Phase Estimation**: Treats wheel rotation phase as a continuous state variable, solving for the absolute rotation angle.
    *   **Static Gravity Alignment**: Automatically initializes wheel IMU orientation by aligning the Y-axis with the gravity vector during static periods.
    *   **Dynamic Misalignment Correction**: Estimates and corrects the cross-coupling (misalignment) between the wheel's spin axis (Z) and the radial/tangential axes (X/Y).
*   **Multi-Sensor Fusion**: Tightly couples GNSS position, Standard IMU (Body), and multiple Wheel IMUs.
*   **Online Calibration**: Jointly optimizes:
    *   IMU Biases (Accel/Gyro) modeled as Random Walk.
    *   Extrinsics (Body-to-Sensor rotation).
    *   Lever Arms (Body-to-Sensor position).
    *   Wheel Radius and Odometry Lever Arms.

## 🛠️ Dependencies

*   **CMake** (>= 3.10)
*   **Eigen3**
*   **Sophus** (Source build recommended for $SE(3)$ manifold support)
*   **Ceres Solver** (For non-linear least squares optimization)
*   **glog / gflags**
*   **yaml-cpp**

## 📦 Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

## 🏃 Run

```bash
./bin/ob_gins_ct config/ob_gins_ct.yaml
```

## ⚙️ Configuration (`config.yaml`)

### Wheel IMU Settings
Each wheel IMU is configured with specific parameters for the solver:

```yaml
center_imu4:
  type: "wheel"
  side: "right"             # 'left' or 'right' (auto-handles Z-axis sign)
  file: "path/to/data.txt"
  # ... standard imu noise params ...
  
  # Optimization Weights
  speed_weight: 10.0        # Weight for Wheel Speed Factor
  nhc_weight: 10.0          # Weight for Non-Holonomic Constraint
  attitude_weight_roll: 100.0  # Weight for Wheel Spin Factor (Roll)
  attitude_weight_yaw: 100.0   # Weight for Wheel Spin Factor (Pitch)
  
  # Priors
  priors:
    radius_std: 0.005       # Wheel radius prior std dev (m)
    lever_std: 0.02         # Lever arm prior std dev (m)
```

## 📊 Visualization & Analysis

### 1. Main Trajectory Analysis
Use `plot_ct_results.py` to compare the optimized trajectory against Ground Truth (GNSS/INS Reference).

```bash
python3 plot_ct_results.py --result output/ct_trajectory.txt --truth path/to/truth.txt
```

### 2. Wheel Phase Analysis
Use `plot_wheel_phase.py` to visualize the optimized continuous wheel rotation angle. This plot reveals the pure rotation of the wheel, decoupled from the vehicle's body motion.

```bash
python3 plot_wheel_phase.py --result output/ct_trajectory.txt
```
*Note: The script automatically looks for `errors_*.txt` files in the output directory.*

## 📐 Methodological Details

### Static Phase Alignment
The system automatically detects static periods (default: first 3 seconds) to compute the initial phase offset. It aligns the sensor's Y-axis with the gravity vector, ensuring a robust starting point for the optimization.

### Wheel Spin Gyro Factor
A custom factor (`WheelSpinGyroFactor`) constrains the relationship between the body angular velocity and the wheel IMU measurements. It includes a **misalignment model** ($k_x, k_y$) to compensate for the Z-axis rotation leaking into X/Y measurements due to imperfect mounting.

$$ \omega_{sensor}^{pred} = q_{bs}^{-1} \otimes \omega_{body} $$
$$ \omega_{meas}^{corr}.x = \omega_{meas}.x - k_x \cdot \omega_{meas}.z $$
$$ \omega_{meas}^{corr}.y = \omega_{meas}.y - k_y \cdot \omega_{meas}.z $$

Minimize: $ || \omega_{sensor}^{pred}.xy - \omega_{meas}^{corr}.xy ||^2 $

---
*Maintained by Xun Yi. Updated Feb 2026.*

## Data Format and YAML Guide (Hailaer / WID workflows)

This section documents the raw data structure, CT_FGO required text formats, and YAML templates used in recent experiments.

### 1) Original raw format (.mat)

The converted dataset file is `aligned_imu_gnss_rtk.mat`, with fields:

- `out.imu`, `out.gnss`, `out.rtk`
- `params.imu_cols`, `params.gnss_cols`, `params.rtk_cols`
- `params.imu_time_scale`, `params.gnss_time_scale`, `params.rtk_time_scale`
- `params.imu_gyro_unit`, `params.imu_acc_unit`

Typical `params` values:

- `imu_time_scale = 1e-4`
- `imu_gyro_unit = degph` (or `degps`)
- `imu_acc_unit = mps2` (or `g`)
- lon/lat in `deg`

### 2) CT_FGO required text format

#### 2.1 Main IMU (`Body_IMU.txt`)

7 columns:

`time dtheta_x dtheta_y dtheta_z dvel_x dvel_y dvel_z`

Notes:

- `time` in seconds.
- `dtheta` must be gyro increment in radians (not rate).
- `dvel` must be accel increment in m/s (not m/s^2).
- If raw gyro is `degph`: `omega_rad_s = gyro_degph * pi / 180 / 3600`, then `dtheta = omega_rad_s * dt`.
- If raw accel is `mps2`: `dvel = accel * dt`.

#### 2.2 GNSS (`GNSS_low.txt`)

Recommended 7 columns:

`time lat lon h std_lat std_lon std_h`

Minimal accepted format is 4 columns:

`time lat lon h`

Notes:

- `lat/lon` in degree.
- `h` in meter.

#### 2.3 Truth (`RTK_truth.txt`)

4 columns:

`time lat lon h`

Used for evaluation/plotting, not mandatory for solver runtime.

### 3) Wheel IMU text format (if enabled)

`ImuFileLoader` supports:

- 7 columns: no wheel odometry speed field.
- 8 columns: one wheel speed field.
- 9 columns: dual wheel speed fields (averaged).

First 7 columns are always:

`time dtheta_x dtheta_y dtheta_z dvel_x dvel_y dvel_z`

### 4) YAML templates

#### 4.1 CT_FGO main IMU + GNSS only (CT_D style)

```yaml
# ob_gins_ct_D.yaml
gnssfile: "D:/path/GNSS_low.txt"
outputpath: "D:/path/ct_D_output"
save_multi_imu: true

imu_main:
  type: "standard"
  file: "D:/path/Body_IMU.txt"
  columns: 7
  rate_hz: 1000
  antlever: [0.0, 0.0, 0.0]
  imunoise:
    accel_noise: 25
    gyro_noise: 0.004
    accel_bias_rw: 5
    gyro_bias_rw: 1
    accel_corr_time: 3600.0
    gyro_corr_time: 3600.0

starttime: 5476.3
endtime: 5789.7
aligntime: 3
kf_interval_sec: 0.1
num_iterations: 20
isearth: true

comparison:
  enable: false
```

#### 4.2 CT_FGO with wheel IMU(s)

```yaml
gnssfile: "D:/path/GNSS_low.txt"
outputpath: "D:/path/output"
save_multi_imu: true

imu_main:
  type: "standard"
  file: "D:/path/Body_IMU.txt"
  columns: 7
  rate_hz: 120
  antlever: [0.0, 0.0, 0.0]
  imunoise:
    accel_noise: 25
    gyro_noise: 0.004
    accel_bias_rw: 5
    gyro_bias_rw: 1
    accel_corr_time: 3600.0
    gyro_corr_time: 3600.0

center_imu4:
  type: "wheel"
  side: "right"
  file: "D:/path/wheel_imu4.txt"
  columns: 8
  rate_hz: 120
  antlever: [0.0, 0.0, 0.0]
  speed_weight: 10.0
  nhc_weight: 10.0
  attitude_weight_roll: 100.0
  attitude_weight_yaw: 100.0
  priors:
    radius_std: 0.005
    lever_std: 0.02

starttime: 1
endtime: 999999
aligntime: 3
kf_interval_sec: 0.1
num_iterations: 20
isearth: true
```

#### 4.3 KF-GINS YAML (for CT vs KF comparison)

```yaml
# kf-gins.yaml
imupath: "D:/path/Body_IMU.txt"
gnsspath: "D:/path/GNSS_low.txt"
outputpath: "D:/path/kf_output"
imudatalen: 7
imudatarate: 1000
starttime: 5476.3
endtime: 5789.7

initpos: [32.7574077000, 35.0221411000, 460.1400]
initvel: [0.0, 0.0, 0.0]
initatt: [0.0, 0.0, 0.0]

initgyrbias: [0, 0, 0]
initaccbias: [0, 0, 0]
initgyrscale: [0, 0, 0]
initaccscale: [0, 0, 0]

initposstd: [2.0, 2.0, 5.0]
initvelstd: [0.5, 0.5, 0.5]
initattstd: [5.0, 5.0, 30.0]

imunoise:
  arw: [0.24, 0.24, 0.24]
  vrw: [0.24, 0.24, 0.24]
  gbstd: [50.0, 50.0, 50.0]
  abstd: [250.0, 250.0, 250.0]
  gsstd: [1000.0, 1000.0, 1000.0]
  asstd: [1000.0, 1000.0, 1000.0]
  corrtime: 1.0

antlever: [0.0, 0.0, 0.0]
```

### 5) Reproducible scripts added in this repo

- `tools/export_hailaer_mat_to_txt.m`: batch export `.mat -> txt` with scientific notation.
- `tools/run_hailaer_ctd_kf.ps1`: run 7 datasets for CT_D and KF-GINS.
- `tools/eval_nav_rmse.py`: evaluate CT/KF RMSE against RTK truth.

Default output root used in recent runs:

`D:/Code/dataset/hailaer/inertail/ctd_kf_compare_20260301`
