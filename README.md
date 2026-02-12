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
