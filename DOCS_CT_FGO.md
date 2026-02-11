# CT_FGO: GNSS/多IMU融合的连续时间因子图优化

## 1. 简介

**CT_FGO** (Continuous-Time Factor Graph Optimization) 是一个专为复杂多传感器系统设计的状态估计框架，主要针对搭载全球导航卫星系统 (GNSS) 和多个惯性测量单元 (IMU) 的载具平台。

与传统的离散时间滤波 (EKF) 或离散优化不同，CT_FGO 采用基于 B-Spline（B样条）的 **连续时间轨迹表示**。这带来了以下优势：
- **异步传感器融合**：可以无缝集成来自多个 IMU 和 GNSS 接收机的测量数据，无需担心它们采样率不同或时间不同步。
- **高频状态查询**：可以查询优化窗口内任意时刻的位姿、速度和加速度。
- **解析导数**：速度和加速度可以直接通过样条曲线的时间导数解析计算，精度更高。

该系统特别针对 "Wheel IMU"（轮端 IMU）配置进行了优化，即安装在车轮上的 IMU，可直接测量车轮转动和动力学特性，辅助里程计和滑移估计。

---

## 2. 系统架构

核心架构包含以下组件：

### 2.1. 轨迹表示 (B-Spline)
载具的轨迹（位置和姿态）由 $SE(3)$ 流形上的 **累积 B-Spline** 参数化。
- **控制点 (Control Points)**：轨迹由一系列时间间隔均匀（例如 0.1s）的控制点定义。
- **状态求值**：任意时刻 $t$ 的状态通过 `BSplineEvaluator` 对最近的 4 个控制点进行插值计算。
- **状态变量**：
    - **位姿**：$T_{wb} \in SE(3)$ (世界系到体系)
    - **速度/加速度**：由样条时间导数解析得出。

### 2.2. 传感器处理器 (`ImuProcessor`)
系统使用工厂模式处理不同类型的 IMU：
- **`StandardImuProcessor`**：处理车身主 IMU。添加标准的惯性因子（加速度和角速度约束），将轨迹与原始 IMU 测量值关联。
- **`WheelImuProcessor`**：处理安装在车轮上的 IMU。它集成了 "轮端机械编排 (Wheel Mechanization)" 逻辑，并添加专用因子：
    - **轮速因子 (Wheel Speed Factor)**：通过陀螺仪测量的车轮转速推算车辆速度（依赖轮径）。
    - **轮端姿态因子 (Wheel Attitude Factor)**：约束车轮相对于车身的姿态（如果可观测）。

### 2.3. 轮端姿态因子 (Wheel Attitude Factor) 详解
这是本系统处理轮端 IMU 的核心创新点之一，用于解决轮端低成本 IMU 姿态漂移快、难以长期积分的问题。

**1. 基本原理**
轮子作为车辆的一个组件，其运动并非完全自由的刚体运动。相对于车身（Body），轮子受机械结构限制，通常只有 **1 个自由度**：绕轮轴的旋转。
在本系统的默认配置中，假设轮端 IMU 的 **Z 轴** 沿轮轴方向安装（即感应轮速），因此轮子相对于车身的安装坐标系，**只有 Z 轴（轮轴）在转动，X 轴和 Y 轴是锁定的**。
**Wheel Attitude Factor** 利用这一几何约束，将轮端 IMU 积分得到的姿态，与车身主 IMU（或轨迹）解算出的姿态进行“软绑定”。

**2. 数学模型**
设：
- $q_{wb}$：世界系到车身系的旋转（待优化状态，由 B-Spline 计算）。
- $q_{sw}$：世界系到轮端传感器系的旋转（由轮端 IMU 陀螺仪积分得到，作为测量输入）。
- $q_{sb}$：车身系到轮端传感器系的外参（待优化参数或固定值）。

我们计算轮子相对于安装座的“残差旋转” $q_{diff}$：
$$ q_{diff} = q_{sb}^{-1} \otimes (q_{wb}^{-1} \otimes q_{sw}) $$

理论上，如果轮子只绕 Z 轴转动，那么 $q_{diff}$ 应该是一个纯粹的 Z 轴旋转四元数：
$$ q_{diff}^{ideal} = [\cos(\theta/2), 0, 0, \sin(\theta/2)]^T $$
即其虚部的 $x$ 分量和 $y$ 分量应为 **0**。

**残差函数 (Residuals)：**
$$ r_{roll} = w_{roll} \cdot q_{diff}.x $$
$$ r_{pitch} = w_{pitch} \cdot q_{diff}.y $$

**3. 作用与意义**
1.  **消除漂移**：轮端 IMU 的陀螺仪（尤其是 MEMS 级）会有零偏和随机游走。如果没有约束，积分出来的 $q_{sw}$ 会迅速漂移。此因子强制轮子的非旋转轴跟随车身，从而**极大限制了轮端姿态的漂移**。
2.  **双向约束**：
    - **IMU -> Body**：当轮端 IMU 姿态 $q_{sw}$ 较准时（短期内），它通过约束告诉系统：“我的 X/Y 轴没有转，所以车身的 X/Y 轴也不应该转”，从而辅助约束车身姿态。
    - **Body -> IMU**：当车身姿态 $q_{wb}$ 由 GNSS 或主 IMU 确定时，它反过来校正轮端 IMU 的积分漂移，防止 $q_{sw}$ 跑偏。
3.  **允许自转**：它完全不约束 Z 轴（Yaw），允许轮子以任意速度、任意角度转动。

### 2.4. 因子图 (Ceres Solver)
优化问题被建模为使用 **Ceres Solver** 求解的非线性最小二乘问题。图包含：
- **连续 GNSS 因子**：约束全局位置。
- **连续惯性因子**：约束轨迹导数（速度、加速度）以匹配 IMU 的加计/陀螺仪读数。
- **轮速因子**：基于车轮转动约束前向速度。
- **零偏随机游走因子**：对 IMU 零偏随时间的漂移进行建模。
- **先验因子**：对静态参数（外参、轮径）的约束，防止漂移。

---

## 3. 安装与编译

### 3.1. 环境要求
系统主要在 **Ubuntu 20.04/22.04** 下测试。

**依赖库：**
- **CMake** (>= 3.10)
- **Eigen3** (线性代数)
- **Sophus** (李代数库，**必须源码编译**)
- **Ceres Solver** (优化库)
- **glog / gflags** (日志与参数解析)
- **yaml-cpp** (配置文件解析)
- **abseil-cpp** (包含在 `src/thirdparty` 中)

### 3.2. 编译指南

1.  **克隆仓库**：
    ```bash
    git clone https://github.com/MaoDou0625/CT_FGO.git
    cd CT_FGO
    ```

2.  **安装基础依赖** (Ubuntu)：
    ```bash
    sudo apt update
    sudo apt install -y cmake build-essential libeigen3-dev libyaml-cpp-dev libceres-dev libgoogle-glog-dev libgflags-dev
    ```

3.  **安装 Sophus** (关键步骤)：
    请勿使用 `apt install ros-*-sophus`，版本通常不兼容。
    ```bash
    cd ~
    git clone https://github.com/strasdat/Sophus.git
    cd Sophus
    mkdir build && cd build
    cmake ..
    make -j4
    sudo make install
    ```

4.  **编译 CT_FGO**：
    ```bash
    cd path/to/CT_FGO
    mkdir build && cd build
    cmake ..
    make -j4
    ```

---

## 4. 配置说明

系统通过 YAML 配置文件驱动（例如 `config/ob_gins_ct.yaml`）。

### 4.1. 全局参数
| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `gnssfile` | String | GNSS 数据文件路径 (格式: Time, Lat, Lon, Alt, ...). |
| `outputpath` | String | 结果保存目录. |
| `windows` | Int | (已弃用/内部使用) 样条窗口数量. |
| `kf_interval_sec` | Double | 样条节点间隔 (默认 0.1s). |
| `num_iterations` | Int | Ceres 求解器的最大迭代次数. |

### 4.2. IMU 配置
每个 IMU 定义为一个独立的段落（例如 `imu_main`, `center_imu1`）。

**通用字段：**
- `type`: `standard` (标准) 或 `wheel` (轮端).
- `file`: IMU 数据文件路径.
- `rate_hz`: 采样率.
- `antlever`: 杆臂值，从车身中心到传感器中心的向量（体系下）.
- `imunoise`: 噪声参数 (加计/陀螺仪白噪声，零偏不稳定性).

**Wheel IMU 特有字段：**
- `wheel_radius`: 初始轮径 (m).
- `speed_weight`: 轮速约束的权重.
- `extrinsic_rotation`: 安装旋转角 (RPY).

---

## 5. 使用方法

### 5.1. 运行优化
编译完成后，可执行文件位于 `build/` 目录下。使用配置文件运行：

```bash
cd build
./ob_gins_ct_optimization ../config/ob_gins_ct.yaml
```

**注意：** 请确保 YAML 配置文件中的所有数据路径都是 **绝对路径**，或者相对于运行目录的正确相对路径。

### 5.2. 可视化与分析
项目提供了一个 Python 脚本用于可视化结果并与真值 (GNSS) 进行对比。

**依赖：**
```bash
pip install numpy matplotlib
```

**运行绘图脚本：**
```bash
python3 plot_ct_results.py --result <output_path>/ct_trajectory.txt --truth <gnss_path>
```
*提示：如果在 YAML 的 `comparison` 部分配置了真值路径，脚本通常会自动找到它。*

---

## 6. 输出文件

所有结果将保存到配置中指定的 `outputpath`。

1.  **`ct_trajectory.txt`**
    - 主要的轨迹输出文件。
    - **格式：** `Time, Lat, Lon, Alt, Vn, Ve, Vd, Roll, Pitch, Yaw, [Ba_x, Ba_y, Ba_z], [Bg_x, Bg_y, Bg_z]`
    - *注：根据导出设置，可能包含零偏状态。*

2.  **`errors_<imu_name>.txt`**
    - 每个 IMU 的估计零偏和标定误差。
    - 用于调试传感器标定情况。

3.  **`comparison_plots/`** (由脚本生成)
    - `.png` 图片，包含轨迹图 (2D)、位置误差 (ENU) 和零偏收敛曲线。

---

## 7. 故障排除 (Troubleshooting)

### 7.1. 常见错误

**错误：`std::bad_alloc` 或 立即发生 Segmentation Fault**
- **原因：** 通常是因为 `FileLoader` 未能打开数据文件，导致代码尝试从无效指针或空缓冲区读取数据。
- **解决：** 仔细检查 `config.yaml` 中的 **所有** 文件路径。确保当前用户有读取权限。
- **解决：** 如果在原生 Linux 环境下运行，确路径不要使用 WSL 格式的 `/mnt/c`，除非你确实挂载了该目录。

**错误：`Sophus::SE3` 相关未定义引用 (undefined references)**
- **原因：** 系统链接到了旧版本或 ROS 自带的 Sophus 库。
- **解决：** 卸载 `ros-*-sophus`，并严格按照 3.2 节的方法源码编译安装 Sophus。

**错误：`Check failed: !IsNaN`**
- **原因：** 优化发散。
- **解决：**
    - 检查 `starttime` 是否在数据覆盖的时间范围内。
    - 如果载体动态很强，尝试减小 `kf_interval_sec`。
    - 检查 IMU 数据单位是否正确（例如：m/s^2 vs g，rad/s vs deg/s）。

### 7.2. 调试技巧
- 启用 `GLOG_v=2` 查看详细的数据加载日志：
  ```bash
  GLOG_v=2 ./ob_gins_ct_optimization ../config/ob_gins_ct.yaml
  ```
