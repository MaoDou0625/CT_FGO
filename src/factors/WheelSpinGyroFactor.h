#ifndef OB_GINS_FACTORS_WHEEL_SPIN_GYRO_FACTOR_H
#define OB_GINS_FACTORS_WHEEL_SPIN_GYRO_FACTOR_H

#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "src/spline/BSplineEvaluator.h"

namespace ob_gins {
namespace factors {

// Advanced Wheel Gyro Factor with Dynamic Derotation.
// Accounts for sensor frame rotation due to wheel spin.
// AND corrects for cross-coupling from Z-spin to X/Y axes (misalignment).
struct WheelSpinGyroFactor {
    WheelSpinGyroFactor(double t, double dt, double t0, 
                        const Eigen::Vector3d& gyro_meas, 
                        double weight_x, double weight_y) 
        : t_(t), dt_(dt), t0_(t0), gyro_meas_(gyro_meas), 
          w_x_(weight_x), w_y_(weight_y) {}

    template <typename T>
    bool operator()(const T* const p0, const T* const p1, const T* const p2, const T* const p3, 
                    const T* const bg0, const T* const bg1, const T* const bg2, const T* const bg3,
                    const T* const q_body_hub_ptr, // Static Extrinsics (Body-to-Hub)
                    const T* const theta0, const T* const theta1, const T* const theta2, const T* const theta3,
                    const T* const misalign_xy, // [k_x, k_y]
                    T* residuals) const {
        
        using SE3T = Sophus::SE3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using QuatT = Eigen::Quaternion<T>;
        using ResT = typename spline::BSplineEvaluator::Result<T>;

        Eigen::Map<const SE3T> T0(p0); Eigen::Map<const SE3T> T1(p1);
        Eigen::Map<const SE3T> T2(p2); Eigen::Map<const SE3T> T3(p3);
        
        Eigen::Map<const Vec3T> bg1_vec(bg1);
        Eigen::Map<const Vec3T> bg2_vec(bg2);
        Eigen::Map<const QuatT> q_bh(q_body_hub_ptr); // Body-to-Hub (Static)

        T t_val = T(t_);
        T t_start = T(t0_) + T(dt_);
        T u = (t_val - t_start) / T(dt_);

        // 1. Evaluate Body State
        ResT res = spline::BSplineEvaluator::Evaluate<T>(u, T(dt_), T0, T1, T2, T3);
        
        // 2. Interpolate Bias
        Vec3T bg = bg1_vec * (T(1.0) - u) + bg2_vec * u;

        // 3. Interpolate Wheel Phase (Theta)
        T u2 = u * u;
        T u3 = u2 * u;
        T b0 = (1.0 - 3.0*u + 3.0*u2 - u3) / 6.0;
        T b1 = (4.0 - 6.0*u2 + 3.0*u3) / 6.0;
        T b2 = (1.0 + 3.0*u + 3.0*u2 - 3.0*u3) / 6.0;
        T b3 = u3 / 6.0;
        
        T theta_wheel = b0 * (*theta0) + b1 * (*theta1) + b2 * (*theta2) + b3 * (*theta3);
        
        // 4. Build Dynamic Rotation R_spin(theta)
        // Wheel rotates around Z axis.
        T half_theta = theta_wheel / T(2.0);
        QuatT q_spin(cos(half_theta), T(0), T(0), sin(half_theta));
        
        // 5. Total Extrinsics: q_bs(t) = q_bh * q_spin
        QuatT q_bs = q_bh * q_spin;

        // 6. Corrected Measurement in Sensor Frame with Misalignment Compensation
        // True_X = Meas_X - k_x * Meas_Z
        // True_Y = Meas_Y - k_y * Meas_Z
        // We use bias-corrected measurements for this calculation
        Vec3T omega_meas_raw_corr = gyro_meas_.cast<T>() - bg;
        T meas_z = omega_meas_raw_corr.z();
        
        T true_meas_x = omega_meas_raw_corr.x() - misalign_xy[0] * meas_z;
        T true_meas_y = omega_meas_raw_corr.y() - misalign_xy[1] * meas_z;

        // 7. Project Body Rate to Sensor Frame
        // omega_sensor_pred = q_bs^{-1} * omega_body
        Vec3T omega_sensor_pred = q_bs.inverse() * res.w_body;

        // 8. Residuals
        // Minimizing difference between Projected Body Rate and True (Corrected) Measured Rate
        residuals[0] = (omega_sensor_pred.x() - true_meas_x) * T(w_x_);
        residuals[1] = (omega_sensor_pred.y() - true_meas_y) * T(w_y_);

        return true;
    }

    static ceres::CostFunction* Create(double t, double dt, double t0, 
                                       const Eigen::Vector3d& gyro_meas, 
                                       double wx, double wy) {
        return new ceres::AutoDiffCostFunction<WheelSpinGyroFactor, 2, 
            7, 7, 7, 7, // Poses
            3, 3, 3, 3, // Biases
            4,          // q_body_hub
            1, 1, 1, 1, // Thetas
            2           // Misalignment [kx, ky]
        >(new WheelSpinGyroFactor(t, dt, t0, gyro_meas, wx, wy));
    }

private:
    double t_, dt_, t0_, w_x_, w_y_;
    Eigen::Vector3d gyro_meas_;
};

} // namespace factors
} // namespace ob_gins

#endif // OB_GINS_FACTORS_WHEEL_SPIN_GYRO_FACTOR_H
