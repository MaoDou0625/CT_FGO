#ifndef OB_GINS_FACTORS_WHEEL_GYRO_FACTOR_H
#define OB_GINS_FACTORS_WHEEL_GYRO_FACTOR_H

#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "src/spline/BSplineEvaluator.h"

namespace ob_gins {
namespace factors {

// Wheel Gyro Factor: Constrains Angular Velocity (Rate), not Integrated Attitude.
// Allows dynamic bias correction.
struct WheelGyroFactor {
    WheelGyroFactor(double t, double dt, double t0, 
                    const Eigen::Vector3d& gyro_meas, 
                    double weight_x, double weight_y) 
        : t_(t), dt_(dt), t0_(t0), gyro_meas_(gyro_meas), 
          w_x_(weight_x), w_y_(weight_y) {}

    template <typename T>
    bool operator()(const T* const p0, const T* const p1, const T* const p2, const T* const p3, 
                    const T* const bg0, const T* const bg1, const T* const bg2, const T* const bg3,
                    const T* const q_body_imu_ptr, // Extrinsics (q_bs)
                    T* residuals) const {
        
        using SE3T = Sophus::SE3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using QuatT = Eigen::Quaternion<T>;
        using ResT = typename spline::BSplineEvaluator::Result<T>;

        Eigen::Map<const SE3T> T0(p0); Eigen::Map<const SE3T> T1(p1);
        Eigen::Map<const SE3T> T2(p2); Eigen::Map<const SE3T> T3(p3);
        
        Eigen::Map<const Vec3T> bg1_vec(bg1);
        Eigen::Map<const Vec3T> bg2_vec(bg2);
        Eigen::Map<const QuatT> q_bs(q_body_imu_ptr); // Body-to-Sensor

        T t_val = T(t_);
        T t_start = T(t0_);
        T u = (t_val - t_start) / T(dt_);

        ResT res = spline::BSplineEvaluator::Evaluate<T>(u, T(dt_), T0, T1, T2, T3);
        
        // 1. Interpolate Bias
        Vec3T bg = bg1_vec * (T(1.0) - u) + bg2_vec * u;

        // 2. Corrected Measurement in Sensor Frame
        Vec3T omega_meas_corr = gyro_meas_.cast<T>() - bg;

        // 3. Predicted Angular Velocity in Body Frame (from B-Spline)
        Vec3T omega_body = res.w_body;

        // 4. Transform Body Rate to Sensor Frame
        // omega_sensor_pred = q_bs^{-1} * omega_body
        // Actually: q_bs rotates FROM Sensor TO Body?
        // Code usually: q_body_imu = q_bs. v_b = q_bs * v_s.
        // So v_s = q_bs^{-1} * v_b.
        Vec3T omega_sensor_pred = q_bs.inverse() * omega_body;

        // 5. Residuals: Constrain Non-Rotation Axes (X and Y)
        // Assumption: Wheel only rotates around Sensor Z axis.
        // So Sensor X and Y rates should match Body rates projected to Sensor X and Y.
        
        residuals[0] = (omega_sensor_pred.x() - omega_meas_corr.x()) * T(w_x_);
        residuals[1] = (omega_sensor_pred.y() - omega_meas_corr.y()) * T(w_y_);
        
        // Z-axis is free (Wheel Spin), no constraint.

        return true;
    }

    static ceres::CostFunction* Create(double t, double dt, double t0, 
                                       const Eigen::Vector3d& gyro_meas, 
                                       double wx, double wy) {
        return new ceres::AutoDiffCostFunction<WheelGyroFactor, 2, 
            7, 7, 7, 7, // Poses
            3, 3, 3, 3, // Biases
            4           // q_body_imu (Extrinsics)
        >(new WheelGyroFactor(t, dt, t0, gyro_meas, wx, wy));
    }

private:
    double t_, dt_, t0_, w_x_, w_y_;
    Eigen::Vector3d gyro_meas_;
};

} // namespace factors
} // namespace ob_gins

#endif // OB_GINS_FACTORS_WHEEL_GYRO_FACTOR_H
