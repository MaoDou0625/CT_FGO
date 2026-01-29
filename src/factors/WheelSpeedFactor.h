#ifndef OB_GINS_FACTORS_WHEEL_SPEED_FACTOR_H
#define OB_GINS_FACTORS_WHEEL_SPEED_FACTOR_H

#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "src/spline/BSplineEvaluator.h"

namespace ob_gins {
namespace factors {

struct WheelSpeedFactor {
    // Note: l_sensor_odopoint is here only for constructor compatibility, but we optimize it now.
    // Wait, in this simplified version, we only optimize radius, NOT lever arm?
    // User requested: Optimize Radius & Lever Arm WITHOUT Install Error.
    // So we need to accept l_sensor_odopoint_ptr.
    
    WheelSpeedFactor(double t, double dt, double t0, const Eigen::Vector3d& gyro_meas, double weight) 
        : t_(t), dt_(dt), t0_(t0), gyro_meas_(gyro_meas), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p0, const T* const p1, const T* const p2, const T* const p3, 
                    const T* const bg0, const T* const bg1, const T* const bg2, const T* const bg3,
                    const T* const q_body_imu_ptr,
                    const T* const l_body_sensor_ptr,
                    const T* const radius_ptr,
                    const T* const l_sensor_odopoint_ptr,
                    T* residuals) const {
        
        using SE3T = Sophus::SE3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using QuatT = Eigen::Quaternion<T>;
        using ResT = typename spline::BSplineEvaluator::Result<T>;

        Eigen::Map<const SE3T> T0(p0); Eigen::Map<const SE3T> T1(p1);
        Eigen::Map<const SE3T> T2(p2); Eigen::Map<const SE3T> T3(p3);
        
        Eigen::Map<const Vec3T> bg1_vec(bg1);
        Eigen::Map<const Vec3T> bg2_vec(bg2);

        Eigen::Map<const QuatT> q_body_imu(q_body_imu_ptr);
        Eigen::Map<const Vec3T> l_body_sensor(l_body_sensor_ptr);
        T radius = *radius_ptr;
        Eigen::Map<const Vec3T> l_sensor_odopoint(l_sensor_odopoint_ptr);

        (void)bg0; (void)bg3;

        T t_val = T(t_);
        T t_start = T(t0_) + T(dt_);
        T u = (t_val - t_start) / T(dt_);

        ResT res = spline::BSplineEvaluator::Evaluate<T>(u, T(dt_), T0, T1, T2, T3);

        // 1. Interpolate Bias
        Vec3T bg = bg1_vec * (T(1.0) - u) + bg2_vec * u;

        // 2. NHC Velocity: v_nhc = v_body + w_body x l_total
        Vec3T l_nhc = l_body_sensor + q_body_imu * l_sensor_odopoint;
        Vec3T v_nhc_b = res.v_body + res.w_body.cross(l_nhc);

        // 3. Wheel Speed Measurement (Simplified Physics for Rotating IMU)
        // Since IMU rotates with wheel, Z-axis is the rotation axis.
        // We assume Z-axis is roughly aligned with axle (no install error correction).
        // So we just take w_z directly.
        // w_sensor = w_meas - bias
        Vec3T w_sensor = gyro_meas_.cast<T>() - bg;
        
        // 4. Compute Linear Velocity
        // v_meas = w_sensor.z * radius
        T v_meas = w_sensor.z() * radius;

        // 5. Residual: v_pred - v_meas
        residuals[0] = (v_nhc_b.x() - v_meas) * T(weight_);

        return true;
    }

    static ceres::CostFunction* Create(double t, double dt, double t0, const Eigen::Vector3d& gyro_meas, double weight) {
        return new ceres::AutoDiffCostFunction<WheelSpeedFactor, 1, 
            7, 7, 7, 7, // Poses
            3, 3, 3, 3, // Biases
            4,          // q_body_imu
            3,          // l_body_sensor
            1,          // radius
            3           // l_sensor_odopoint (Optimized)
        >(new WheelSpeedFactor(t, dt, t0, gyro_meas, weight));
    }

private:
    double t_, dt_, t0_, weight_;
    Eigen::Vector3d gyro_meas_;
};

} // namespace factors
} // namespace ob_gins

#endif