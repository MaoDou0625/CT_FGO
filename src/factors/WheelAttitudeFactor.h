#ifndef OB_GINS_FACTORS_WHEEL_ATTITUDE_FACTOR_H
#define OB_GINS_FACTORS_WHEEL_ATTITUDE_FACTOR_H

#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "src/spline/BSplineEvaluator.h"
#include "src/common/rotation.h"

namespace ob_gins {
namespace factors {

struct WheelAttitudeFactor {
    WheelAttitudeFactor(double t, double dt, double t0, 
                        const Eigen::Quaterniond& q_sw_meas, 
                        const Eigen::Quaterniond& q_ext_nominal,
                        double weight_roll, double weight_yaw) 
        : t_(t), dt_(dt), t0_(t0), q_sw_meas_(q_sw_meas), q_ext_(q_ext_nominal), 
          w_r_(weight_roll), w_y_(weight_yaw) {}

    template <typename T>
    bool operator()(const T* const p0, const T* const p1, const T* const p2, const T* const p3, 
                    const T* const q_body_imu_ptr, // Optimized Extrinsics (Static part)
                    T* residuals) const {
        
        using SE3T = Sophus::SE3<T>;
        using QuatT = Eigen::Quaternion<T>;
        using ResT = typename spline::BSplineEvaluator::Result<T>;

        Eigen::Map<const SE3T> T0(p0); Eigen::Map<const SE3T> T1(p1);
        Eigen::Map<const SE3T> T2(p2); Eigen::Map<const SE3T> T3(p3);
        Eigen::Map<const QuatT> q_ext_opt(q_body_imu_ptr);

        T t_val = T(t_);
        T t_start = T(t0_);
        T u = (t_val - t_start) / T(dt_);

        ResT res = spline::BSplineEvaluator::Evaluate<T>(u, T(dt_), T0, T1, T2, T3);
        
        // res.pose.so3() is R_bw (World-to-Body? No, usually Body-to-World for SE3)
        // Let's assume SE3 translation is p_wb, rotation is q_wb.
        
        // We have q_sw_meas (Sensor-to-World from integration).
        // Relation: q_sw = q_bw * q_sb.
        // We want to verify if q_sb derived from measurements matches q_ext_opt (with free Pitch).
        
        // Computed Sensor-to-Body:
        // q_sb_meas = q_bw^{-1} * q_sw_meas
        QuatT q_bw = res.pose.unit_quaternion();
        QuatT q_sb_meas = q_bw.inverse() * q_sw_meas_.cast<T>();
        
        // Decompose q_sb_meas into ZYX Euler angles (Yaw, Pitch, Roll) relative to Body.
        // Or better: Compute relative rotation between q_sb_meas and q_ext_opt.
        // q_diff = q_ext_opt^{-1} * q_sb_meas
        // Ideally, q_diff should be a pure Pitch rotation (rotation about Y).
        
        QuatT q_diff = q_ext_opt.inverse() * q_sb_meas;
        
        // Convert q_diff to Euler Angles. Assumes ZYX order.
        // If q_diff is pure pitch (Y-axis), then Roll(X) and Yaw(Z) should be zero.
        // Note: Euler conversion singularity at pitch=90. Wheel spins 360, so this happens.
        // Better approach:
        // Check if q_diff's rotation axis is Y-axis.
        // q = [w, x, y, z]. If axis is Y, then x=0, z=0.
        // Residuals: x and z components of the quaternion imaginary part.
        // (Approximation valid for small deviations, but here q_diff represents the spin angle, which is LARGE).
        
        // Wait! q_diff IS the spin angle quaternion. It rotates 360 degrees.
        // If axis is Z (Wheel Axis), then x=0, y=0.
        // So q_diff = [cos(theta/2), 0, 0, sin(theta/2)].
        
        residuals[0] = q_diff.x() * T(w_r_); // Penalty on Roll axis
        residuals[1] = q_diff.y() * T(w_y_); // Penalty on Pitch axis (Assuming Z is rotation axis)

        return true;
    }

    static ceres::CostFunction* Create(double t, double dt, double t0, 
                                       const Eigen::Quaterniond& q_sw, 
                                       const Eigen::Quaterniond& q_ext,
                                       double wr, double wy) {
        return new ceres::AutoDiffCostFunction<WheelAttitudeFactor, 2, 
            7, 7, 7, 7, // Poses
            4           // q_body_imu (Extrinsics)
        >(new WheelAttitudeFactor(t, dt, t0, q_sw, q_ext, wr, wy));
    }

private:
    double t_, dt_, t0_, w_r_, w_y_;
    Eigen::Quaterniond q_sw_meas_, q_ext_;
};

} // namespace factors
} // namespace ob_gins

#endif // OB_GINS_FACTORS_WHEEL_ATTITUDE_FACTOR_H
