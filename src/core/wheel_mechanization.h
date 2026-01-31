#ifndef OB_GINS_CORE_WHEEL_MECHANIZATION_H
#define OB_GINS_CORE_WHEEL_MECHANIZATION_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "src/common/types.h"

namespace ob_gins {

class WheelMechanization {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    WheelMechanization() {
        q_sw_ = Eigen::Quaterniond::Identity(); // Sensor-to-World (Navigation)
        v_w_ = Eigen::Vector3d::Zero();
        p_w_ = Eigen::Vector3d::Zero();
    }

    // Initialize orientation using gravity alignment
    // Assumes stationary or constant velocity, where accel ~ gravity
    void Initialize(const Eigen::Vector3d& acc_mean) {
        Eigen::Vector3d gravity_dir = acc_mean.normalized();
        Eigen::Vector3d up(0, 0, 1); // Navigation frame Up
        // We want R_sw * up = -gravity_dir (accelerometer measures upward force counteracting gravity?)
        // Actually accel measures specific force: f = a - g. Stationary: f = -g.
        // So f_body = R_ws * (0,0,g).
        // Let's assume standard alignment: Z-axis points Up.
        // Accel measures +1g when Z is Up.
        
        Eigen::Quaterniond q_align = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), acc_mean.normalized());
        // This aligns Body Z with Accel vector.
        // If Accel is (0,0,9.8), q is Identity.
        q_sw_ = q_align;
    }

    // Simple Strapdown Integration (0-order hold)
    void Propagate(const IMU& imu, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
        double dt = imu.dt;
        Eigen::Vector3d dtheta = imu.dtheta - bg * dt;
        Eigen::Vector3d dvel = imu.dvel - ba * dt;

        // 1. Update Attitude
        // dq = [1, 0.5 * dtheta]
        Eigen::Quaterniond dq(1, 0.5 * dtheta.x(), 0.5 * dtheta.y(), 0.5 * dtheta.z());
        dq.normalize();
        q_sw_ = (q_sw_ * dq).normalized();

        // 2. Update Velocity (Optional, if we want to use pos/vel)
        // v = v + R * acc_body * dt + g * dt
        // For pure attitude factor, we might skip this to save compute, 
        // but for a full mechanization it's good to have.
    }

    const Eigen::Quaterniond& GetAttitude() const { return q_sw_; }

private:
    Eigen::Quaterniond q_sw_; // Sensor frame to World/Nav frame
    Eigen::Vector3d v_w_;
    Eigen::Vector3d p_w_;
};

} // namespace ob_gins

#endif // OB_GINS_CORE_WHEEL_MECHANIZATION_H
