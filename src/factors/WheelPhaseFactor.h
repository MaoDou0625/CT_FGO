#ifndef OB_GINS_FACTORS_WHEEL_PHASE_FACTOR_H
#define OB_GINS_FACTORS_WHEEL_PHASE_FACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace ob_gins {
namespace factors {

// Constraints the evolution of wheel phase angle between two control points.
// theta_{k+1} = theta_k + integral(omega_z - bg_z)
// Or simplified: theta_{k+1} - theta_k = dtheta_meas_z - bg_z * dt
struct WheelPhaseFactor {
    WheelPhaseFactor(double dtheta_meas_z, double dt, double weight) 
        : dtheta_meas_z_(dtheta_meas_z), dt_(dt), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const theta_k, const T* const theta_k1, 
                    const T* const bg_k, // Bias at k (approximation for interval)
                    T* residuals) const {
        
        T d_theta_pred = *theta_k1 - *theta_k;
        
        // Corrected measurement: dtheta_corr = dtheta_meas - bg_z * dt
        // bg_k is vector3
        T bg_z = bg_k[2]; 
        T d_theta_meas = T(dtheta_meas_z_) - bg_z * T(dt_);
        
        residuals[0] = (d_theta_pred - d_theta_meas) * T(weight_);
        
        return true;
    }

    static ceres::CostFunction* Create(double dtheta_meas_z, double dt, double weight) {
        return new ceres::AutoDiffCostFunction<WheelPhaseFactor, 1, 1, 1, 3>(
            new WheelPhaseFactor(dtheta_meas_z, dt, weight));
    }

private:
    double dtheta_meas_z_;
    double dt_;
    double weight_;
};

} // namespace factors
} // namespace ob_gins

#endif // OB_GINS_FACTORS_WHEEL_PHASE_FACTOR_H
