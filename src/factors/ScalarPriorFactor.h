#ifndef OB_GINS_FACTORS_SCALAR_PRIOR_FACTOR_H
#define OB_GINS_FACTORS_SCALAR_PRIOR_FACTOR_H

#include <ceres/ceres.h>

namespace ob_gins {
namespace factors {

struct ScalarPriorFactor {
    ScalarPriorFactor(double meas, double std_dev) 
        : meas_(meas), scale_(1.0 / std_dev) {}

    template <typename T>
    bool operator()(const T* const param, T* residuals) const {
        residuals[0] = (param[0] - T(meas_)) * T(scale_);
        return true;
    }

    static ceres::CostFunction* Create(double meas, double std_dev) {
        return new ceres::AutoDiffCostFunction<ScalarPriorFactor, 1, 1>(
            new ScalarPriorFactor(meas, std_dev));
    }

    double meas_;
    double scale_;
};

} // namespace factors
} // namespace ob_gins

#endif // OB_GINS_FACTORS_SCALAR_PRIOR_FACTOR_H