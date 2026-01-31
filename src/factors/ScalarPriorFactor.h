#ifndef OB_GINS_FACTORS_SCALAR_PRIOR_FACTOR_H
#define OB_GINS_FACTORS_SCALAR_PRIOR_FACTOR_H

#include <ceres/ceres.h>

namespace ob_gins {
namespace factors {

struct ScalarPriorFactor {
    ScalarPriorFactor(double prior_val, double std_dev) 
        : prior_(prior_val), scale_(1.0 / std_dev) {}

    template <typename T>
    bool operator()(const T* const x_ptr, T* residuals) const {
        T x = x_ptr[0];
        residuals[0] = (x - T(prior_)) * T(scale_);
        return true;
    }

    static ceres::CostFunction* Create(double prior_val, double std_dev) {
        return new ceres::AutoDiffCostFunction<ScalarPriorFactor, 1, 1>(
            new ScalarPriorFactor(prior_val, std_dev));
    }

private:
    double prior_;
    double scale_;
};

} // namespace factors
} // namespace ob_gins

#endif // OB_GINS_FACTORS_SCALAR_PRIOR_FACTOR_H
