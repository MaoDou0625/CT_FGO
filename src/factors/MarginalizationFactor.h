#ifndef OB_GINS_FACTORS_MARGINALIZATION_FACTOR_H
#define OB_GINS_FACTORS_MARGINALIZATION_FACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <memory>

namespace ob_gins {

// Forward declarations
class MarginalizationInfo;

class ResidualBlockInfo {
public:
    ResidualBlockInfo(ceres::CostFunction* cost_function,
                      ceres::LossFunction* loss_function,
                      std::vector<double*> parameter_blocks,
                      std::vector<int> drop_set);

    void Evaluate();

    ceres::CostFunction* cost_function;
    ceres::LossFunction* loss_function;
    std::vector<double*> parameter_blocks;
    std::vector<int> drop_set;

    double** raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;
    
    ~ResidualBlockInfo();
};

class MarginalizationInfo {
public:
    MarginalizationInfo();
    ~MarginalizationInfo();

    void AddResidualBlockInfo(ResidualBlockInfo* residual_block_info);
    void PreMarginalize();
    void Marginalize();
    std::vector<double*> GetParameterBlocks(std::unordered_map<long, double*>& addr_shift);

    std::vector<ResidualBlockInfo*> factors;
    int m, n; // m: size of marginalized variables, n: size of remaining variables
    std::unordered_map<long, int> parameter_block_size; // <memory address, local size>
    std::unordered_map<long, int> parameter_block_global_size; 
    std::unordered_map<long, int> parameter_block_idx;  // <memory address, starting index in H>
    std::vector<double*> keep_block_addr;
    std::vector<int> keep_block_size;
    std::vector<int> keep_block_global_size;
    
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    
    // For evaluating marginalized factor
    std::unordered_map<long, Eigen::VectorXd> keep_block_data; // snapshot of data at marginalization
};

class MarginalizationFactor : public ceres::CostFunction {
public:
    MarginalizationFactor(MarginalizationInfo* marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info_;
};

} // namespace ob_gins

#endif // OB_GINS_FACTORS_MARGINALIZATION_FACTOR_H
