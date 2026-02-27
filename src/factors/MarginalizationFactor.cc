#include "src/factors/MarginalizationFactor.h"
#include "src/spline/SophusSE3Manifold.h"
#include <glog/logging.h>
#include <iostream>

namespace ob_gins {

ResidualBlockInfo::ResidualBlockInfo(ceres::CostFunction* cost_function,
                                     ceres::LossFunction* loss_function,
                                     std::vector<double*> parameter_blocks,
                                     std::vector<int> drop_set)
    : cost_function(cost_function), loss_function(loss_function),
      parameter_blocks(parameter_blocks), drop_set(drop_set) {
    raw_jacobians = nullptr;
}

ResidualBlockInfo::~ResidualBlockInfo() {
    if (raw_jacobians) {
        delete[] raw_jacobians;
    }
}

void ResidualBlockInfo::Evaluate() {
    int residual_size = cost_function->num_residuals();
    int parameter_block_count = parameter_blocks.size();

    residuals.resize(residual_size);
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    
    if (raw_jacobians) delete[] raw_jacobians;
    raw_jacobians = new double*[parameter_block_count];
    jacobians.resize(parameter_block_count);

    for (int i = 0; i < parameter_block_count; i++) {
        jacobians[i].resize(residual_size, block_sizes[i]);
        jacobians[i].setZero();
        raw_jacobians[i] = jacobians[i].data();
    }

    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    if (loss_function) {
        double residual_scaling_, alpha_sq_norm_;
        double sq_norm, rho[3];
        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        double sqrt_rho1_ = sqrt(rho[1]);
        if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        } else {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }
        for (int i = 0; i < parameter_block_count; i++) {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }
        residuals *= residual_scaling_;
    }

    ob_gins::spline::SophusSE3Manifold se3_manifold;

    for (int i = 0; i < parameter_block_count; i++) {
        if (block_sizes[i] == 7) {
            int local_size = 6;
            int global_size = 7;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_jacobian(residual_size, local_size);
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> plus_jacobian(global_size, local_size);
            se3_manifold.PlusJacobian(parameter_blocks[i], plus_jacobian.data());
            local_jacobian = jacobians[i] * plus_jacobian;
            jacobians[i] = local_jacobian;
        }
    }
}

MarginalizationInfo::MarginalizationInfo() {
    m = 0;
    n = 0;
}

MarginalizationInfo::~MarginalizationInfo() {
    for (auto it : factors) {
        delete it;
    }
}

void MarginalizationInfo::AddResidualBlockInfo(ResidualBlockInfo* residual_block_info) {
    factors.push_back(residual_block_info);

    std::vector<double*>& parameter_blocks = residual_block_info->parameter_blocks;
    std::vector<int> block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for (size_t i = 0; i < parameter_blocks.size(); i++) {
        long addr = reinterpret_cast<long>(parameter_blocks[i]);
        if (parameter_block_size.find(addr) == parameter_block_size.end()) {
            parameter_block_global_size[addr] = block_sizes[i];
            if (block_sizes[i] == 7) {
                parameter_block_size[addr] = 6;
            } else {
                parameter_block_size[addr] = block_sizes[i];
            }
        }
    }
}

void MarginalizationInfo::PreMarginalize() {
    for (auto it : factors) {
        it->Evaluate();
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = it->cost_function->parameter_block_sizes()[i];
            if (keep_block_data.find(addr) == keep_block_data.end()) {
                Eigen::VectorXd data(size);
                for (int j = 0; j < size; j++) {
                    data(j) = it->parameter_blocks[i][j];
                }
                keep_block_data[addr] = data;
            }
        }
    }
}

void MarginalizationInfo::Marginalize() {
    int pos = 0;
    for (auto& it : parameter_block_size) {
        it.second = parameter_block_size[it.first];
    }

    std::vector<long> drop_addrs;
    std::vector<long> keep_addrs;
    
    // figure out which are dropped and which are kept
    for (auto it : factors) {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            bool is_drop = false;
            for (int j = 0; j < static_cast<int>(it->drop_set.size()); j++) {
                if (it->parameter_blocks[i] == it->parameter_blocks[it->drop_set[j]]) {
                    is_drop = true;
                    break;
                }
            }
            if (is_drop) {
                if (std::find(drop_addrs.begin(), drop_addrs.end(), addr) == drop_addrs.end())
                    drop_addrs.push_back(addr);
            } else {
                if (std::find(keep_addrs.begin(), keep_addrs.end(), addr) == keep_addrs.end())
                    keep_addrs.push_back(addr);
            }
        }
    }

    m = 0;
    for (long addr : drop_addrs) {
        parameter_block_idx[addr] = m;
        m += parameter_block_size[addr];
    }

    n = 0;
    for (long addr : keep_addrs) {
        parameter_block_idx[addr] = m + n;
        n += parameter_block_size[addr];
        keep_block_addr.push_back(reinterpret_cast<double*>(addr));
        keep_block_size.push_back(parameter_block_size[addr]);
        keep_block_global_size.push_back(parameter_block_global_size[addr]);
    }

    Eigen::MatrixXd H(m + n, m + n);
    Eigen::VectorXd b(m + n);
    H.setZero();
    b.setZero();

    for (auto it : factors) {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
            long addr_i = reinterpret_cast<long>(it->parameter_blocks[i]);
            int idx_i = parameter_block_idx[addr_i];
            int size_i = parameter_block_size[addr_i];
            
            Eigen::MatrixXd jacobian_i = it->jacobians[i];
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
            
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++) {
                long addr_j = reinterpret_cast<long>(it->parameter_blocks[j]);
                int idx_j = parameter_block_idx[addr_j];
                int size_j = parameter_block_size[addr_j];
                
                Eigen::MatrixXd jacobian_j = it->jacobians[j];
                Eigen::MatrixXd hessian = jacobian_i.transpose() * jacobian_j;
                H.block(idx_i, idx_j, size_i, size_j) += hessian;
                if (i != j) {
                    H.block(idx_j, idx_i, size_j, size_i) += hessian.transpose();
                }
            }
        }
    }

    // Schur complement
    if (m > 0) {
        Eigen::MatrixXd Hmm = 0.5 * (H.block(0, 0, m, m) + H.block(0, 0, m, m).transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Hmm);
        Eigen::MatrixXd Hmm_inv = saes.eigenvectors() * 
            Eigen::VectorXd((saes.eigenvalues().array() > 1e-6).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * 
            saes.eigenvectors().transpose();
        
        Eigen::VectorXd bmm = b.segment(0, m);
        Eigen::MatrixXd Hmr = H.block(0, m, m, n);
        Eigen::MatrixXd Hrm = H.block(m, 0, n, m);
        Eigen::MatrixXd Hrr = H.block(m, m, n, n);
        Eigen::VectorXd brr = b.segment(m, n);
        
        Eigen::MatrixXd H_marg = Hrr - Hrm * Hmm_inv * Hmr;
        Eigen::VectorXd b_marg = brr - Hrm * Hmm_inv * bmm;
        
        H_marg = 0.5 * (H_marg + H_marg.transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_marg);
        Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > 1e-6).select(saes2.eigenvalues(), 0));
        Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > 1e-6).select(saes2.eigenvalues().array().inverse(), 0));
        
        Eigen::VectorXd S_sqrt = S.cwiseSqrt();
        Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
        
        linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
        linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b_marg;
    } else {
        // No variables marginalized
        linearized_jacobians = Eigen::MatrixXd::Zero(n, n);
        linearized_residuals = Eigen::VectorXd::Zero(n);
    }

    for (auto it : factors) {
        delete it;
    }
    factors.clear();
}

std::vector<double*> MarginalizationInfo::GetParameterBlocks(std::unordered_map<long, double*>& addr_shift) {
    std::vector<double*> keep_block_addr_shifted;
    for (int i = 0; i < static_cast<int>(keep_block_addr.size()); i++) {
        long addr = reinterpret_cast<long>(keep_block_addr[i]);
        if (addr_shift.find(addr) != addr_shift.end()) {
            keep_block_addr_shifted.push_back(addr_shift[addr]);
        } else {
            keep_block_addr_shifted.push_back(keep_block_addr[i]);
        }
    }
    return keep_block_addr_shifted;
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* marginalization_info)
    : marginalization_info_(marginalization_info) {
    for (int size : marginalization_info_->keep_block_global_size) {
        mutable_parameter_block_sizes()->push_back(size);
    }
    set_num_residuals(marginalization_info_->n);
}

bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    int n = marginalization_info_->n;
    int m = marginalization_info_->m;
    Eigen::VectorXd dx(n);
    ob_gins::spline::SophusSE3Manifold se3_manifold;

    for (int i = 0; i < static_cast<int>(marginalization_info_->keep_block_size.size()); i++) {
        int local_size = marginalization_info_->keep_block_size[i];
        int global_size = marginalization_info_->keep_block_global_size[i];
        long addr = reinterpret_cast<long>(marginalization_info_->keep_block_addr[i]);
        int idx = marginalization_info_->parameter_block_idx[addr] - m;
        
        Eigen::Map<const Eigen::VectorXd> x(parameters[i], global_size);
        Eigen::VectorXd x0 = marginalization_info_->keep_block_data[addr];

        if (global_size == 7) {
            Eigen::VectorXd delta(local_size);
            se3_manifold.Minus(x.data(), x0.data(), delta.data());
            dx.segment(idx, local_size) = delta;
        } else {
            dx.segment(idx, local_size) = x - x0;
        }
    }

    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info_->linearized_residuals + marginalization_info_->linearized_jacobians * dx;

    if (jacobians) {
        for (int i = 0; i < static_cast<int>(marginalization_info_->keep_block_size.size()); i++) {
            if (jacobians[i]) {
                int local_size = marginalization_info_->keep_block_size[i];
                int global_size = marginalization_info_->keep_block_global_size[i];
                long addr = reinterpret_cast<long>(marginalization_info_->keep_block_addr[i]);
                int idx = marginalization_info_->parameter_block_idx[addr] - m;

                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian_i(jacobians[i], n, global_size);
                Eigen::MatrixXd J_local = marginalization_info_->linearized_jacobians.block(0, idx, n, local_size);

                if (global_size == 7) {
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> minus_jacobian(local_size, global_size);
                    se3_manifold.MinusJacobian(parameters[i], minus_jacobian.data());
                    jacobian_i = J_local * minus_jacobian;
                } else {
                    jacobian_i = J_local;
                }
            }
        }
    }
    return true;
}

} // namespace ob_gins
