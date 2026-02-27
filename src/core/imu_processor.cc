#include "src/core/imu_processor.h"
#include <glog/logging.h>
#include "src/fileio/imufileloader.h"
#include "src/fileio/filesaver.h"
#include "src/factors/ContinuousInertialFactor.h"
#include "src/factors/WheelNHCFactor.h"
#include "src/factors/WheelSpeedFactor.h"
#include "src/factors/WheelGyroFactor.h"
#include "src/factors/WheelSpinGyroFactor.h" // NEW
#include "src/factors/WheelPhaseFactor.h" // NEW
#include "src/factors/BiasRandomWalkFactor.h"
#include "src/factors/PriorFactors.h"
#include "src/factors/ScalarPriorFactor.h"
// #include "src/factors/WheelAttitudeFactor.h" // Deprecated

namespace ob_gins {

int findControlPointIndex(double t, double t0, double dt, int max_idx) {
    return static_cast<int>(std::floor((t - dt - t0) / dt));
}

// ---------------------- Factory Implementation ----------------------
std::unique_ptr<ImuProcessor> ImuProcessor::Create(const std::string& type) {
    if (type == "standard") {
        return std::make_unique<StandardImuProcessor>();
    } else if (type == "wheel") {
        return std::make_unique<WheelImuProcessor>();
    }
    LOG(ERROR) << "Unknown IMU type: " << type;
    return nullptr;
}
// --------------------------------------------------------------------

void ImuProcessor::SaveErrors(const std::string& output_path, const std::vector<spline::ControlPoint>& control_points, double spline_dt, double t_start_global) {
    if (bg_.empty() || bg_.size() != control_points.size()) return;

    std::string file_name = output_path + "/errors_" + name_ + ".txt";
    // Columns: t, bg(3), ba(3), lever_arm(3), q_body_imu(4), td(1)
    FileSaver saver(file_name, 1 + 3 + 3 + 3 + 4 + 1);

    for (size_t i = 0; i < control_points.size(); ++i) {
        double t = control_points[i].timestamp();
        
        std::vector<double> data;
        data.push_back(t);
        
        data.push_back(bg_[i].x());
        data.push_back(bg_[i].y());
        data.push_back(bg_[i].z());

        data.push_back(ba_[i].x());
        data.push_back(ba_[i].y());
        data.push_back(ba_[i].z());

        data.push_back(l_body_sensor_.x());
        data.push_back(l_body_sensor_.y());
        data.push_back(l_body_sensor_.z());

        data.push_back(q_body_imu_.x());
        data.push_back(q_body_imu_.y());
        data.push_back(q_body_imu_.z());
        data.push_back(q_body_imu_.w());

        data.push_back(td_);

        saver.dump(data);
    }
    saver.close();
    LOG(INFO) << "Saved errors for " << name_ << " to " << file_name;
}

bool ImuProcessor::FetchDataFromBufferInternal(const DataBuffer& buffer, double t_start, double t_end) {
    valid_imu_data_ = buffer.GetImuData(name_, t_start, t_end);
    return !valid_imu_data_.empty();
}

Eigen::Vector3d ImuProcessor::LoadLeverArm(const YAML::Node& config_node, const std::string& key) {
    if (config_node[key]) {
        auto vec = config_node[key].as<std::vector<double>>();
        if (vec.size() == 3) return Eigen::Vector3d(vec[0], vec[1], vec[2]);
    }
    return Eigen::Vector3d::Zero();
}

void ImuProcessor::LoadExtrinsics(const YAML::Node& config_node) {
    if (config_node["extrinsic_rotation"]) {
        auto rpy = config_node["extrinsic_rotation"].as<std::vector<double>>();
        double d2r = M_PI / 180.0;
        Eigen::AngleAxisd roll(rpy[0] * d2r, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch(rpy[1] * d2r, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw(rpy[2] * d2r, Eigen::Vector3d::UnitZ());
        q_body_imu_initial_ = yaw * pitch * roll;
        q_body_imu_ = q_body_imu_initial_;
    }
}

void ImuProcessor::LoadImuNoise(const YAML::Node& config_node) {
    if (config_node["imunoise"]) {
        const auto& n = config_node["imunoise"];
        
        // Load Correlation Times first (needed for Bias RW conversion)
        if (n["accel_corr_time"]) 
            acc_corr_time_ = std::max(1.0, n["accel_corr_time"].as<double>());
        if (n["gyro_corr_time"]) 
            gyr_corr_time_ = std::max(1.0, n["gyro_corr_time"].as<double>());

        double g = 9.80665;
        double d2r = M_PI / 180.0;

        // Accel Noise: ug/sqrt(Hz) -> m/s^2 (discrete sigma)
        // val (ug/rtHz) * 1e-6 * g -> m/s^2/rtHz * sqrt(rate) -> m/s^2
        if (n["accel_noise"]) {
            double val = n["accel_noise"].as<double>();
            acc_noise_ = val * 1.0e-6 * g * std::sqrt(rate_hz_);
        }

        // Gyro Noise: deg/s/sqrt(Hz) -> rad/s (discrete sigma)
        // val (deg/s/rtHz) * d2r -> rad/s/rtHz * sqrt(rate) -> rad/s
        if (n["gyro_noise"]) {
            double val = n["gyro_noise"].as<double>();
            gyr_noise_ = val * d2r * std::sqrt(rate_hz_);
        }

        // Accel Bias Instability: ug -> Driving Noise Density (m/s^3/sqrt(Hz) equivalent)
        // val (ug) * 1e-6 * g -> m/s^2 (sigma_bias)
        // sigma_driving = sigma_bias * sqrt(2/tau)
        if (n["accel_bias_rw"]) {
            double val = n["accel_bias_rw"].as<double>();
            double sigma_b = val * 1.0e-6 * g;
            acc_bias_rw_ = sigma_b * std::sqrt(2.0 / acc_corr_time_);
        }

        // Gyro Bias Instability: deg/h -> Driving Noise Density (rad/s^2/sqrt(Hz) equivalent)
        // val (deg/h) * d2r / 3600 -> rad/s (sigma_bias)
        // sigma_driving = sigma_bias * sqrt(2/tau)
        if (n["gyro_bias_rw"]) {
            double val = n["gyro_bias_rw"].as<double>();
            double sigma_b = val * d2r / 3600.0;
            gyr_bias_rw_ = sigma_b * std::sqrt(2.0 / gyr_corr_time_);
        }
    }
}

bool StandardImuProcessor::LoadConfig(const YAML::Node& config_node, const std::string& imu_name) {
    name_ = imu_name;
    if (!config_node["file"]) return false;
    file_path_ = config_node["file"].as<std::string>();
    columns_ = config_node["columns"].as<int>();
    rate_hz_ = config_node["rate_hz"].as<double>();
    l_body_sensor_ = LoadLeverArm(config_node, "antlever");
    LoadExtrinsics(config_node);
    LoadImuNoise(config_node);
    return true;
}

bool StandardImuProcessor::FetchDataFromBuffer(const DataBuffer& buffer, double t_start, double t_end) {
    return FetchDataFromBufferInternal(buffer, t_start, t_end);
}

void StandardImuProcessor::AddFactors(ceres::Problem& problem, 
                                      std::vector<spline::ControlPoint>& control_points, 
                                      double spline_dt, double t0_spline,
                                      const Eigen::Vector3d& gravity_vec, 
                                      const Eigen::Vector3d& omega_ie_local) {
    if (bg_.empty()) {
        bg_.resize(control_points.size(), Eigen::Vector3d::Zero());
        ba_.resize(control_points.size(), Eigen::Vector3d::Zero());
    }

    problem.AddParameterBlock(q_body_imu_.coeffs().data(), 4);
    problem.SetManifold(q_body_imu_.coeffs().data(), new ceres::EigenQuaternionManifold());
    problem.AddResidualBlock(factors::RotationPriorFactor::Create(q_body_imu_initial_, 0.01), nullptr, q_body_imu_.coeffs().data());

    problem.AddParameterBlock(l_body_sensor_.data(), 3);
    problem.AddResidualBlock(factors::LeverArmPriorFactor::Create(l_body_sensor_, 0.05), nullptr, l_body_sensor_.data());

    problem.AddParameterBlock(&td_, 1);
    problem.AddResidualBlock(factors::ScalarPriorFactor::Create(0.0, 0.01), nullptr, &td_); // Prior: mean 0, std 10ms

    for (size_t i = 0; i < control_points.size(); ++i) {
        problem.AddParameterBlock(bg_[i].data(), 3);
        problem.AddParameterBlock(ba_[i].data(), 3);
    }

    for (const auto& imu : valid_imu_data_) {
        // Find index using time + td (approximate for knot finding)
        double t_sys = imu.time + td_; // Use current estimate or 0
        int k = findControlPointIndex(t_sys, t0_spline, spline_dt, (int)control_points.size());
        if (k < 0 || k + 3 >= (int)control_points.size()) continue;
        double dt = imu.dt;
        if (dt < 1e-6) continue;

        Eigen::Vector3d gyro_meas = imu.dtheta / dt;
        Eigen::Vector3d accel_meas = imu.dvel / dt;

        auto* inertial_factor = factors::ContinuousInertialFactor::Create(
            imu.time, accel_meas, gyro_meas, gravity_vec, omega_ie_local,
            spline_dt, control_points[k].timestamp(), acc_noise_, gyr_noise_
        );
        problem.AddResidualBlock(inertial_factor, new ceres::HuberLoss(1.0), 
            control_points[k].pose_data(), control_points[k+1].pose_data(), 
            control_points[k+2].pose_data(), control_points[k+3].pose_data(),
            bg_[k].data(), bg_[k+1].data(), 
            bg_[k+2].data(), bg_[k+3].data(),
            ba_[k].data(), ba_[k+1].data(), 
            ba_[k+2].data(), ba_[k+3].data(),
            l_body_sensor_.data(),
            &td_
        );
    }
}

void StandardImuProcessor::AddBiasFactors(ceres::Problem& problem, 
                                          std::vector<spline::ControlPoint>& control_points, 
                                          double spline_dt) {
    if (bg_.empty()) return;
    for (size_t i = 0; i < bg_.size() - 1; ++i) {
        problem.AddResidualBlock(factors::BiasRandomWalkFactor::Create(spline_dt, gyr_bias_rw_, gyr_corr_time_),
            nullptr, bg_[i].data(), bg_[i+1].data());
        problem.AddResidualBlock(factors::BiasRandomWalkFactor::Create(spline_dt, acc_bias_rw_, acc_corr_time_),
            nullptr, ba_[i].data(), ba_[i+1].data());
    }
}

bool WheelImuProcessor::LoadConfig(const YAML::Node& config_node, const std::string& imu_name) {
    name_ = imu_name;
    file_path_ = config_node["file"].as<std::string>();
    columns_ = config_node["columns"].as<int>();
    rate_hz_ = config_node["rate_hz"].as<double>();
    l_body_sensor_ = LoadLeverArm(config_node, "antlever");
    
    // Store initial values for Priors
    l_sensor_odopoint_initial_ = LoadLeverArm(config_node, "odolever");
    l_sensor_odopoint_ = l_sensor_odopoint_initial_;
    
    side_ = config_node["side"].as<std::string>();

    LoadExtrinsics(config_node);
    
    // -----------------------------------------------------------------------
    // Automatic Default Extrinsics Setup for Wheel IMUs
    // Target: Align Sensor Y-axis with Gravity (DOWN)
    // 
    // Assumed Body Frame: X-Forward, Y-Left, Z-Up
    // Gravity in Body Frame: [0, 0, -9.8] (Vector points Down)
    // 
    // Wheel Frame Target: 
    //   Y-axis -> Points DOWN (aligned with Gravity)
    //   Z-axis -> Wheel Spin Axis (Outwards)
    //
    // Left Wheel (side="left"):
    //   Spin Axis (Z) points LEFT (same as Body Y)
    //   So Sensor Z aligns with Body Y [0, 1, 0]
    //   Target Sensor Y aligns with Body -Z [0, 0, -1]
    //   Target Sensor X = Y cross Z = (-Z) cross (Y) = [1, 0, 0] (Body X)
    //   R_body_sensor (Left) = [ 1  0  0 ]
    //                          [ 0  0 -1 ]
    //                          [ 0  1  0 ]
    //
    // Right Wheel (side="right"):
    //   Spin Axis (Z) points RIGHT (opposite to Body Y)
    //   So Sensor Z aligns with Body -Y [0, -1, 0]
    //   Target Sensor Y aligns with Body -Z [0, 0, -1] (Gravity)
    //   Target Sensor X = Y cross Z = (-Z) cross (-Y) = [-1, 0, 0] (Body -X? Or X?)
    //   Let's check: (-k) x (-j) = k x j = -i. So X points BACK.
    //   R_body_sensor (Right) = [-1  0  0 ]
    //                           [ 0  0 -1 ]
    //                           [ 0 -1  0 ]
    // -----------------------------------------------------------------------
    
    // Override q_body_imu_initial_ based on side if not manually set (or always override?)
    // User instruction implies setting a "default", so let's set it here.
    // If config has specific "extrinsic_rotation", LoadExtrinsics already set it.
    // But usually config has 0,0,0. We should provide a better default.
    
    // Only override if config is effectively zero (identity)
    if (q_body_imu_initial_.isApprox(Eigen::Quaterniond::Identity(), 1e-3)) {
        Eigen::Matrix3d R_bs;
        if (side_ == "left") {
            // X_s = X_b, Y_s = -Z_b, Z_s = Y_b
            R_bs << 1,  0,  0,
                    0,  0,  1,
                    0, -1,  0;
        } else {
            // Right Side
            // X_s = -X_b, Y_s = -Z_b, Z_s = -Y_b
            R_bs << -1,  0,  0,
                     0,  0, -1,
                     0, -1,  0;
        }
        q_body_imu_initial_ = Eigen::Quaterniond(R_bs);
        q_body_imu_ = q_body_imu_initial_;
        
        LOG(INFO) << "WheelImuProcessor (" << name_ << "): Applied default rotation for side '" << side_ << "'";
    }

    if (config_node["nhc_weight"]) nhc_weight_ = config_node["nhc_weight"].as<double>();
    if (config_node["speed_weight"]) speed_weight_ = config_node["speed_weight"].as<double>();
    if (config_node["attitude_weight_roll"]) att_weight_roll_ = config_node["attitude_weight_roll"].as<double>();
    if (config_node["attitude_weight_yaw"]) att_weight_pitch_ = config_node["attitude_weight_yaw"].as<double>();
    
    if (config_node["wheel_radius"]) wheel_radius_initial_ = config_node["wheel_radius"].as<double>();
    wheel_radius_ = wheel_radius_initial_;
    
    // Load Priors (Std Dev) from config, default to 5mm / 2cm if not set
    if (config_node["priors"]) {
        const auto& p = config_node["priors"];
        if (p["radius_std"]) prior_radius_std_ = p["radius_std"].as<double>();
        if (p["lever_std"]) prior_lever_std_ = p["lever_std"].as<double>();
    }

    LoadImuNoise(config_node);
    return true;
}

bool WheelImuProcessor::FetchDataFromBuffer(const DataBuffer& buffer, double t_start, double t_end) {
    if (!FetchDataFromBufferInternal(buffer, t_start, t_end)) return false;
    for (auto& imu : valid_imu_data_) {
        if (side_ == "right") imu.dtheta.z() *= -1.0;
    }

    integrated_attitudes_.clear();
    Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
    int N = std::min((int)valid_imu_data_.size(), 100);
    for(int i=0; i<N; ++i) acc_mean += valid_imu_data_[i].dvel / valid_imu_data_[i].dt;
    if(N > 0) acc_mean /= N;
    
    mechanization_.Initialize(acc_mean);
    
    for (const auto& imu : valid_imu_data_) {
        mechanization_.Propagate(imu, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        integrated_attitudes_.push_back(mechanization_.GetAttitude());
    }

    // Initialize Phase Control Points using Gyro Z integration (Approx)
    // Theta(t) = Integral(w_z)
    wheel_phases_.clear();
    double curr_theta = 0.0;
    
    // We need phases at spline knots (t0, t0+dt, ...)
    // Resample/Integrate up to those points
    
    // Find start time of spline (usually control_points[0].timestamp())
    // Let's assume control_points are not created yet? No, they are passed in AddFactors.
    // We need to init them before AddFactors.
    // But phases_ size must match control_points.
    // We'll init them in AddFactors if empty? 
    // No, better to do it here if we know the timeline.
    // Actually, AddFactors is where we set up the graph.
    // Let's defer initialization to AddFactors or assume a fixed size based on config?
    // Let's init in AddFactors.

    return true;
}

void WheelImuProcessor::AddFactors(ceres::Problem& problem, 
                                   std::vector<spline::ControlPoint>& control_points, 
                                   double spline_dt, double t0_spline,
                                   const Eigen::Vector3d& gravity_vec, 
                                   const Eigen::Vector3d& omega_ie_local) {
    if (bg_.empty()) {
        bg_.resize(control_points.size(), Eigen::Vector3d::Zero());
        ba_.resize(control_points.size(), Eigen::Vector3d::Zero());
    }
    
    // Initialize Phases
    if (wheel_phases_.empty()) {
        wheel_phases_.resize(control_points.size(), 0.0);
        // Integrate Gyro Z to fill phases
        double theta = 0.0;
        int imu_idx = 0;
        for (size_t i = 0; i < control_points.size(); ++i) {
            double t_target = control_points[i].timestamp();
            while(imu_idx < (int)valid_imu_data_.size() && valid_imu_data_[imu_idx].time < t_target) {
                theta += valid_imu_data_[imu_idx].dtheta.z(); // Increment
                imu_idx++;
            }
            wheel_phases_[i] = theta;
        }

        // ---------------------------------------------------------
        // New: Coarse Phase Alignment (Static Average)
        // Use static data (aligntime) to compute average accel vector.
        // Compute rotation angle to align sensor Y with Gravity (Down).
        // ---------------------------------------------------------
        
        // 1. Calculate Mean Accel during static alignment period
        Eigen::Vector3d acc_sum = Eigen::Vector3d::Zero();
        int acc_count = 0;
        
        // Use t0_spline as start time
        double t_align_end = t0_spline + 3.0; 
        
        for (const auto& imu : valid_imu_data_) {
            if (imu.time > t_align_end) break;
            
            double dt_safe = imu.dt > 1e-6 ? imu.dt : 1.0/rate_hz_;
            Eigen::Vector3d acc_meas = imu.dvel / dt_safe;
            
            // Basic static check
            if (std::abs(acc_meas.norm() - 9.81) < 1.0) {
                acc_sum += acc_meas;
                acc_count++;
            }
        }
        
        double init_offset = 0.0;
        
        if (acc_count > 10) {
            Eigen::Vector3d acc_mean = acc_sum / acc_count;
            // acc_mean measures Reaction Force = -g.
            
            // Need q_wb average
            Eigen::Vector3d g_world_up(0, 0, 9.81);
            
            // Get q_wb at start (approx)
            int k0 = findControlPointIndex(t0_spline + 0.1, t0_spline, spline_dt, (int)control_points.size());
            if (k0 >= 0) {
                Eigen::Quaterniond q_wb_0 = control_points[k0].pose().so3().unit_quaternion();
                
                // g_body_up
                Eigen::Vector3d g_body_up = q_wb_0.inverse() * g_world_up;
                
                // g_hub_up (Zero-Phase Sensor Frame)
                Eigen::Vector3d g_hub_up = q_body_imu_.inverse() * g_body_up;
                
                // We want to find theta such that R_z(theta)^T * g_hub_up has X=0 and Y<0.
                // R_z(theta)^T * [gx, gy, gz]^T = [gx c + gy s, -gx s + gy c, gz]
                
                // 1. Constraint X=0:  gx*cos(th) + gy*sin(th) = 0
                //    => tan(th) = -gx/gy
                //    Two solutions, separated by 180 deg.
                
                // 2. Constraint Y<0: -gx*sin(th) + gy*cos(th) < 0
                
                double gx = g_hub_up.x();
                double gy = g_hub_up.y();
                
                double th1 = std::atan2(-gx, gy);      // Solution 1
                double th2 = std::atan2(gx, -gy);      // Solution 2 (+180)
                
                // Check Y component for th1
                double y1 = -gx * std::sin(th1) + gy * std::cos(th1);
                
                if (y1 < 0) {
                    init_offset = th1;
                } else {
                    init_offset = th2;
                }
                
                LOG(INFO) << "IMU " << name_ << " Static Alignment (3s mean): Computed Offset = " << init_offset * 180.0 / M_PI << " deg";
            }
        } else {
            LOG(WARNING) << "IMU " << name_ << ": Not enough static data for alignment.";
        }

        // Apply offset to all phases
        for (auto& p : wheel_phases_) p += init_offset;
    }

    problem.AddParameterBlock(q_body_imu_.coeffs().data(), 4);
    problem.SetManifold(q_body_imu_.coeffs().data(), new ceres::EigenQuaternionManifold());
    problem.AddParameterBlock(l_body_sensor_.data(), 3);

    // Optimize Wheel Radius (with Prior)
    problem.AddParameterBlock(&wheel_radius_, 1);
    problem.AddResidualBlock(factors::ScalarPriorFactor::Create(wheel_radius_initial_, prior_radius_std_), nullptr, &wheel_radius_);

    // Optimize Wheel Lever Arm (with Prior)
    problem.AddParameterBlock(l_sensor_odopoint_.data(), 3);
    problem.AddResidualBlock(factors::LeverArmPriorFactor::Create(l_sensor_odopoint_initial_, prior_lever_std_), nullptr, l_sensor_odopoint_.data());

    // Optimize Misalignment [kx, ky] (with Prior to keep it small)
    problem.AddParameterBlock(misalignment_xy_.data(), 2);
    // Add weak prior to keep it close to 0 (e.g. sigma=0.05 approx 3 deg coupling)
    problem.AddResidualBlock(factors::Vector2PriorFactor::Create(Eigen::Vector2d::Zero(), 0.05), nullptr, misalignment_xy_.data());

    problem.AddParameterBlock(&td_, 1);
    problem.AddResidualBlock(factors::ScalarPriorFactor::Create(0.0, 0.01), nullptr, &td_); // Prior: mean 0, std 10ms

    problem.AddResidualBlock(factors::RotationPriorFactor::Create(q_body_imu_initial_, 0.01), nullptr, q_body_imu_.coeffs().data());
    problem.AddResidualBlock(factors::LeverArmPriorFactor::Create(l_body_sensor_, 0.05), nullptr, l_body_sensor_.data());

    for (size_t i = 0; i < control_points.size(); ++i) {
        problem.AddParameterBlock(bg_[i].data(), 3);
        problem.AddParameterBlock(ba_[i].data(), 3);
        problem.AddParameterBlock(&wheel_phases_[i], 1); // Phase scalar
    }
    
    // Add Phase Evolution Factors (Constraint between theta_k and theta_k+1)
    // Based on IMU Gyro Z integration
    // This connects the phase states
    
    // Note: We need to aggregate Gyro Z between knots
    int start_imu_idx = 0;
    for (size_t i = 0; i < control_points.size() - 1; ++i) {
        double t_curr = control_points[i].timestamp();
        double t_next = control_points[i+1].timestamp();
        
        // Find IMUs in this interval
        double dtheta_z_sum = 0.0;
        double dt_sum = 0.0;
        
        // Find start
        while(start_imu_idx < (int)valid_imu_data_.size() && valid_imu_data_[start_imu_idx].time <= t_curr) {
            start_imu_idx++;
        }
        
        int curr_idx = start_imu_idx;
        while(curr_idx < (int)valid_imu_data_.size() && valid_imu_data_[curr_idx].time <= t_next) {
            dtheta_z_sum += valid_imu_data_[curr_idx].dtheta.z();
            dt_sum += valid_imu_data_[curr_idx].dt;
            curr_idx++;
        }
        
        if (dt_sum > 1e-6) {
            auto* phase_factor = factors::WheelPhaseFactor::Create(dtheta_z_sum, dt_sum, 1000.0); // High weight for continuity
            problem.AddResidualBlock(phase_factor, nullptr, 
                &wheel_phases_[i], &wheel_phases_[i+1], bg_[i].data()
            );
        }
    }

    for (size_t idx = 0; idx < valid_imu_data_.size(); ++idx) {
        const auto& imu = valid_imu_data_[idx];
        
        double t_sys = imu.time + td_;
        int k = findControlPointIndex(t_sys, t0_spline, spline_dt, (int)control_points.size());
        if (k < 0 || k + 3 >= (int)control_points.size()) continue;

        double dt = imu.dt > 0 ? imu.dt : 1.0/rate_hz_;
        if (dt < 1e-6) continue;

        Eigen::Vector3d gyro_meas = imu.dtheta / dt;
        Eigen::Vector3d accel_meas = imu.dvel / dt;

        // Add Wheel Spin Gyro Factor (Replaces WheelGyroFactor)
        if (att_weight_roll_ > 0 || att_weight_pitch_ > 0) {
            auto* spin_factor = factors::WheelSpinGyroFactor::Create(
                imu.time, spline_dt, control_points[k].timestamp(),
                gyro_meas, att_weight_roll_, att_weight_pitch_
            );
            problem.AddResidualBlock(spin_factor, new ceres::HuberLoss(1.0),
                control_points[k].pose_data(), control_points[k+1].pose_data(), 
                control_points[k+2].pose_data(), control_points[k+3].pose_data(),
                bg_[k].data(), bg_[k+1].data(), 
                bg_[k+2].data(), bg_[k+3].data(),
                q_body_imu_.coeffs().data(),
                &wheel_phases_[k], &wheel_phases_[k+1], &wheel_phases_[k+2], &wheel_phases_[k+3],
                misalignment_xy_.data(),
                &td_
            );
        }

        // Pass l_sensor_odopoint_ as optimization variable (pointer)
        auto* nhc_factor = factors::WheelNHCFactor::Create(
            imu.time, spline_dt, control_points[k].timestamp(), nhc_weight_, l_sensor_odopoint_
        );
        problem.AddResidualBlock(nhc_factor, new ceres::HuberLoss(1.0), 
            control_points[k].pose_data(), control_points[k+1].pose_data(), 
            control_points[k+2].pose_data(), control_points[k+3].pose_data(),
            q_body_imu_.coeffs().data(),
            l_body_sensor_.data(),
            l_sensor_odopoint_.data(),
            &td_
        );


        auto* speed_factor = factors::WheelSpeedFactor::Create(
            imu.time, spline_dt, control_points[k].timestamp(), gyro_meas, speed_weight_, l_sensor_odopoint_
        );
        problem.AddResidualBlock(speed_factor, new ceres::HuberLoss(1.0),
            control_points[k].pose_data(), control_points[k+1].pose_data(), 
            control_points[k+2].pose_data(), control_points[k+3].pose_data(),
            bg_[k].data(), bg_[k+1].data(), 
            bg_[k+2].data(), bg_[k+3].data(),
            q_body_imu_.coeffs().data(),
            l_body_sensor_.data(),
            &wheel_radius_,
            l_sensor_odopoint_.data(),
            &td_
        );
    }
}

void WheelImuProcessor::AddBiasFactors(ceres::Problem& problem, 
                                       std::vector<spline::ControlPoint>& control_points, 
                                       double spline_dt) {
    if (bg_.empty()) return;
    for (size_t i = 0; i < bg_.size() - 1; ++i) {
        problem.AddResidualBlock(factors::BiasRandomWalkFactor::Create(spline_dt, gyr_bias_rw_, gyr_corr_time_),
            nullptr, bg_[i].data(), bg_[i+1].data());
        problem.AddResidualBlock(factors::BiasRandomWalkFactor::Create(spline_dt, acc_bias_rw_, acc_corr_time_),
            nullptr, ba_[i].data(), ba_[i+1].data());
    }
}

void WheelImuProcessor::SaveErrors(const std::string& output_path, const std::vector<spline::ControlPoint>& control_points, double spline_dt, double t_start_global) {
    if (bg_.empty() || bg_.size() != control_points.size()) return;

    std::string file_name = output_path + "/errors_" + name_ + ".txt";
    // Columns: t, bg(3), ba(3), lever_arm(3), q_body_imu(4), l_sensor_odopoint(3), wheel_radius(1), wheel_phase(1), td(1)
    FileSaver saver(file_name, 1 + 3 + 3 + 3 + 4 + 3 + 1 + 1 + 1);
    
    // Also save integrated attitude for debugging
    std::string att_file_name = output_path + "/attitude_" + name_ + ".txt";
    FileSaver att_saver(att_file_name, 1 + 4); // t, q(4)

    for (size_t i = 0; i < control_points.size(); ++i) {
        double t = control_points[i].timestamp();
        
        std::vector<double> data;
        data.push_back(t);
        
        data.push_back(bg_[i].x());
        data.push_back(bg_[i].y());
        data.push_back(bg_[i].z());

        data.push_back(ba_[i].x());
        data.push_back(ba_[i].y());
        data.push_back(ba_[i].z());

        data.push_back(l_body_sensor_.x());
        data.push_back(l_body_sensor_.y());
        data.push_back(l_body_sensor_.z());

        data.push_back(q_body_imu_.x());
        data.push_back(q_body_imu_.y());
        data.push_back(q_body_imu_.z());
        data.push_back(q_body_imu_.w());

        // Append optimized wheel params
        data.push_back(l_sensor_odopoint_.x());
        data.push_back(l_sensor_odopoint_.y());
        data.push_back(l_sensor_odopoint_.z());
        data.push_back(wheel_radius_);

        // Append optimized wheel phase
        if (i < wheel_phases_.size()) {
            data.push_back(wheel_phases_[i]);
        } else {
            data.push_back(0.0);
        }

        data.push_back(td_);

        saver.dump(data);
    }
    
    // Save Integrated Attitudes (sampled at IMU rate, not control point rate)
    for (size_t i = 0; i < valid_imu_data_.size(); ++i) {
        if (i >= integrated_attitudes_.size()) break;
        std::vector<double> att_data;
        att_data.push_back(valid_imu_data_[i].time);
        att_data.push_back(integrated_attitudes_[i].x());
        att_data.push_back(integrated_attitudes_[i].y());
        att_data.push_back(integrated_attitudes_[i].z());
        att_data.push_back(integrated_attitudes_[i].w());
        att_saver.dump(att_data);
    }
    
    saver.close();
    
    // Save Corrected Attitudes (Re-integrated with optimized bias)
    std::string att_corr_file_name = output_path + "/attitude_corrected_" + name_ + ".txt";
    FileSaver att_corr_saver(att_corr_file_name, 1 + 4); // t, q(4)
    
    Eigen::Quaterniond q_corr = Eigen::Quaterniond::Identity();
    // Align with initial mechanization? Or just Identity?
    // Let's use the first mechanization attitude as start to align frame
    if (!integrated_attitudes_.empty()) q_corr = integrated_attitudes_[0];

    for (size_t i = 0; i < valid_imu_data_.size(); ++i) {
        const auto& imu = valid_imu_data_[i];
        double t = imu.time;
        
        // 1. Interpolate Bias
        // Find bg index
        int k = findControlPointIndex(t, t_start_global, spline_dt, (int)control_points.size());
        Eigen::Vector3d bg_val = Eigen::Vector3d::Zero();
        
        if (k >= 0 && k + 1 < (int)bg_.size()) {
            double t_k = control_points[k].timestamp();
            double u = (t - t_k) / spline_dt; // Approximation, spline_dt is knot interval
            // BSpline control points are spaced by spline_dt.
            // t_k is start of interval.
            // Simple linear interp:
            if (u >= 0 && u <= 1.0) {
                bg_val = bg_[k] * (1.0 - u) + bg_[k+1] * u;
            } else {
                bg_val = bg_[k]; // Clamp
            }
        } else if (!bg_.empty()) {
             bg_val = bg_.front(); // Fallback
        }

        // 2. Correct dtheta
        // imu.dtheta is incremental angle. bg_val is rate (rad/s).
        // dt is imu.dt
        Eigen::Vector3d dtheta_corr = imu.dtheta - bg_val * imu.dt;
        
        // 3. Integrate
        // q_{k+1} = q_k * exp(0.5 * dtheta)
        Eigen::Quaterniond dq(1, 0.5 * dtheta_corr.x(), 0.5 * dtheta_corr.y(), 0.5 * dtheta_corr.z());
        dq.normalize();
        q_corr = (q_corr * dq).normalized();
        
        // 4. Save
        std::vector<double> att_data;
        att_data.push_back(t);
        att_data.push_back(q_corr.x());
        att_data.push_back(q_corr.y());
        att_data.push_back(q_corr.z());
        att_data.push_back(q_corr.w());
        att_corr_saver.dump(att_data);
    }
    att_corr_saver.close();

    att_saver.close();
    LOG(INFO) << "Saved errors for " << name_ << " to " << file_name;
    LOG(INFO) << "Saved raw attitudes for " << name_ << " to " << att_file_name;
    LOG(INFO) << "Saved corrected attitudes for " << name_ << " to " << att_corr_file_name;
}

} // namespace ob_gins