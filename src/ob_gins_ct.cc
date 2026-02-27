#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ceres/ceres.h>

DECLARE_bool(logtostderr);
DECLARE_int32(v);

#include <yaml-cpp/yaml.h>
#include <sophus/se3.hpp>

#include "src/common/types.h"
#include "src/common/earth.h"
#include "src/common/angle.h"
#include "src/common/rotation.h"
#include "src/fileio/imufileloader.h"
#include "src/fileio/gnssfileloader.h"
#include "src/fileio/filesaver.h"
#include "src/spline/SplineInitializer.h"
#include "src/spline/BSplineEvaluator.h"
#include "src/spline/SophusSE3Manifold.h"
#include "src/factors/ContinuousInertialFactor.h"
#include "src/factors/ContinuousGnssFactor.h"
#include "src/factors/BiasRandomWalkFactor.h"
#include "src/factors/PriorFactors.h"
#include "src/factors/MarginalizationFactor.h"
#include "src/core/imu_processor.h"
#include "src/core/data_buffer.h"
#include "src/core/window_manager.h"

using namespace ob_gins;
using namespace ob_gins::spline;
using namespace ob_gins::factors;

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    if (argc != 2) {
        LOG(ERROR) << "Usage: ./ob_gins_ct <config_file_path>";
        return -1;
    }

    LOG(INFO) << "OB_GINS Continuous Time Optimization";
    
    // 1. Load Config
    YAML::Node config;
    try {
        config = YAML::LoadFile(argv[1]);
    } catch (YAML::Exception& e) {
        LOG(ERROR) << "Failed to load config: " << e.what();
        return -1;
    }

    // 设置调试级别
    if (config["debug"] && config["debug"]["level"]) {
        FLAGS_v = config["debug"]["level"].as<int>();
    }

    std::string output_path = config["outputpath"].as<std::string>();
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directories(output_path);
    }

    // 2. Load Data and Process Imus
    LOG(INFO) << "Creating IMU Processors...";
    std::vector<std::unique_ptr<ImuProcessor>> imu_processors;
    
    // Iterate through config to find all IMU entries
    for (YAML::const_iterator it = config.begin(); it != config.end(); ++it) {
        std::string key = it->first.as<std::string>();
        if (it->second.IsMap() && it->second["type"]) { 
            std::string type = it->second["type"].as<std::string>();
            
            // Factory Pattern: Create specific processor based on type string
            auto processor = ImuProcessor::Create(type);
            
            if (processor) {
                if (processor->LoadConfig(it->second, key)) {
                    LOG(INFO) << "Loaded IMU: " << key << " (Type: " << type << ")";
                    imu_processors.push_back(std::move(processor));
                } else {
                    LOG(ERROR) << "Failed to load config for IMU: " << key;
                }
            }
        }
    }

    if (imu_processors.empty()) {
        LOG(ERROR) << "No IMU configurations found or loaded.";
        return -1;
    }

    // Setup DataBuffer and Loaders
    DataBuffer data_buffer;
    std::string gnss_path = config["gnssfile"].as<std::string>();
    GnssFileLoader gnss_loader(gnss_path);
    
    struct ImuLoaderContext {
        std::string name;
        std::unique_ptr<ImuFileLoader> loader;
    };
    std::vector<ImuLoaderContext> imu_loaders;
    for (const auto& processor : imu_processors) {
        imu_loaders.push_back({
            processor->GetName(),
            std::make_unique<ImuFileLoader>(processor->GetFilePath(), processor->GetColumns(), processor->GetRateHz())
        });
    }

    LOG(INFO) << "Streaming data into buffer...";
    bool all_eof = false;
    while (!all_eof) {
        all_eof = true;
        
        if (!gnss_loader.isEof()) {
            data_buffer.AddGnssData(gnss_loader.next());
            all_eof = false;
        }

        for (auto& ctx : imu_loaders) {
            if (!ctx.loader->isEof()) {
                data_buffer.AddImuData(ctx.name, ctx.loader->next());
                all_eof = false;
            }
        }
    }

    double t_start_global = config["starttime"] ? config["starttime"].as<double>() : data_buffer.GetEarliestTime();
    double t_end_global = config["endtime"] ? config["endtime"].as<double>() : data_buffer.GetLatestTime();

    // Include GNSS in the intersection
    auto gnss_data_all = data_buffer.GetGnssData(0.0, 1e15);
    if (!gnss_data_all.empty()) {
        t_start_global = std::max(t_start_global, gnss_data_all.front().time);
        t_end_global = std::min(t_end_global, gnss_data_all.back().time);
    }

    LOG(INFO) << "Initial Time window from Buffer: " << std::fixed << t_start_global << " to " << t_end_global 
              << " (Duration: " << (t_end_global - t_start_global) << "s)";

    // Fetch data for all IMUs from buffer and adjust global time window
    for (const auto& processor : imu_processors) {
        if (!processor->FetchDataFromBuffer(data_buffer, t_start_global, t_end_global)) {
            LOG(ERROR) << "Failed to fetch data for IMU: " << processor->GetName();
            return -1;
        }
        // Adjust global time window based on actual fetched IMU data
        if (!processor->GetImuData().empty()) {
            t_start_global = std::max(t_start_global, processor->GetImuData().front().time);
            t_end_global = std::min(t_end_global, processor->GetImuData().back().time);
        }
    }
    
    // Filter GNSS data based on final global time window
    GNSS origin_gnss;
    bool origin_set = false;
    std::vector<GNSS> valid_gnss = data_buffer.GetGnssData(t_start_global, t_end_global);
    
    if (!valid_gnss.empty()) {
        origin_gnss = valid_gnss.front();
        origin_set = true;
    }
    
    if (!origin_set || valid_gnss.empty()) {
        LOG(ERROR) << "No valid GNSS data in final time window [" << t_start_global << ", " << t_end_global << "].";
        return -1;
    }

    LOG(INFO) << "Final Time window: " << std::fixed << t_start_global << " to " << t_end_global 
              << " (Duration: " << (t_end_global - t_start_global) << "s)";
    
    // 3. Initialize Spline and Earth Model
    double spline_dt = 1.0; 
    if (config["kf_interval_sec"]) spline_dt = config["kf_interval_sec"].as<double>();
    
    Earth earth;
    Vector3d gravity_l, omega_ie_l;
    // 以第一点 GNSS 作为局部坐标系原点
    Vector3d origin_ecef = earth.blh2ecef(valid_gnss.front().blh);

    if (config["isearth"] && config["isearth"].as<bool>()) {
        double g = earth.gravity(valid_gnss.front().blh);
        gravity_l << 0, 0, -g; // 导航系(ENU)下的重力
        omega_ie_l = earth.iewn(valid_gnss.front().blh(0)); // 导航系下的地球自转
    } else {
        gravity_l << 0, 0, -9.80665;
        omega_ie_l.setZero();
    }

    // 4. Build Optimization Problem (Sliding Window MVP)
    double window_size = config["window_size"] ? config["window_size"].as<double>() : 10.0;
    double step_size = config["step_size"] ? config["step_size"].as<double>() : 5.0;

    // Initialize all control points upfront for the offline MVP
    std::vector<GNSS> gnss_enu = valid_gnss;
    std::vector<std::pair<double, Sophus::SE3d>> path_for_init;

    for (auto& g : gnss_enu) {
        g.blh = earth.global2local(valid_gnss.front().blh, g.blh); 
        path_for_init.emplace_back(g.time, Sophus::SE3d(Eigen::Quaterniond::Identity(), g.blh));
    }

    std::vector<ControlPoint> control_points = SplineInitializer::InitializeFromPath(path_for_init, spline_dt);

    double current_window_start = t_start_global;
    MarginalizationInfo* last_marg_info = nullptr;
    Eigen::Vector3d gnss_lever_arm = Eigen::Vector3d::Zero();
    
    while (current_window_start < t_end_global) {
        double current_window_end = std::min(current_window_start + window_size, t_end_global);
        
        LOG(INFO) << "Optimizing Window: [" << std::fixed << current_window_start << ", " << current_window_end << "]";
        
        ceres::Problem problem;
        
        // Setup Manifolds and freeze historical states
        for (auto& cp : control_points) {
            problem.AddParameterBlock(cp.pose_data(), 7);
            problem.SetManifold(cp.pose_data(), new SophusSE3Manifold());
            
            // If we have last_marg_info, we don't freeze variables just because they are old,
            // UNLESS they are truly out of the sliding window and already marginalized.
            // Wait, variables that are marginalized are removed from optimization. 
            // We can still freeze them so Ceres doesn't change them, but they might be in keep_block_addr!
            // Wait: If a variable is in keep_block_addr, it MUST NOT be constant, otherwise its Jacobian is 0 and it won't affect the prior!
            // Actually, variables that were completely dropped are just left alone (not added to problem or frozen).
            // But here we add ALL control points to the problem.
            if (cp.timestamp() < current_window_start - 3.0 * spline_dt) {
                problem.SetParameterBlockConstant(cp.pose_data());
                cp.set_state(ControlPoint::State::MARGINALIZED);
            } else if (cp.timestamp() <= current_window_end + 3.0 * spline_dt) {
                cp.set_state(ControlPoint::State::ACTIVE);
            }
        }

        if (last_marg_info && last_marg_info->keep_block_size.size() > 0) {
            auto* factor = new MarginalizationFactor(last_marg_info);
            problem.AddResidualBlock(factor, nullptr, last_marg_info->keep_block_addr);
            
            // Ensure variables in the prior are not constant (unfreeze them if they were frozen)
            for (auto* addr : last_marg_info->keep_block_addr) {
                if (problem.HasParameterBlock(addr) && problem.IsParameterBlockConstant(addr)) {
                    problem.SetParameterBlockVariable(addr);
                }
            }
        }

        problem.AddParameterBlock(gnss_lever_arm.data(), 3);
        problem.SetParameterBlockConstant(gnss_lever_arm.data());

        Eigen::Vector3d gnss_std(1.0/0.1, 1.0/0.1, 1.0/0.2);
        Matrix3d gnss_sqrt_info = gnss_std.asDiagonal(); 
        
        for (const auto& gnss : gnss_enu) {
            if (gnss.time < current_window_start || gnss.time >= current_window_end) continue;
            
            int k = findControlPointIndex(gnss.time, t_start_global, spline_dt, (int)control_points.size());
            if (k < 0 || k + 3 >= (int)control_points.size()) continue;

            double max_wheel_speed = 0.0;
            int wheel_imu_count = 0;

            for (const auto& processor : imu_processors) {
                if (processor->GetName().find("imu_main") == std::string::npos) {
                    const auto& imu_data = processor->GetImuData();
                    auto it = std::lower_bound(imu_data.begin(), imu_data.end(), gnss.time, 
                        [](const IMU& a, double t) { return a.time < t; });
                    
                    if (it != imu_data.end() && it != imu_data.begin()) {
                        double speed_sum = 0;
                        int count = 0;
                        auto start_it = (it - imu_data.begin() >= 10) ? (it - 10) : imu_data.begin();
                        auto end_it = (imu_data.end() - it >= 10) ? (it + 10) : imu_data.end();
                        
                        for(auto it2 = start_it; it2 != end_it; ++it2) {
                            speed_sum += std::abs(it2->odovel / it2->dt);
                            count++;
                        }
                        if (count > 0) {
                            max_wheel_speed = std::max(max_wheel_speed, speed_sum / count);
                            wheel_imu_count++;
                        }
                    }
                }
            }

            Matrix3d current_gnss_sqrt_info = gnss_sqrt_info;
            if (wheel_imu_count > 0 && max_wheel_speed < 0.05) {
                current_gnss_sqrt_info = gnss_sqrt_info * 0.001; 
            }

            auto* factor = ContinuousGnssFactor::Create(gnss.time, spline_dt, t_start_global, gnss.blh, current_gnss_sqrt_info);
            problem.AddResidualBlock(factor, nullptr, 
                control_points[k].pose_data(), control_points[k+1].pose_data(), 
                control_points[k+2].pose_data(), control_points[k+3].pose_data(),
                gnss_lever_arm.data()
            );
        }

        for (auto& processor : imu_processors) {
            processor->AddFactors(problem, control_points, spline_dt, t_start_global, gravity_l, omega_ie_l, current_window_start, current_window_end);
            processor->AddBiasFactors(problem, control_points, spline_dt, current_window_start, current_window_end);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = config["num_iterations"] ? config["num_iterations"].as<int>() : 10;
        options.minimizer_progress_to_stdout = false; // Mute for multi-window
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        LOG(INFO) << "Window Solved. Cost: " << summary.final_cost << " / Iterations: " << summary.iterations.size();
        
        // 2. Marginalization
        double next_window_start = current_window_start + step_size;
        std::vector<double*> drop_set_addrs;
        
        for (auto& cp : control_points) {
            if (cp.timestamp() >= current_window_start - 3.0 * spline_dt && 
                cp.timestamp() < next_window_start - 3.0 * spline_dt) {
                drop_set_addrs.push_back(cp.pose_data());
            }
        }
        
        if (!drop_set_addrs.empty() && next_window_start < t_end_global) {
            MarginalizationInfo* marg_info = new MarginalizationInfo();
            
            std::vector<ceres::ResidualBlockId> residual_blocks;
            problem.GetResidualBlocks(&residual_blocks);
            
            for (auto& rb : residual_blocks) {
                std::vector<double*> param_blocks;
                problem.GetParameterBlocksForResidualBlock(rb, &param_blocks);
                
                bool involves_drop = false;
                std::vector<int> drop_set;
                for (size_t i = 0; i < param_blocks.size(); i++) {
                    if (std::find(drop_set_addrs.begin(), drop_set_addrs.end(), param_blocks[i]) != drop_set_addrs.end()) {
                        involves_drop = true;
                        drop_set.push_back(i);
                    }
                }
                
                if (involves_drop) {
                    auto* cost_func = const_cast<ceres::CostFunction*>(problem.GetCostFunctionForResidualBlock(rb));
                    auto* loss_func = const_cast<ceres::LossFunction*>(problem.GetLossFunctionForResidualBlock(rb));
                    
                    ResidualBlockInfo* rb_info = new ResidualBlockInfo(cost_func, loss_func, param_blocks, drop_set);
                    marg_info->AddResidualBlockInfo(rb_info);
                }
            }
            
            marg_info->PreMarginalize();
            marg_info->Marginalize();
            
            if (last_marg_info) {
                delete last_marg_info;
            }
            last_marg_info = marg_info;
        }

        current_window_start += step_size;
    }

    // 6. Save Results
    std::string result_file = output_path + "/ct_trajectory.txt";
    // 10 columns: time, lat, lon, alt, vx, vy, vz, roll, pitch, yaw
    FileSaver saver(result_file, 10); 
    
    double output_interval = config["kf_interval_sec"] ? config["kf_interval_sec"].as<double>() : 0.1;
    for (double t = t_start_global + spline_dt; t < t_end_global - spline_dt; t += output_interval) {
        int k = findControlPointIndex(t, t_start_global, spline_dt, (int)control_points.size());
        if (k < 0 || k + 3 >= (int)control_points.size()) continue;

        double u = (t - (t_start_global + k * spline_dt)) / spline_dt;
        auto res = BSplineEvaluator::Evaluate<double>(u, spline_dt, 
            control_points[k].pose(), control_points[k+1].pose(), 
            control_points[k+2].pose(), control_points[k+3].pose());
        
        // Position: ENU -> BLH
        Vector3d enu_pos = res.pose.translation();
        Vector3d blh = earth.local2global(valid_gnss.front().blh, enu_pos);
        blh[0] *= R2D; // Rad to Deg
        blh[1] *= R2D;

        // Velocity: ENU
        Vector3d vel = res.v_world;

        // Attitude: ENU Quaternion -> Euler (Deg)
        Vector3d euler = Rotation::quaternion2euler(res.pose.so3().unit_quaternion());
        euler *= R2D;

        saver.dump({t, blh[0], blh[1], blh[2], vel.x(), vel.y(), vel.z(), euler[0], euler[1], euler[2]});
    }

    saver.close(); // Ensure file is flushed before python script reads it

    LOG(INFO) << "Trajectory saved to: " << result_file;

    // Save Errors for each IMU
    for (auto& processor : imu_processors) {
        processor->SaveErrors(output_path, control_points, spline_dt, t_start_global);
    }

    // 7. Comparison Script
        if (config["comparison"] && config["comparison"]["enable"].as<bool>()) {
            std::string python_exe = "python";
            std::string script = config["comparison"]["python_script"].as<std::string>();        std::string truth = config["comparison"]["truth_file"].as<std::string>();
        
        std::string cmd = python_exe + " " + script + " --result " + result_file + " --truth " + truth;
        LOG(INFO) << "Running comparison: " << cmd;
        int ret = std::system(cmd.c_str());
        if (ret != 0) LOG(WARNING) << "Comparison script returned non-zero code: " << ret;
    }

    return 0;
}
