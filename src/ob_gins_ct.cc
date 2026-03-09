#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <limits>

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
#include "src/core/imu_processor.h"

using namespace ob_gins;
using namespace ob_gins::spline;
using namespace ob_gins::factors;

namespace {

struct GnssQualityConfig {
    bool enable = true;
    double gap_threshold_sec = 2.5;
    double jump_distance_threshold_m = 40.0;
    double jump_speed_threshold_mps = 35.0;
    double horizontal_std_threshold_m = 8.0;
    double vertical_std_threshold_m = 12.0;
    double gap_scale = 0.15;
    double jump_scale = 0.05;
    double std_scale = 0.25;
    double min_scale = 0.01;
};

struct GnssQualitySample {
    double weight_scale = 1.0;
    double dt_prev = 0.0;
    double dt_next = 0.0;
    double dist_prev = 0.0;
    double dist_next = 0.0;
    double speed_prev = 0.0;
    double speed_next = 0.0;
    bool gap_adjacent = false;
    bool jump_adjacent = false;
    bool std_outlier = false;
};

GnssQualityConfig LoadGnssQualityConfig(const YAML::Node& config) {
    GnssQualityConfig quality;
    if (!config["gnss_quality"]) {
        return quality;
    }

    const auto& node = config["gnss_quality"];
    if (node["enable"]) quality.enable = node["enable"].as<bool>();
    if (node["gap_threshold_sec"]) quality.gap_threshold_sec = node["gap_threshold_sec"].as<double>();
    if (node["jump_distance_threshold_m"]) quality.jump_distance_threshold_m = node["jump_distance_threshold_m"].as<double>();
    if (node["jump_speed_threshold_mps"]) quality.jump_speed_threshold_mps = node["jump_speed_threshold_mps"].as<double>();
    if (node["horizontal_std_threshold_m"]) quality.horizontal_std_threshold_m = node["horizontal_std_threshold_m"].as<double>();
    if (node["vertical_std_threshold_m"]) quality.vertical_std_threshold_m = node["vertical_std_threshold_m"].as<double>();
    if (node["gap_scale"]) quality.gap_scale = node["gap_scale"].as<double>();
    if (node["jump_scale"]) quality.jump_scale = node["jump_scale"].as<double>();
    if (node["std_scale"]) quality.std_scale = node["std_scale"].as<double>();
    if (node["min_scale"]) quality.min_scale = node["min_scale"].as<double>();
    return quality;
}

double DistanceMeters(const Vector3d& a, const Vector3d& b) {
    return (a - b).norm();
}

std::vector<GnssQualitySample> AnalyzeGnssQuality(const std::vector<GNSS>& gnss_enu,
                                                  const GnssQualityConfig& config) {
    std::vector<GnssQualitySample> quality(gnss_enu.size());
    if (!config.enable) {
        return quality;
    }

    for (size_t i = 0; i < gnss_enu.size(); ++i) {
        auto& sample = quality[i];

        if (i > 0) {
            sample.dt_prev = gnss_enu[i].time - gnss_enu[i - 1].time;
            sample.dist_prev = DistanceMeters(gnss_enu[i].blh, gnss_enu[i - 1].blh);
            if (sample.dt_prev > 1.0e-6) {
                sample.speed_prev = sample.dist_prev / sample.dt_prev;
            }
        }
        if (i + 1 < gnss_enu.size()) {
            sample.dt_next = gnss_enu[i + 1].time - gnss_enu[i].time;
            sample.dist_next = DistanceMeters(gnss_enu[i + 1].blh, gnss_enu[i].blh);
            if (sample.dt_next > 1.0e-6) {
                sample.speed_next = sample.dist_next / sample.dt_next;
            }
        }

        sample.gap_adjacent =
            sample.dt_prev > config.gap_threshold_sec || sample.dt_next > config.gap_threshold_sec;
        sample.jump_adjacent =
            sample.dist_prev > config.jump_distance_threshold_m ||
            sample.dist_next > config.jump_distance_threshold_m ||
            sample.speed_prev > config.jump_speed_threshold_mps ||
            sample.speed_next > config.jump_speed_threshold_mps;
        sample.std_outlier =
            std::max(gnss_enu[i].std.x(), gnss_enu[i].std.y()) > config.horizontal_std_threshold_m ||
            gnss_enu[i].std.z() > config.vertical_std_threshold_m;

        double scale = 1.0;
        if (sample.gap_adjacent) scale *= config.gap_scale;
        if (sample.jump_adjacent) scale *= config.jump_scale;
        if (sample.std_outlier) scale *= config.std_scale;
        sample.weight_scale = std::max(config.min_scale, scale);
    }

    return quality;
}

void SaveGnssQualityProfile(const std::string& output_path,
                            const std::vector<GNSS>& gnss_global,
                            const std::vector<GnssQualitySample>& quality) {
    if (gnss_global.size() != quality.size()) {
        return;
    }

    FileSaver saver(output_path + "/gnss_weight_profile.txt", 11);
    for (size_t i = 0; i < gnss_global.size(); ++i) {
        double flags = 0.0;
        if (quality[i].gap_adjacent) flags += 1.0;
        if (quality[i].jump_adjacent) flags += 2.0;
        if (quality[i].std_outlier) flags += 4.0;

        saver.dump({
            gnss_global[i].time,
            gnss_global[i].blh.x() * R2D,
            gnss_global[i].blh.y() * R2D,
            gnss_global[i].blh.z(),
            gnss_global[i].std.x(),
            gnss_global[i].std.y(),
            gnss_global[i].std.z(),
            quality[i].weight_scale,
            quality[i].dt_prev,
            quality[i].dt_next,
            flags,
        });
    }
    saver.close();
}

}  // namespace

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

    // 2. Load Data - GNSS First (for initial time window and origin)
    LOG(INFO) << "Loading GNSS data...";
    std::string gnss_path = config["gnssfile"].as<std::string>();
    GnssFileLoader gnss_loader(gnss_path);

    std::vector<GNSS> gnss_data;
    while (!gnss_loader.isEof()) {
        gnss_data.push_back(gnss_loader.next());
    }
    if (gnss_data.empty()) {
        LOG(ERROR) << "Empty GNSS data loaded.";
        return -1;
    }

    double t_start_global = gnss_data.front().time;
    double t_end_global = gnss_data.back().time;

    if (config["starttime"]) t_start_global = std::max(t_start_global, config["starttime"].as<double>());
    if (config["endtime"]) t_end_global = std::min(t_end_global, config["endtime"].as<double>());

    LOG(INFO) << "Initial Time window from GNSS: " << std::fixed << t_start_global << " to " << t_end_global 
              << " (Duration: " << (t_end_global - t_start_global) << "s)";

    // Store all ImuProcessors
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

    // Load data for all IMUs and adjust global time window
    for (const auto& processor : imu_processors) {
        if (!processor->LoadData(t_start_global, t_end_global)) {
            LOG(ERROR) << "Failed to load data for IMU: " << processor->GetName();
            return -1;
        }
        // Adjust global time window based on actual loaded IMU data
        if (!processor->GetImuData().empty()) {
            t_start_global = std::max(t_start_global, processor->GetImuData().front().time);
            t_end_global = std::min(t_end_global, processor->GetImuData().back().time);
        }
    }
    
    // Filter GNSS data based on final global time window
    GNSS origin_gnss;
    bool origin_set = false;
    std::vector<GNSS> valid_gnss;
    
    for (const auto& gnss : gnss_data) {
        if (gnss.time >= t_start_global) {
            if (!origin_set) {
                origin_gnss = gnss;
                origin_set = true;
            }
            if (gnss.time <= t_end_global) {
                valid_gnss.push_back(gnss);
            }
        }
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

    // 4. Build Optimization Problem
    ceres::Problem problem;
    
    // 初始化样条曲线控制点 (将 GNSS 转换为局部 ENU 进行初始化)
    std::vector<GNSS> gnss_enu = valid_gnss;
    std::vector<std::pair<double, Sophus::SE3d>> path_for_init;

    for (auto& g : gnss_enu) {
        // Use global2local to convert BLH to local frame (ENU/NED)
        // Store result in g.blh temporarily
        g.blh = earth.global2local(valid_gnss.front().blh, g.blh); 
        
        // Prepare path for SplineInitializer
        // Assume identity rotation for initialization if not available
        path_for_init.emplace_back(g.time, Sophus::SE3d(Eigen::Quaterniond::Identity(), g.blh));
    }

    std::vector<ControlPoint> control_points = SplineInitializer::InitializeFromPath(path_for_init, spline_dt);

    // 设置位姿流形
    for (auto& cp : control_points) {
        problem.AddParameterBlock(cp.pose_data(), 7);
        problem.SetManifold(cp.pose_data(), new SophusSE3Manifold());
        // Biases are now managed by ImuProcessors individually
    }

    // 添加 GNSS 因子
    // Body Frame is defined as GNSS Center, so Lever Arm is ZERO.
    Eigen::Vector3d gnss_lever_arm = Eigen::Vector3d::Zero();
    problem.AddParameterBlock(gnss_lever_arm.data(), 3);
    problem.SetParameterBlockConstant(gnss_lever_arm.data());

    GnssQualityConfig gnss_quality_config = LoadGnssQualityConfig(config);
    std::vector<GnssQualitySample> gnss_quality = AnalyzeGnssQuality(gnss_enu, gnss_quality_config);
    SaveGnssQualityProfile(output_path, valid_gnss, gnss_quality);

    size_t gap_count = 0;
    size_t jump_count = 0;
    size_t std_outlier_count = 0;
    for (const auto& sample : gnss_quality) {
        if (sample.gap_adjacent) ++gap_count;
        if (sample.jump_adjacent) ++jump_count;
        if (sample.std_outlier) ++std_outlier_count;
    }
    LOG(INFO) << "GNSS quality summary: gap_adjacent=" << gap_count
              << ", jump_adjacent=" << jump_count
              << ", std_outlier=" << std_outlier_count;

    for (size_t gnss_idx = 0; gnss_idx < gnss_enu.size(); ++gnss_idx) {
        const auto& gnss = gnss_enu[gnss_idx];
        int k = findControlPointIndex(gnss.time, t_start_global, spline_dt, (int)control_points.size());
        if (k < 0 || k + 3 >= (int)control_points.size()) continue;

        // Detect if stationary at gnss.time
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
                    // Average over a window of ~20 samples (around 0.16s at 120Hz)
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

        Eigen::Vector3d base_std = gnss.std.cwiseMax(Eigen::Vector3d::Constant(0.05));
        Matrix3d current_gnss_sqrt_info = base_std.cwiseInverse().asDiagonal();
        if (wheel_imu_count > 0 && max_wheel_speed < 0.05) {
            // If stationary, reduce GNSS weight significantly to prevent position drift
            current_gnss_sqrt_info *= 0.001;
            LOG_EVERY_N(INFO, 100) << "Detected stationary at t=" << gnss.time << " (max_speed=" << max_wheel_speed << "), reducing GNSS weight.";
        } else {
            LOG_EVERY_N(INFO, 100) << "Not stationary at t=" << gnss.time << " (wheel_imu_count=" << wheel_imu_count << ", max_speed=" << max_wheel_speed << ")";
        }

        current_gnss_sqrt_info *= gnss_quality[gnss_idx].weight_scale;

        // Keep spline local-time parameterization consistent with other factors.
        auto* factor = ContinuousGnssFactor::Create(
            gnss.time, spline_dt, control_points[k].timestamp(), gnss.blh, current_gnss_sqrt_info);
        problem.AddResidualBlock(factor, nullptr, 
            control_points[k].pose_data(), control_points[k+1].pose_data(), 
            control_points[k+2].pose_data(), control_points[k+3].pose_data(),
            gnss_lever_arm.data() // GNSS lever arm is zero
        );
    }

    // 添加所有 IMU 约束 (Standard + Wheel)
    for (auto& processor : imu_processors) {
        processor->AddFactors(problem, control_points, spline_dt, t_start_global, gravity_l, omega_ie_l);
        processor->AddBiasFactors(problem, control_points, spline_dt);
    }

    // 5. Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = config["num_iterations"] ? config["num_iterations"].as<int>() : 20;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.BriefReport();

    // 6. Save Results
    std::string result_file = output_path + "/ct_trajectory.txt";
    // 10 columns: time, lat, lon, alt, vx, vy, vz, roll, pitch, yaw
    FileSaver saver(result_file, 10); 
    
    double output_interval = config["kf_interval_sec"] ? config["kf_interval_sec"].as<double>() : 0.1;
    for (double t = t_start_global + spline_dt; t < t_end_global - spline_dt; t += output_interval) {
        int k = findControlPointIndex(t, t_start_global, spline_dt, (int)control_points.size());
        if (k < 0 || k + 3 >= (int)control_points.size()) continue;

        // Keep local parameterization consistent with all factors: u = (t - t0) / dt.
        double u = (t - control_points[k].timestamp()) / spline_dt;
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
