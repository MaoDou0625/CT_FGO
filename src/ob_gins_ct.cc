#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sophus/se3.hpp>
#include <yaml-cpp/yaml.h>

DECLARE_bool(logtostderr);
DECLARE_int32(v);

#include "src/common/angle.h"
#include "src/common/earth.h"
#include "src/common/rotation.h"
#include "src/common/types.h"
#include "src/core/imu_processor.h"
#include "src/factors/BiasRandomWalkFactor.h"
#include "src/factors/ContinuousGnssFactor.h"
#include "src/factors/PriorFactors.h"
#include "src/fileio/filesaver.h"
#include "src/fileio/gnssfileloader.h"
#include "src/spline/BSplineEvaluator.h"
#include "src/spline/SophusSE3Manifold.h"
#include "src/spline/SplineInitializer.h"

using namespace ob_gins;
using namespace ob_gins::factors;
using namespace ob_gins::spline;

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

struct GnssBiasConfig {
    bool enable = true;
    double random_walk_sigma = 0.5;
    double correlation_time_sec = 120.0;
    double initial_bias_std = 5.0;
};

struct GnssInnovationGateConfig {
    bool enable = true;
    bool only_anomaly_context = true;
    double horizontal_threshold_m = 12.0;
    double vertical_threshold_m = 6.0;
    double sigma_threshold = 4.0;
    double cooldown_sec = 8.0;
    int reacquire_consecutive = 2;
    double reacquire_horizontal_threshold_m = 6.0;
    double reacquire_vertical_threshold_m = 3.0;
    double reacquire_sigma_threshold = 2.5;
    double rejected_scale = 0.05;
    int warmup_iterations = 4;
    int anomaly_context_samples = 3;
};

struct GnssQualitySample {
    double weight_scale = 1.0;
    double dt_prev = 0.0;
    double dt_next = 0.0;
    double dist_prev = 0.0;
    double dist_next = 0.0;
    double speed_prev = 0.0;
    double speed_next = 0.0;
    double innovation_horizontal = 0.0;
    double innovation_vertical = 0.0;
    bool gap_adjacent = false;
    bool jump_adjacent = false;
    bool std_outlier = false;
    bool anomaly_context = false;
    bool innovation_rejected = false;
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

GnssBiasConfig LoadGnssBiasConfig(const YAML::Node& config) {
    GnssBiasConfig bias;
    if (!config["gnss_bias"]) {
        return bias;
    }

    const auto& node = config["gnss_bias"];
    if (node["enable"]) bias.enable = node["enable"].as<bool>();
    if (node["random_walk_sigma"]) bias.random_walk_sigma = node["random_walk_sigma"].as<double>();
    if (node["correlation_time_sec"]) bias.correlation_time_sec = node["correlation_time_sec"].as<double>();
    if (node["initial_bias_std"]) bias.initial_bias_std = node["initial_bias_std"].as<double>();
    return bias;
}

GnssInnovationGateConfig LoadGnssInnovationGateConfig(const YAML::Node& config, int max_iterations) {
    GnssInnovationGateConfig gate;
    gate.warmup_iterations = std::max(2, max_iterations / 2);
    if (!config["gnss_innovation_gate"]) {
        return gate;
    }

    const auto& node = config["gnss_innovation_gate"];
    if (node["enable"]) gate.enable = node["enable"].as<bool>();
    if (node["only_anomaly_context"]) gate.only_anomaly_context = node["only_anomaly_context"].as<bool>();
    if (node["horizontal_threshold_m"]) gate.horizontal_threshold_m = node["horizontal_threshold_m"].as<double>();
    if (node["vertical_threshold_m"]) gate.vertical_threshold_m = node["vertical_threshold_m"].as<double>();
    if (node["sigma_threshold"]) gate.sigma_threshold = node["sigma_threshold"].as<double>();
    if (node["cooldown_sec"]) gate.cooldown_sec = node["cooldown_sec"].as<double>();
    if (node["reacquire_consecutive"]) gate.reacquire_consecutive = node["reacquire_consecutive"].as<int>();
    if (node["reacquire_horizontal_threshold_m"]) gate.reacquire_horizontal_threshold_m = node["reacquire_horizontal_threshold_m"].as<double>();
    if (node["reacquire_vertical_threshold_m"]) gate.reacquire_vertical_threshold_m = node["reacquire_vertical_threshold_m"].as<double>();
    if (node["reacquire_sigma_threshold"]) gate.reacquire_sigma_threshold = node["reacquire_sigma_threshold"].as<double>();
    if (node["rejected_scale"]) gate.rejected_scale = node["rejected_scale"].as<double>();
    if (node["warmup_iterations"]) gate.warmup_iterations = node["warmup_iterations"].as<int>();
    if (node["anomaly_context_samples"]) gate.anomaly_context_samples = node["anomaly_context_samples"].as<int>();
    gate.warmup_iterations = std::max(1, gate.warmup_iterations);
    gate.reacquire_consecutive = std::max(1, gate.reacquire_consecutive);
    gate.anomaly_context_samples = std::max(0, gate.anomaly_context_samples);
    return gate;
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

void ExpandGnssAnomalyContext(std::vector<GnssQualitySample>* quality, int context_samples) {
    if (!quality || quality->empty()) {
        return;
    }

    const size_t span = static_cast<size_t>(std::max(0, context_samples));
    for (size_t i = 0; i < quality->size(); ++i) {
        if (!(*quality)[i].gap_adjacent && !(*quality)[i].jump_adjacent && !(*quality)[i].std_outlier) {
            continue;
        }
        const size_t start = (i > span) ? (i - span) : 0;
        const size_t end = std::min(quality->size() - 1, i + span);
        for (size_t j = start; j <= end; ++j) {
            (*quality)[j].anomaly_context = true;
        }
    }
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
        if (quality[i].innovation_rejected) flags += 8.0;

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

bool EvaluateSplinePosition(const std::vector<ControlPoint>& control_points,
                            double t,
                            double t0_spline,
                            double spline_dt,
                            Vector3d* enu_pos,
                            Vector3d* vel_world = nullptr) {
    int k = findControlPointIndex(t, t0_spline, spline_dt, static_cast<int>(control_points.size()));
    if (k < 0 || k + 3 >= static_cast<int>(control_points.size())) {
        return false;
    }

    double u = (t - control_points[k].timestamp()) / spline_dt;
    auto res = BSplineEvaluator::Evaluate<double>(
        u,
        spline_dt,
        control_points[k].pose(),
        control_points[k + 1].pose(),
        control_points[k + 2].pose(),
        control_points[k + 3].pose());
    if (enu_pos) {
        *enu_pos = res.pose.translation();
    }
    if (vel_world) {
        *vel_world = res.v_world;
    }
    return true;
}

Vector3d InterpolateGnssBias(const std::vector<Vector3d>& gnss_biases, int k, double u) {
    if (gnss_biases.empty()) {
        return Vector3d::Zero();
    }
    int k1 = std::min(k + 1, static_cast<int>(gnss_biases.size()) - 1);
    double clamped_u = std::clamp(u, 0.0, 1.0);
    return (1.0 - clamped_u) * gnss_biases[k] + clamped_u * gnss_biases[k1];
}

void SaveGnssBiasProfile(const std::string& output_path,
                         const std::vector<ControlPoint>& control_points,
                         const std::vector<Vector3d>& gnss_biases) {
    if (control_points.size() != gnss_biases.size()) {
        return;
    }

    FileSaver saver(output_path + "/gnss_bias_knots.txt", 4);
    for (size_t i = 0; i < control_points.size(); ++i) {
        saver.dump({
            control_points[i].timestamp(),
            gnss_biases[i].x(),
            gnss_biases[i].y(),
            gnss_biases[i].z(),
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

    YAML::Node config;
    try {
        config = YAML::LoadFile(argv[1]);
    } catch (YAML::Exception& e) {
        LOG(ERROR) << "Failed to load config: " << e.what();
        return -1;
    }

    if (config["debug"] && config["debug"]["level"]) {
        FLAGS_v = config["debug"]["level"].as<int>();
    }

    std::string output_path = config["outputpath"].as<std::string>();
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directories(output_path);
    }

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

    std::vector<std::unique_ptr<ImuProcessor>> imu_processors;
    for (YAML::const_iterator it = config.begin(); it != config.end(); ++it) {
        std::string key = it->first.as<std::string>();
        if (!it->second.IsMap() || !it->second["type"]) {
            continue;
        }

        std::string type = it->second["type"].as<std::string>();
        auto processor = ImuProcessor::Create(type);
        if (!processor) {
            continue;
        }
        if (!processor->LoadConfig(it->second, key)) {
            LOG(ERROR) << "Failed to load config for IMU: " << key;
            continue;
        }
        LOG(INFO) << "Loaded IMU: " << key << " (Type: " << type << ")";
        imu_processors.push_back(std::move(processor));
    }

    if (imu_processors.empty()) {
        LOG(ERROR) << "No IMU configurations found or loaded.";
        return -1;
    }

    for (const auto& processor : imu_processors) {
        if (!processor->LoadData(t_start_global, t_end_global)) {
            LOG(ERROR) << "Failed to load data for IMU: " << processor->GetName();
            return -1;
        }
        if (!processor->GetImuData().empty()) {
            t_start_global = std::max(t_start_global, processor->GetImuData().front().time);
            t_end_global = std::min(t_end_global, processor->GetImuData().back().time);
        }
    }

    bool origin_set = false;
    std::vector<GNSS> valid_gnss;
    for (const auto& gnss : gnss_data) {
        if (gnss.time < t_start_global || gnss.time > t_end_global) {
            continue;
        }
        if (!origin_set) {
            origin_set = true;
        }
        valid_gnss.push_back(gnss);
    }

    if (!origin_set || valid_gnss.empty()) {
        LOG(ERROR) << "No valid GNSS data in final time window [" << t_start_global << ", " << t_end_global << "].";
        return -1;
    }

    LOG(INFO) << "Final Time window: " << std::fixed << t_start_global << " to " << t_end_global
              << " (Duration: " << (t_end_global - t_start_global) << "s)";

    double spline_dt = config["kf_interval_sec"] ? config["kf_interval_sec"].as<double>() : 1.0;
    int total_iterations = config["num_iterations"] ? config["num_iterations"].as<int>() : 20;

    Earth earth;
    Vector3d gravity_l;
    Vector3d omega_ie_l;
    if (config["isearth"] && config["isearth"].as<bool>()) {
        double g = earth.gravity(valid_gnss.front().blh);
        gravity_l << 0.0, 0.0, -g;
        omega_ie_l = earth.iewn(valid_gnss.front().blh(0));
    } else {
        gravity_l << 0.0, 0.0, -9.80665;
        omega_ie_l.setZero();
    }

    std::vector<GNSS> gnss_enu = valid_gnss;
    std::vector<std::pair<double, Sophus::SE3d>> path_for_init;
    for (auto& g : gnss_enu) {
        g.blh = earth.global2local(valid_gnss.front().blh, g.blh);
        path_for_init.emplace_back(g.time, Sophus::SE3d(Eigen::Quaterniond::Identity(), g.blh));
    }

    std::vector<ControlPoint> control_points = SplineInitializer::InitializeFromPath(path_for_init, spline_dt);
    std::vector<Vector3d> gnss_biases(control_points.size(), Vector3d::Zero());
    Eigen::Vector3d gnss_lever_arm = Eigen::Vector3d::Zero();

    GnssQualityConfig gnss_quality_config = LoadGnssQualityConfig(config);
    GnssBiasConfig gnss_bias_config = LoadGnssBiasConfig(config);
    GnssInnovationGateConfig gnss_gate_config = LoadGnssInnovationGateConfig(config, total_iterations);
    std::vector<GnssQualitySample> gnss_quality = AnalyzeGnssQuality(gnss_enu, gnss_quality_config);
    ExpandGnssAnomalyContext(&gnss_quality, gnss_gate_config.anomaly_context_samples);

    auto log_gnss_quality_summary = [&](const std::vector<GnssQualitySample>& quality, const std::string& stage_name) {
        size_t gap_count = 0;
        size_t jump_count = 0;
        size_t std_outlier_count = 0;
        size_t anomaly_context_count = 0;
        size_t innovation_rejected_count = 0;
        for (const auto& sample : quality) {
            if (sample.gap_adjacent) ++gap_count;
            if (sample.jump_adjacent) ++jump_count;
            if (sample.std_outlier) ++std_outlier_count;
            if (sample.anomaly_context) ++anomaly_context_count;
            if (sample.innovation_rejected) ++innovation_rejected_count;
        }
        LOG(INFO) << "GNSS quality summary [" << stage_name << "]: gap_adjacent=" << gap_count
                  << ", jump_adjacent=" << jump_count
                  << ", std_outlier=" << std_outlier_count
                  << ", anomaly_context=" << anomaly_context_count
                  << ", innovation_rejected=" << innovation_rejected_count;
    };

    auto compute_stationary_scale = [&](double gnss_time) {
        double max_wheel_speed = 0.0;
        int wheel_imu_count = 0;
        for (const auto& processor : imu_processors) {
            if (processor->GetName().find("imu_main") != std::string::npos) {
                continue;
            }
            const auto& imu_data = processor->GetImuData();
            auto it = std::lower_bound(
                imu_data.begin(),
                imu_data.end(),
                gnss_time,
                [](const IMU& a, double t) { return a.time < t; });
            if (it == imu_data.end() || it == imu_data.begin()) {
                continue;
            }

            double speed_sum = 0.0;
            int count = 0;
            auto start_it = (it - imu_data.begin() >= 10) ? (it - 10) : imu_data.begin();
            auto end_it = (imu_data.end() - it >= 10) ? (it + 10) : imu_data.end();
            for (auto it2 = start_it; it2 != end_it; ++it2) {
                speed_sum += std::abs(it2->odovel / it2->dt);
                ++count;
            }
            if (count > 0) {
                max_wheel_speed = std::max(max_wheel_speed, speed_sum / count);
                ++wheel_imu_count;
            }
        }
        return (wheel_imu_count > 0 && max_wheel_speed < 0.05) ? 0.001 : 1.0;
    };

    auto add_pose_blocks = [&](ceres::Problem& problem) {
        for (auto& cp : control_points) {
            problem.AddParameterBlock(cp.pose_data(), 7);
            problem.SetManifold(cp.pose_data(), new SophusSE3Manifold());
        }
    };

    auto add_gnss_bias_blocks = [&](ceres::Problem& problem) {
        for (auto& bias : gnss_biases) {
            problem.AddParameterBlock(bias.data(), 3);
            if (!gnss_bias_config.enable) {
                problem.SetParameterBlockConstant(bias.data());
            }
        }

        if (!gnss_biases.empty()) {
            problem.AddResidualBlock(
                LeverArmPriorFactor::Create(Vector3d::Zero(), gnss_bias_config.initial_bias_std),
                nullptr,
                gnss_biases.front().data());
        }

        if (gnss_bias_config.enable) {
            for (size_t i = 0; i + 1 < gnss_biases.size(); ++i) {
                problem.AddResidualBlock(
                    BiasRandomWalkFactor::Create(
                        spline_dt,
                        gnss_bias_config.random_walk_sigma,
                        gnss_bias_config.correlation_time_sec),
                    nullptr,
                    gnss_biases[i].data(),
                    gnss_biases[i + 1].data());
            }
        }
    };

    auto add_gnss_factors = [&](ceres::Problem& problem, const std::vector<GnssQualitySample>& quality) {
        problem.AddParameterBlock(gnss_lever_arm.data(), 3);
        problem.SetParameterBlockConstant(gnss_lever_arm.data());

        for (size_t gnss_idx = 0; gnss_idx < gnss_enu.size(); ++gnss_idx) {
            const auto& gnss = gnss_enu[gnss_idx];
            int k = findControlPointIndex(gnss.time, t_start_global, spline_dt, static_cast<int>(control_points.size()));
            if (k < 0 || k + 3 >= static_cast<int>(control_points.size()) || k + 1 >= static_cast<int>(gnss_biases.size())) {
                continue;
            }

            Eigen::Vector3d base_std = gnss.std.cwiseMax(Eigen::Vector3d::Constant(0.05));
            Matrix3d current_gnss_sqrt_info = base_std.cwiseInverse().asDiagonal();
            current_gnss_sqrt_info *= compute_stationary_scale(gnss.time);
            current_gnss_sqrt_info *= quality[gnss_idx].weight_scale;

            auto* factor = ContinuousGnssFactor::Create(
                gnss.time,
                spline_dt,
                control_points[k].timestamp(),
                gnss.blh,
                current_gnss_sqrt_info);
            problem.AddResidualBlock(
                factor,
                nullptr,
                control_points[k].pose_data(),
                control_points[k + 1].pose_data(),
                control_points[k + 2].pose_data(),
                control_points[k + 3].pose_data(),
                gnss_biases[k].data(),
                gnss_biases[k + 1].data(),
                gnss_lever_arm.data());
        }
    };

    auto add_all_imu_factors = [&](ceres::Problem& problem) {
        for (auto& processor : imu_processors) {
            processor->AddFactors(problem, control_points, spline_dt, t_start_global, gravity_l, omega_ie_l);
            processor->AddBiasFactors(problem, control_points, spline_dt);
        }
    };

    auto build_problem = [&](ceres::Problem& problem, const std::vector<GnssQualitySample>& quality) {
        add_pose_blocks(problem);
        add_gnss_bias_blocks(problem);
        add_gnss_factors(problem, quality);
        add_all_imu_factors(problem);
    };

    auto apply_innovation_gate = [&](std::vector<GnssQualitySample>& quality) {
        if (!gnss_gate_config.enable) {
            return;
        }

        double cooldown_until = -std::numeric_limits<double>::infinity();
        int reacquire_good_count = 0;

        for (size_t gnss_idx = 0; gnss_idx < gnss_enu.size(); ++gnss_idx) {
            auto& sample = quality[gnss_idx];
            const auto& gnss = gnss_enu[gnss_idx];
            int k = findControlPointIndex(gnss.time, t_start_global, spline_dt, static_cast<int>(control_points.size()));
            if (k < 0 || k + 1 >= static_cast<int>(gnss_biases.size())) {
                continue;
            }

            Vector3d pred_pos;
            if (!EvaluateSplinePosition(control_points, gnss.time, t_start_global, spline_dt, &pred_pos)) {
                continue;
            }

            double u = (gnss.time - control_points[k].timestamp()) / spline_dt;
            Vector3d pred_with_bias = pred_pos + InterpolateGnssBias(gnss_biases, k, u);
            Vector3d residual = pred_with_bias - gnss.blh;
            sample.innovation_horizontal = residual.head<2>().norm();
            sample.innovation_vertical = std::abs(residual.z());

            double sigma_h = std::max({gnss.std.x(), gnss.std.y(), 0.5});
            double sigma_v = std::max(gnss.std.z(), 0.5);
            bool innovation_bad =
                sample.innovation_horizontal > gnss_gate_config.horizontal_threshold_m ||
                sample.innovation_vertical > gnss_gate_config.vertical_threshold_m ||
                sample.innovation_horizontal / sigma_h > gnss_gate_config.sigma_threshold ||
                sample.innovation_vertical / sigma_v > gnss_gate_config.sigma_threshold;
            bool innovation_good =
                sample.innovation_horizontal < gnss_gate_config.reacquire_horizontal_threshold_m &&
                sample.innovation_vertical < gnss_gate_config.reacquire_vertical_threshold_m &&
                sample.innovation_horizontal / sigma_h < gnss_gate_config.reacquire_sigma_threshold &&
                sample.innovation_vertical / sigma_v < gnss_gate_config.reacquire_sigma_threshold;

            bool reject_sample = false;
            const bool gate_armed =
                (gnss_gate_config.only_anomaly_context ? sample.anomaly_context : true) ||
                gnss.time < cooldown_until;
            if (!gate_armed) {
                continue;
            }

            if (gnss.time < cooldown_until) {
                if (innovation_good) {
                    ++reacquire_good_count;
                    if (reacquire_good_count >= gnss_gate_config.reacquire_consecutive) {
                        cooldown_until = -std::numeric_limits<double>::infinity();
                        reacquire_good_count = 0;
                    } else {
                        reject_sample = true;
                    }
                } else {
                    cooldown_until = std::max(cooldown_until, gnss.time + gnss_gate_config.cooldown_sec);
                    reacquire_good_count = 0;
                    reject_sample = true;
                }
            } else if (sample.anomaly_context && innovation_bad) {
                cooldown_until = gnss.time + gnss_gate_config.cooldown_sec;
                reacquire_good_count = 0;
                reject_sample = true;
            }

            if (reject_sample) {
                sample.weight_scale *= gnss_gate_config.rejected_scale;
                sample.innovation_rejected = true;
            }
        }
    };

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    std::vector<GnssQualitySample> final_gnss_quality = gnss_quality;
    log_gnss_quality_summary(gnss_quality, "self_quality");

    if (gnss_gate_config.enable) {
        options.max_num_iterations = std::min(total_iterations, gnss_gate_config.warmup_iterations);
        ceres::Problem warmup_problem;
        build_problem(warmup_problem, gnss_quality);
        ceres::Solver::Summary warmup_summary;
        ceres::Solve(options, &warmup_problem, &warmup_summary);
        LOG(INFO) << "Warmup solve: " << warmup_summary.BriefReport();

        final_gnss_quality = gnss_quality;
        apply_innovation_gate(final_gnss_quality);
        log_gnss_quality_summary(final_gnss_quality, "innovation_gated");
    }

    SaveGnssQualityProfile(output_path, valid_gnss, final_gnss_quality);
    options.max_num_iterations = total_iterations;
    ceres::Problem final_problem;
    build_problem(final_problem, final_gnss_quality);
    ceres::Solve(options, &final_problem, &summary);
    LOG(INFO) << "Final solve: " << summary.BriefReport();

    std::string result_file = output_path + "/ct_trajectory.txt";
    FileSaver saver(result_file, 10);
    double output_interval = config["kf_interval_sec"] ? config["kf_interval_sec"].as<double>() : 0.1;
    for (double t = t_start_global + spline_dt; t < t_end_global - spline_dt; t += output_interval) {
        int k = findControlPointIndex(t, t_start_global, spline_dt, static_cast<int>(control_points.size()));
        if (k < 0 || k + 3 >= static_cast<int>(control_points.size())) {
            continue;
        }

        double u = (t - control_points[k].timestamp()) / spline_dt;
        auto res = BSplineEvaluator::Evaluate<double>(
            u,
            spline_dt,
            control_points[k].pose(),
            control_points[k + 1].pose(),
            control_points[k + 2].pose(),
            control_points[k + 3].pose());

        Vector3d enu_pos = res.pose.translation();
        Vector3d blh = earth.local2global(valid_gnss.front().blh, enu_pos);
        blh[0] *= R2D;
        blh[1] *= R2D;

        Vector3d vel = res.v_world;
        Vector3d euler = Rotation::quaternion2euler(res.pose.so3().unit_quaternion());
        euler *= R2D;

        saver.dump({t, blh[0], blh[1], blh[2], vel.x(), vel.y(), vel.z(), euler[0], euler[1], euler[2]});
    }
    saver.close();
    SaveGnssBiasProfile(output_path, control_points, gnss_biases);
    LOG(INFO) << "Trajectory saved to: " << result_file;

    for (auto& processor : imu_processors) {
        processor->SaveErrors(output_path, control_points, spline_dt, t_start_global);
    }

    if (config["comparison"] && config["comparison"]["enable"].as<bool>()) {
        std::string python_exe = "python";
        std::string script = config["comparison"]["python_script"].as<std::string>();
        std::string truth = config["comparison"]["truth_file"].as<std::string>();

        std::string cmd = python_exe + " " + script + " --result " + result_file + " --truth " + truth;
        LOG(INFO) << "Running comparison: " << cmd;
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            LOG(WARNING) << "Comparison script returned non-zero code: " << ret;
        }
    }

    return 0;
}
