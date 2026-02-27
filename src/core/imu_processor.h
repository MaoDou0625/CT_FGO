#ifndef OB_GINS_CORE_IMU_PROCESSOR_H
#define OB_GINS_CORE_IMU_PROCESSOR_H

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <ceres/ceres.h>

#include "src/common/types.h"
#include "src/core/data_buffer.h"
#include "src/spline/BSplineEvaluator.h"
#include "src/core/wheel_mechanization.h"

namespace ob_gins {

// 辅助函数：为时间 t 寻找控制点索引
int findControlPointIndex(double t, double t0, double dt, int max_idx);

// 抽象IMU处理器基类
class ImuProcessor {
public:
    virtual ~ImuProcessor() = default;

    // Factory method to create specific processor based on type
    static std::unique_ptr<ImuProcessor> Create(const std::string& type);

    virtual bool LoadConfig(const YAML::Node& config_node, const std::string& imu_name) = 0;
    
    // Instead of reading file, fetch data from buffer
    virtual bool FetchDataFromBuffer(const class DataBuffer& buffer, double t_start, double t_end) = 0;

    virtual void AddFactors(ceres::Problem& problem, 
                            std::vector<spline::ControlPoint>& control_points, 
                            double spline_dt, double t0_spline,
                            const Eigen::Vector3d& gravity_vec, 
                            const Eigen::Vector3d& omega_ie_local,
                            double t_window_start, double t_window_end) = 0;
    
    virtual void AddBiasFactors(ceres::Problem& problem, 
                                std::vector<spline::ControlPoint>& control_points, 
                                double spline_dt,
                                double t_window_start, double t_window_end) = 0;

    virtual std::vector<double*> GetVariablesToDrop(double t_drop_start, double t_drop_end, const std::vector<spline::ControlPoint>& cps);

    const std::vector<IMU>& GetImuData() const { return valid_imu_data_; }
    const std::string& GetName() const { return name_; }
    const std::string& GetFilePath() const { return file_path_; }
    int GetColumns() const { return columns_; }
    double GetRateHz() const { return rate_hz_; }

    double* GetLeverArmData() { return l_body_sensor_.data(); }

    virtual void SaveErrors(const std::string& output_path, const std::vector<spline::ControlPoint>& control_points, double spline_dt, double t_start_global);

protected:
    std::string name_;
    std::string file_path_;
    int columns_;
    double rate_hz_;
    std::vector<IMU> all_imu_data_;
    std::vector<IMU> valid_imu_data_;
    Eigen::Vector3d l_body_sensor_ = Eigen::Vector3d::Zero();

    double acc_noise_ = 1.0e-2;
    double gyr_noise_ = 1.0e-3;
    double acc_bias_rw_ = 1.0e-4;
    double gyr_bias_rw_ = 1.0e-5;
    double acc_corr_time_ = 3600.0; // 默认 1 小时
    double gyr_corr_time_ = 3600.0;

    bool FetchDataFromBufferInternal(const DataBuffer& buffer, double t_start, double t_end);
    Eigen::Vector3d LoadLeverArm(const YAML::Node& config_node, const std::string& key);
    void LoadExtrinsics(const YAML::Node& config_node);
    void LoadImuNoise(const YAML::Node& config_node);

    // IMU biases (one per control point)
    std::vector<Eigen::Vector3d> bg_;
    std::vector<Eigen::Vector3d> ba_;

    // Extrinsics: Rotation from Body to IMU
    Eigen::Quaterniond q_body_imu_initial_ = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond q_body_imu_ = Eigen::Quaterniond::Identity();

    // Time offset (t_sys = t_imu + td)
    double td_ = 0.0;
};

// 标准IMU处理器
class StandardImuProcessor : public ImuProcessor {
public:
    bool LoadConfig(const YAML::Node& config_node, const std::string& imu_name) override;
    bool FetchDataFromBuffer(const class DataBuffer& buffer, double t_start, double t_end) override;
    void AddFactors(ceres::Problem& problem, 
                    std::vector<spline::ControlPoint>& control_points, 
                    double spline_dt, double t0_spline,
                    const Eigen::Vector3d& gravity_vec, 
                    const Eigen::Vector3d& omega_ie_local,
                    double t_window_start, double t_window_end) override;

    void AddBiasFactors(ceres::Problem& problem, 
                        std::vector<spline::ControlPoint>& control_points, 
                        double spline_dt,
                        double t_window_start, double t_window_end) override;
};

// 轮式IMU处理器
class WheelImuProcessor : public ImuProcessor {
public:
    bool LoadConfig(const YAML::Node& config_node, const std::string& imu_name) override;
    bool FetchDataFromBuffer(const class DataBuffer& buffer, double t_start, double t_end) override;
    void AddFactors(ceres::Problem& problem, 
                    std::vector<spline::ControlPoint>& control_points, 
                    double spline_dt, double t0_spline,
                    const Eigen::Vector3d& gravity_vec, 
                    const Eigen::Vector3d& omega_ie_local,
                    double t_window_start, double t_window_end) override;

    void AddBiasFactors(ceres::Problem& problem, 
                        std::vector<spline::ControlPoint>& control_points, 
                        double spline_dt,
                        double t_window_start, double t_window_end) override;

    std::vector<double*> GetVariablesToDrop(double t_drop_start, double t_drop_end, const std::vector<spline::ControlPoint>& cps) override;

    void SaveErrors(const std::string& output_path, const std::vector<spline::ControlPoint>& control_points, double spline_dt, double t_start_global) override;

    // Wheel Phase Control Points
    std::vector<double> wheel_phases_; // One per control point

private:
    // Mechanization
    WheelMechanization mechanization_;
    std::vector<Eigen::Quaterniond> integrated_attitudes_;

    std::string side_;
    // Now optimization variables
    Eigen::Vector3d l_sensor_odopoint_;
    
    // Initial values for Priors
    Eigen::Vector3d l_sensor_odopoint_initial_;
    double wheel_radius_initial_ = 0.3;
    
    // Prior constraints (Standard Deviations)
    double prior_radius_std_ = 0.005; // Default 5mm
    double prior_lever_std_ = 0.02;   // Default 2cm

    double wheel_radius_ = 0.3;
    double speed_weight_ = 1.0;
    double nhc_weight_ = 1.0;
    double att_weight_roll_ = 0.0;
    double att_weight_pitch_ = 0.0;
    
    // Misalignment params [kx, ky]
    Eigen::Vector2d misalignment_xy_ = Eigen::Vector2d::Zero();
};

} // namespace ob_gins
#endif // OB_GINS_CORE_IMU_PROCESSOR_H