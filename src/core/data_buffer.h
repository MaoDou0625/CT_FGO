#ifndef OB_GINS_CORE_DATA_BUFFER_H
#define OB_GINS_CORE_DATA_BUFFER_H

#include <vector>
#include <deque>
#include <map>
#include <string>
#include <mutex>
#include "src/common/types.h"

namespace ob_gins {

class DataBuffer {
public:
    DataBuffer() = default;

    // 添加 GNSS 数据
    void AddGnssData(const GNSS& data);

    // 添加 IMU 数据 (支持多个 IMU)
    void AddImuData(const std::string& imu_name, const IMU& data);

    // 获取特定时间窗口的 GNSS 数据
    std::vector<GNSS> GetGnssData(double t_start, double t_end) const;

    // 获取特定时间窗口的 IMU 数据
    std::vector<IMU> GetImuData(const std::string& imu_name, double t_start, double t_end) const;

    // 移除早于给定时间的数据，释放内存
    void PopOldData(double t_remove_before);

    // 检查缓冲区是否包含指定时间段的数据
    bool HasGnssDataUntil(double t_end) const;
    bool HasImuDataUntil(const std::string& imu_name, double t_end) const;

    // 获取当前缓冲区的起止时间（以GNSS为基准或融合所有传感器）
    double GetEarliestTime() const;
    double GetLatestTime() const;

private:
    mutable std::mutex mutex_;
    std::deque<GNSS> gnss_buffer_;
    std::map<std::string, std::deque<IMU>> imu_buffers_;
};

} // namespace ob_gins

#endif // OB_GINS_CORE_DATA_BUFFER_H
