#include "src/core/data_buffer.h"
#include <algorithm>

namespace ob_gins {

void DataBuffer::AddGnssData(const GNSS& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    gnss_buffer_.push_back(data);
}

void DataBuffer::AddImuData(const std::string& imu_name, const IMU& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    imu_buffers_[imu_name].push_back(data);
}

std::vector<GNSS> DataBuffer::GetGnssData(double t_start, double t_end) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<GNSS> result;
    for (const auto& d : gnss_buffer_) {
        if (d.time >= t_start && d.time <= t_end) {
            result.push_back(d);
        } else if (d.time > t_end) {
            break;
        }
    }
    return result;
}

std::vector<IMU> DataBuffer::GetImuData(const std::string& imu_name, double t_start, double t_end) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<IMU> result;
    auto it = imu_buffers_.find(imu_name);
    if (it != imu_buffers_.end()) {
        for (const auto& d : it->second) {
            if (d.time >= t_start && d.time <= t_end) {
                result.push_back(d);
            } else if (d.time > t_end) {
                break;
            }
        }
    }
    return result;
}

void DataBuffer::PopOldData(double t_remove_before) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!gnss_buffer_.empty() && gnss_buffer_.front().time < t_remove_before) {
        gnss_buffer_.pop_front();
    }
    for (auto& pair : imu_buffers_) {
        auto& buffer = pair.second;
        while (!buffer.empty() && buffer.front().time < t_remove_before) {
            buffer.pop_front();
        }
    }
}

bool DataBuffer::HasGnssDataUntil(double t_end) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gnss_buffer_.empty()) return false;
    return gnss_buffer_.back().time >= t_end;
}

bool DataBuffer::HasImuDataUntil(const std::string& imu_name, double t_end) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = imu_buffers_.find(imu_name);
    if (it == imu_buffers_.end() || it->second.empty()) return false;
    return it->second.back().time >= t_end;
}

double DataBuffer::GetEarliestTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    double min_time = 1e15; // A large number
    if (!gnss_buffer_.empty()) {
        min_time = std::min(min_time, gnss_buffer_.front().time);
    }
    for (const auto& pair : imu_buffers_) {
        if (!pair.second.empty()) {
            min_time = std::min(min_time, pair.second.front().time);
        }
    }
    return min_time;
}

double DataBuffer::GetLatestTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    double max_time = -1.0;
    if (!gnss_buffer_.empty()) {
        max_time = std::max(max_time, gnss_buffer_.back().time);
    }
    for (const auto& pair : imu_buffers_) {
        if (!pair.second.empty()) {
            max_time = std::max(max_time, pair.second.back().time);
        }
    }
    return max_time;
}

} // namespace ob_gins
