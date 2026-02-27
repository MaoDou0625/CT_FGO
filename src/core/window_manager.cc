#include "src/core/window_manager.h"
#include <algorithm>

namespace ob_gins {

bool WindowManager::TryGetNextWindow(const DataBuffer& buffer, Window& out_window) {
    if (!initialized_) {
        double earliest = buffer.GetEarliestTime();
        double latest = buffer.GetLatestTime();
        
        // 确保缓冲区至少有足够的数据覆盖第一个窗口和两侧边缘
        if (latest - earliest >= window_size_ + buffer_margin_ * 2.0) {
            current_window_start_ = earliest + buffer_margin_; // 留出左侧 margin
            initialized_ = true;
        } else {
            return false;
        }
    }

    double expected_end = current_window_start_ + window_size_;
    
    // 我们需要确保缓冲区中最新的数据至少能覆盖到 expected_end + 额外右侧 margin (供样条评估)
    if (buffer.GetLatestTime() >= expected_end + buffer_margin_) {
        out_window.t_start = current_window_start_;
        out_window.t_end = expected_end;
        return true;
    }

    return false;
}

void WindowManager::AdvanceWindow(DataBuffer& buffer) {
    if (!initialized_) return;
    
    current_window_start_ += step_size_;
    
    // 清理缓冲区数据，只保留当前窗口所需的数据（考虑左侧 margin）
    double remove_before = current_window_start_ - buffer_margin_;
    buffer.PopOldData(remove_before);
}

} // namespace ob_gins