#ifndef OB_GINS_CORE_WINDOW_MANAGER_H
#define OB_GINS_CORE_WINDOW_MANAGER_H

#include "src/core/data_buffer.h"

namespace ob_gins {

class WindowManager {
public:
    struct Window {
        double t_start;
        double t_end;
    };

    WindowManager(double window_size, double step_size, double buffer_margin)
        : window_size_(window_size), step_size_(step_size), buffer_margin_(buffer_margin) {}

    // 检查缓冲区是否有足够的数据来构成下一个滑动窗口
    bool TryGetNextWindow(const DataBuffer& buffer, Window& out_window);

    // 推进滑动窗口，并从缓冲区中清理过期数据
    void AdvanceWindow(DataBuffer& buffer);

    // 获取当前窗口状态
    bool IsInitialized() const { return initialized_; }
    double GetCurrentWindowStart() const { return current_window_start_; }

private:
    double window_size_;
    double step_size_;
    double buffer_margin_; // 样条曲线所需的额外边缘数据（如前后各需要 1.0 秒）
    
    bool initialized_ = false;
    double current_window_start_ = 0.0;
};

} // namespace ob_gins

#endif // OB_GINS_CORE_WINDOW_MANAGER_H