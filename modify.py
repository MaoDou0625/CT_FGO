import os

file_path = 'src/ob_gins_ct.cc'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = '    // 4. Build Optimization Problem'
end_marker = '    // 6. Save Results'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    old_code = content[start_idx:end_idx]
    
    new_code = """    // 4. Build Optimization Problem (Sliding Window MVP)
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
    
    while (current_window_start < t_end_global) {
        double current_window_end = std::min(current_window_start + window_size, t_end_global);
        
        LOG(INFO) << "Optimizing Window: [" << std::fixed << current_window_start << ", " << current_window_end << "]";
        
        ceres::Problem problem;
        
        // Setup Manifolds and freeze historical states
        for (auto& cp : control_points) {
            problem.AddParameterBlock(cp.pose_data(), 7);
            problem.SetManifold(cp.pose_data(), new SophusSE3Manifold());
            
            // Task 2.3: Freeze historical states (Marginalization)
            if (cp.timestamp() < current_window_start - 3.0 * spline_dt) {
                problem.SetParameterBlockConstant(cp.pose_data());
                cp.set_state(ControlPoint::State::MARGINALIZED);
            } else if (cp.timestamp() <= current_window_end + 3.0 * spline_dt) {
                cp.set_state(ControlPoint::State::ACTIVE);
            }
        }

        Eigen::Vector3d gnss_lever_arm = Eigen::Vector3d::Zero();
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
        
        current_window_start += step_size;
    }

"""
    
    content = content[:start_idx] + new_code + content[end_idx:]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('ob_gins_ct replaced')
else:
    print('Markers not found')
