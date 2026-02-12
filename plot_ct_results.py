import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def blh_to_enu(blh, origin_blh):
    """
    将 BLH (纬度, 经度, 高度) 转换为局部 ENU 坐标系。
    blh: Nx3 数组 (度, 度, 米)
    origin_blh: 1x3 数组 (起始点)
    """
    lat = np.radians(blh[:, 0])
    lon = np.radians(blh[:, 1])
    h = blh[:, 2]
    
    lat0 = np.radians(origin_blh[0])
    lon0 = np.radians(origin_blh[1])
    h0 = origin_blh[2]
    
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    
    def ecef(lat, lon, h):
        v = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        x = (v + h) * np.cos(lat) * np.cos(lon)
        y = (v + h) * np.cos(lat) * np.sin(lon)
        z = (v * (1 - e2) + h) * np.sin(lat)
        if np.isscalar(lat) or lat.ndim == 0:
             return np.array([x, y, z])
        return np.stack([x, y, z], axis=1)

    p = ecef(lat, lon, h)
    p0 = ecef(lat0, lon0, h0)
    
    dp = p - p0
    
    sin_lat0, cos_lat0 = np.sin(lat0), np.cos(lat0)
    sin_lon0, cos_lon0 = np.sin(lon0), np.cos(lon0)
    
    t = np.array([
        [-sin_lon0, cos_lon0, 0],
        [-sin_lat0 * cos_lon0, -sin_lat0 * sin_lon0, cos_lat0],
        [cos_lat0 * cos_lon0, cos_lat0 * sin_lon0, sin_lat0]
    ])
    
    return dp @ t.T

def main():
    parser = argparse.ArgumentParser(description="Plot OB-GINS-CT results vs Truth")
    parser.add_argument("--result", type=str, required=True, help="Path to ct_trajectory.txt")
    parser.add_argument("--truth", type=str, required=True, help="Path to GNSS truth file")
    parser.add_argument("--label", type=str, default="OB_GINS_CT")
    args = parser.parse_args()

    if not os.path.exists(args.result):
        print(f"Error: Result file {args.result} not found.")
        return

    output_dir = os.path.dirname(args.result)

    # 1. Load Navigation Results
    # Format: time, lat, lon, alt, vx, vy, vz, roll, pitch, yaw
    print(f"Loading result: {args.result}")
    res_data = np.loadtxt(args.result)
    if res_data.ndim == 1: res_data = res_data.reshape(1, -1)
    res_time = res_data[:, 0]
    res_blh = res_data[:, 1:4]
    res_vel = res_data[:, 4:7]
    res_att = res_data[:, 7:10]

    # 2. Load Truth Data
    # Format: time, lat, lon, alt, ...
    print(f"Loading truth: {args.truth}")
    truth_data = np.loadtxt(args.truth)
    if truth_data.ndim == 1: truth_data = truth_data.reshape(1, -1)
    truth_time = truth_data[:, 0]
    truth_blh = truth_data[:, 1:4]

    # 3. Convert to ENU for comparison
    origin_blh = truth_blh[0]
    truth_enu = blh_to_enu(truth_blh, origin_blh)
    res_enu = blh_to_enu(res_blh, origin_blh)

    # Align trajectories (time-based check)
    t0 = res_time[0]
    t_end = res_time[-1]
    
    # Filter truth to match result time window roughly for easier plotting
    mask = (truth_time >= t0 - 1.0) & (truth_time <= t_end + 1.0)
    truth_time_plot = truth_time[mask]
    truth_enu_plot = truth_enu[mask]
    truth_blh_plot = truth_blh[mask]

    # --- Plot 1: Position Comparison (ENU) ---
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ['East (m)', 'North (m)', 'Up (m)']
    for i in range(3):
        axs1[i].plot(truth_time_plot, truth_enu_plot[:, i], 'k--', label='Truth')
        axs1[i].plot(res_time, res_enu[:, i], 'r-', label=args.label)
        axs1[i].set_ylabel(labels[i])
        axs1[i].grid(True)
        axs1[i].legend()
    axs1[2].set_xlabel('Time (s)')
    fig1.suptitle('Position Comparison (Local ENU)')
    plt.tight_layout()
    
    path1 = os.path.join(output_dir, 'ct_position_comparison.png')
    fig1.savefig(path1)
    print(f"Saved {path1}")
    plt.close(fig1)

    # --- Plot 2: 2D Trajectory ---
    fig2 = plt.figure(figsize=(8, 8))
    plt.plot(truth_enu_plot[:, 0], truth_enu_plot[:, 1], 'k--', label='Truth')
    plt.plot(res_enu[:, 0], res_enu[:, 1], 'r-', label=args.label)
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Horizontal Trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    path2 = os.path.join(output_dir, 'ct_trajectory_2d.png')
    plt.savefig(path2)
    print(f"Saved {path2}")
    plt.close(fig2)

    # --- Plot 3: Position Errors ---
    # Interpolate Truth to Result Time
    truth_east_interp = np.interp(res_time, truth_time, truth_enu[:, 0])
    truth_north_interp = np.interp(res_time, truth_time, truth_enu[:, 1])
    truth_alt_interp = np.interp(res_time, truth_time, truth_blh[:, 2])
    
    res_east = res_enu[:, 0]
    res_north = res_enu[:, 1]
    res_alt = res_blh[:, 2]

    error_east = res_east - truth_east_interp
    error_north = res_north - truth_north_interp
    error_horiz = np.sqrt(error_east**2 + error_north**2)
    error_height = res_alt - truth_alt_interp

    fig3 = plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(res_time, error_horiz, 'r-', label='Horizontal Error')
    plt.ylabel('Error (m)')
    plt.title('Horizontal Position Error')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(res_time, error_east, label='East Error')
    plt.plot(res_time, error_north, label='North Error')
    plt.ylabel('Error (m)')
    plt.title('East/North Errors')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(res_time, error_height, 'b-', label='Height Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Height Error (Nav - Truth)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    path3 = os.path.join(output_dir, 'ct_position_errors.png')
    plt.savefig(path3)
    print(f"Saved {path3}")
    plt.close(fig3)

    # --- Plot 4+: IMU Errors & Attitude ---
    error_files = glob.glob(os.path.join(output_dir, "errors_*.txt"))
    
    for ef in error_files:
        imu_name = os.path.basename(ef).replace("errors_", "").replace(".txt", "")
        print(f"Plotting biases for: {imu_name}")
        
        # Format: t, bg(3), ba(3), l(3), r(4), [wheel_params(4)]
        err_data = np.loadtxt(ef)
        if err_data.ndim == 1: err_data = err_data.reshape(1, -1)
        if err_data.shape[0] == 0: continue

        e_time = err_data[:, 0]
        bg = err_data[:, 1:4]
        ba = err_data[:, 4:7]

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(e_time, bg[:, 0], label='X')
        axs[0].plot(e_time, bg[:, 1], label='Y')
        axs[0].plot(e_time, bg[:, 2], label='Z')
        axs[0].set_title(f'{imu_name} Gyro Bias')
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(e_time, ba[:, 0], label='X')
        axs[1].plot(e_time, ba[:, 1], label='Y')
        axs[1].plot(e_time, ba[:, 2], label='Z')
        axs[1].set_title(f'{imu_name} Accel Bias')
        axs[1].legend()
        axs[1].grid(True)
        
        axs[1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        path_imu = os.path.join(output_dir, f'ct_bias_{imu_name}.png')
        plt.savefig(path_imu)
        print(f"Saved {path_imu}")
        plt.close(fig)

            # Check for Attitude file
        att_file = os.path.join(output_dir, f"attitude_{imu_name}.txt")
        if os.path.exists(att_file):
            print(f"Plotting differential attitude vs Raw for: {imu_name}")
            att_data = np.loadtxt(att_file)
            if att_data.ndim == 1: att_data = att_data.reshape(1, -1)
            
            # t, qx, qy, qz, qw
            t_att = att_data[:, 0]
            qs = att_data[:, 1:5] # xyzw

            # 1. Calculate Differential (Step-wise) Rotation Vector
            # q_step[k] = q[k-1]^{-1} * q[k]
            # This represents the rotation accumulated over the interval (dt)
            
            diff_rv_list = []
            diff_t = []
            
            for i in range(1, len(qs)):
                q_prev = qs[i-1]
                q_curr = qs[i]
                dt = t_att[i] - t_att[i-1]
                if dt <= 0: continue

                # q_step = q_prev^{-1} * q_curr
                x1, y1, z1, w1 = -q_prev[0], -q_prev[1], -q_prev[2], q_prev[3] # Inverse
                x2, y2, z2, w2 = q_curr
                
                w_rel = w1*w2 - x1*x2 - y1*y2 - z1*z2
                x_rel = w1*x2 + x1*w2 + y1*z2 - z1*y2
                y_rel = w1*y2 - x1*z2 + y1*w2 + z1*x2
                z_rel = w1*z2 + x1*y2 - y1*x2 + z1*w2
                
                vec = np.array([x_rel, y_rel, z_rel])
                norm_vec = np.linalg.norm(vec)
                
                if norm_vec < 1e-8:
                    rv = np.zeros(3)
                else:
                    angle = 2.0 * np.arctan2(norm_vec, w_rel)
                    rv = (angle / norm_vec) * vec
                
                # Convert to deg/s (Angular Velocity equivalent) for easier comparison
                # rv is rotation in rad over dt.
                # omega = rv / dt
                
                diff_rv_list.append(rv * 180.0 / np.pi / dt) 
                diff_t.append(t_att[i])

            diff_rv = np.array(diff_rv_list)
            diff_t = np.array(diff_t)

            # 2. Load Raw IMU Data for Comparison
            # Assumption: Raw file is in the parent dataset directory
            # Naming convention: center_imu1 -> Center_IMU1.txt
            
            raw_filename = imu_name.replace("center_imu", "Center_IMU").replace("imu_main", "Body_IMU") + ".txt"
            # Try to locate it relative to output path
            # Output: .../output/
            # Data: .../trial01/
            
            dataset_dir = os.path.abspath(os.path.join(output_dir, "../"))
            raw_path = os.path.join(dataset_dir, raw_filename)
            
            if not os.path.exists(raw_path):
                # Try finding it recursively or assume fixed path from config?
                # Fallback to hardcoded known path for this session if needed
                fallback_path = f"/home/xunyi/code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/{raw_filename}"
                if os.path.exists(fallback_path):
                    raw_path = fallback_path
            
            if os.path.exists(raw_path):
                print(f"Loading raw data: {raw_path}")
                raw_data = np.loadtxt(raw_path)
                # Cols: Time, Gx, Gy, Gz, Ax, Ay, Az (based on head output)
                # Check column count. If 9 cols? 
                # Head: 0.000000000 0.00e+00 ...
                # Let's assume standard index 1,2,3 for Gyro
                
                t_raw = raw_data[:, 0]
                gyro_raw = raw_data[:, 1:4] 
                # Raw units? 
                # Config says "gyro_noise: deg/s/sqrt(Hz)", usually input is rad/s or deg/s?
                # Typical WID dataset is rad/s.
                # Let's assume rad/s and convert to deg/s
                
                gyro_raw_deg = gyro_raw * 180.0 / np.pi
                
                # Filter raw to plot range
                mask = (t_raw >= t_att[0]) & (t_raw <= t_att[-1])
                t_raw = t_raw[mask]
                gyro_raw_deg = gyro_raw_deg[mask]
                
                fig_comp, axs_comp = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
                
                axis_names = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']
                
                for i in range(3):
                    # Left Axis: Raw Data (Blue)
                    color_raw = 'tab:blue'
                    axs_comp[i].set_ylabel('Raw Gyro (deg/s)', color=color_raw)
                    line1, = axs_comp[i].plot(t_raw, gyro_raw_deg[:, i], color=color_raw, label='Raw Gyro', linewidth=1.5, alpha=0.7)
                    axs_comp[i].tick_params(axis='y', labelcolor=color_raw)
                    
                    # Right Axis: Integrated Step (Red/Orange)
                    ax2 = axs_comp[i].twinx()
                    color_int = 'tab:red'
                    ax2.set_ylabel('Integrated Step (deg/s)', color=color_int)
                    line2, = ax2.plot(diff_t, diff_rv[:, i], color=color_int, label='Integrated', linewidth=1.5, linestyle='--', alpha=0.8)
                    ax2.tick_params(axis='y', labelcolor=color_int)
                    
                    axs_comp[i].set_title(f'{imu_name} Angular Velocity {axis_names[i]}')
                    
                    # Combine legends
                    lines = [line1, line2]
                    labels = [l.get_label() for l in lines]
                    axs_comp[i].legend(lines, labels, loc='upper right')
                    
                    axs_comp[i].grid(True)
                
                axs_comp[2].set_xlabel('Time (s)')
                plt.tight_layout()
                
                path_comp = os.path.join(output_dir, f'ct_step_vs_raw_{imu_name}.png')
                plt.savefig(path_comp)
                print(f"Saved {path_comp}")
                plt.close(fig_comp)
            else:
                print(f"Warning: Raw file {raw_path} not found.")


    # --- Plot 5: Optimized Attitude vs Integrated Attitude ---
    # Load optimized trajectory (ct_trajectory.txt)
    # Format: Time, Lat, Lon, Alt, Vn, Ve, Vd, Roll, Pitch, Yaw, ...
    # Optimized Attitude (Body-to-World) q_wb
    
    # We need to compute q_sw_opt = q_wb_opt * q_bs_opt
    # q_bs_opt is not directly in trajectory file, but we can assume identity for now 
    # OR better: read errors_*.txt to get optimized q_body_imu (q_sb)
    # q_sw = q_bw^{-1} * q_bs^{-1} ? No.
    # q_sw = R_sw.
    # R_sw = R_sb * R_bw ? No.
    # Standard chain: R_ws = R_wb * R_bs. 
    # So R_sw = R_bs^T * R_wb^T. (Sensor to World? No, usually World to Sensor)
    # Let's stick to quaternion notation.
    # q_sw (Sensor to World) = q_bw (Body to World) * q_sb (Sensor to Body)
    # q_sw = q_wb * q_bs_inv.
    
    # Wait, in code: q_diff = q_ext_opt.inverse() * q_sb_meas;
    # q_ext is Body-to-Sensor (q_bs).
    # q_sw_meas is Sensor-to-World.
    # q_bw = q_sw_meas * q_bs^{-1}.
    # So q_sw_opt = q_wb_opt * q_bs_opt.
    
    # Let's align optimized attitude to sensor frame for comparison.
    
    # Load Trajectory
    traj_data = np.loadtxt(args.result)
    if traj_data.ndim == 1: traj_data = traj_data.reshape(1, -1)
    t_traj = traj_data[:, 0]
    rpy_traj = traj_data[:, 7:10] * np.pi / 180.0 # to radians
    
    # Convert Traj RPY to Quat (q_wb)
    # ZYX order
    q_wb_list = []
    for r, p, y in rpy_traj:
        # q = from_euler(r, p, y)
        sr, cr = np.sin(r/2), np.cos(r/2)
        sp, cp = np.sin(p/2), np.cos(p/2)
        sy, cy = np.sin(y/2), np.cos(y/2)
        
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        q_wb_list.append(np.array([x, y, z, w]))
    
    q_wb_arr = np.array(q_wb_list)

    error_files = glob.glob(os.path.join(output_dir, "errors_*.txt"))
    for ef in error_files:
        imu_name = os.path.basename(ef).replace("errors_", "").replace(".txt", "")
        # Filter for wheel IMUs only
        if "center_imu" not in imu_name.lower(): continue 
        
        print(f"Comparing Optimized vs Integrated for: {imu_name}")
        
        # 1. Get Optimized Extrinsics q_bs (Body-to-Sensor) from errors file (last column usually static/converged)
        # Format: ..., q_body_imu(4), ...
        err_data = np.loadtxt(ef)
        if err_data.ndim == 1: err_data = err_data.reshape(1, -1)
        if err_data.shape[0] == 0: continue
        
        # Take the last extrinsic estimate
        # Columns: t(1) + bg(3) + ba(3) + l(3) + q(4) ...
        # Index: 1+3+3+3 = 10. So cols 10,11,12,13 are q_bs
        q_bs_opt = err_data[-1, 10:14] # xyzw
        
        # 2. Compute q_sw_opt = q_wb(t) * q_bs_opt
        # q_mult(q1, q2):
        # w = w1w2 - x1x2 - y1y2 - z1z2
        # x = w1x2 + x1w2 + y1z2 - z1y2
        # y = w1y2 - x1z2 + y1w2 + z1x2
        # z = w1z2 + x1y2 - y1x2 + z1w2
        
        # q_wb is Body-to-World (or World-to-Body? Usually Nav output is Body-to-World/Nav)
        # Code says: q_sw = q_bw * q_sb. Wait.
        # Let's assume standard: T_wb means Frame Body expressed in World.
        # q_wb rotates vector from Body to World.
        # q_bs rotates vector from Sensor to Body.
        # v_w = q_wb * (q_bs * v_s).
        # So q_ws = q_wb * q_bs.
        
        x2, y2, z2, w2 = q_bs_opt
        
        q_ws_opt_list = []
        t_opt_list = []
        
        # Interpolate or match timestamps?
        # Trajectory is 0.1s usually. Errors is 0.1s.
        # Integrated attitude is High Rate.
        # Let's plot at trajectory rate.
        
        for i in range(len(t_traj)):
            x1, y1, z1, w1 = q_wb_arr[i] # q_wb
            
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            
            q_ws_opt_list.append(np.array([x, y, z, w]))
            t_opt_list.append(t_traj[i])
            
        # 3. Load Integrated Attitude (Raw Integration)
        att_file = os.path.join(output_dir, f"attitude_{imu_name}.txt")
        if not os.path.exists(att_file): continue
        att_data = np.loadtxt(att_file)
        if att_data.ndim == 1: att_data = att_data.reshape(1, -1)
        t_int = att_data[:, 0]
        q_int = att_data[:, 1:5] # xyzw (Sensor-to-World integrated)
        
        # 4. Use Cumulative Differential Integration to avoid Wrapping/Sign issues
        # Theta(t) = sum( RotVec(q_{k-1}^{-1} * q_k) )
        
        def compute_cumulative_rotvec(times, quats):
            cum_rv = np.zeros((len(times), 3))
            current_rv = np.zeros(3)
            
            for i in range(1, len(times)):
                # q_step = q_{k-1}^{-1} * q_k
                q_prev = quats[i-1]
                q_curr = quats[i]
                
                # Inverse of prev
                x1, y1, z1, w1 = -q_prev[0], -q_prev[1], -q_prev[2], q_prev[3]
                x2, y2, z2, w2 = q_curr
                
                # Multiply
                wr = w1*w2 - x1*x2 - y1*y2 - z1*z2
                xr = w1*x2 + x1*w2 + y1*z2 - z1*y2
                yr = w1*y2 - x1*z2 + y1*w2 + z1*x2
                zr = w1*z2 + x1*y2 - y1*x2 + z1*w2
                
                # Robust Angle-Axis
                # Force positive w (shortest path for step) - Valid for small steps
                if wr < 0:
                    wr, xr, yr, zr = -wr, -xr, -yr, -zr
                
                vec = np.array([xr, yr, zr])
                n = np.linalg.norm(vec)
                if n < 1e-8:
                    step_rv = np.zeros(3)
                else:
                    angle = 2.0 * np.arctan2(n, wr)
                    step_rv = (angle / n) * vec
                
                current_rv += step_rv
                cum_rv[i] = current_rv
                
            return cum_rv * 180.0 / np.pi

        rv_opt = compute_cumulative_rotvec(t_opt_list, q_ws_opt_list)
        rv_int = compute_cumulative_rotvec(t_int, q_int)
        
        # Plot
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        cols = ['r', 'g', 'b']
        lbls = ['Cumul X (Roll)', 'Cumul Y (Pitch)', 'Cumul Z (Yaw/Rotation)']
        
        for i in range(3):
            axs[i].plot(t_opt_list, rv_opt[:, i], 'k-', linewidth=2, label='Optimized')
            axs[i].plot(t_int, rv_int[:, i], color=cols[i], linestyle='--', linewidth=1.5, label='Raw Integration')
            axs[i].set_title(f'{imu_name} {lbls[i]} (deg)')
            axs[i].grid(True)
            axs[i].legend()
            
        axs[2].set_xlabel('Time (s)')
        fig.suptitle(f'{imu_name}: Cumulative Rotation Vector (Optimized vs Raw)')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'ct_opt_vs_raw_{imu_name}.png')
        plt.savefig(path)
        print(f"Saved {path}")
        plt.close(fig)

        # 5. Compute Relative Attitude (Wheel relative to Body)
        # q_rel = q_opt^{-1} * q_int
        # This represents the rotation of the wheel relative to the body (Sensor Frame)
        
        q_diff_list = []
        t_diff = []
        
        # Match timestamps roughly (interp integrated to optimized time)
        # Or just loop optimized and find closest integrated
        
        from scipy.spatial.transform import Rotation as R
        
        # Convert all to scipy for easier interp and math
        # Opt (Body)
        r_opt = R.from_quat(q_ws_opt_list) # xyzw
        
        # Int (Wheel)
        # Need to interp q_int to t_opt_list
        # SLERP is ideal, but for plotting, Nearest Neighbor or Linear is okay if dense enough
        
        # Let's do simple nearest neighbor for now as rates are high
        
        q_int_arr = np.array(q_int)
        
        diff_rv_list = []
        
        for i, t in enumerate(t_opt_list):
            # Find closest in t_int
            idx = np.searchsorted(t_int, t)
            if idx == 0: idx = 0
            elif idx >= len(t_int): idx = len(t_int) - 1
            else:
                if abs(t - t_int[idx-1]) < abs(t - t_int[idx]):
                    idx = idx - 1
            
            if abs(t - t_int[idx]) > 0.05: continue # Too far
            
            q_b = q_ws_opt_list[i] # Body-in-World (projected to sensor frame)
            q_w = q_int_arr[idx]   # Wheel-in-World
            
            # q_rel = q_b^{-1} * q_w  (Wheel w.r.t Body)
            x1, y1, z1, w1 = -q_b[0], -q_b[1], -q_b[2], q_b[3]
            x2, y2, z2, w2 = q_w
            
            wr = w1*w2 - x1*x2 - y1*y2 - z1*z2
            xr = w1*x2 + x1*w2 + y1*z2 - z1*y2
            yr = w1*y2 - x1*z2 + y1*w2 + z1*x2
            zr = w1*z2 + x1*y2 - y1*x2 + z1*w2
            
            # Rotation Vector
            vec = np.array([xr, yr, zr])
            n = np.linalg.norm(vec)
            if n < 1e-8:
                rv = np.zeros(3)
            else:
                angle = 2.0 * np.arctan2(n, wr)
                rv = (angle / n) * vec
            
            diff_rv_list.append(rv)
            t_diff.append(t)
            
        diff_rv = np.array(diff_rv_list)
        
        # Unwrap Z (The actual wheel rotation angle)
        diff_rv[:, 2] = np.unwrap(diff_rv[:, 2]) 
        
        diff_deg = diff_rv * 180.0 / np.pi
        
        # Plot Difference
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        axs[0].plot(t_diff, diff_deg[:, 0], 'r')
        axs[0].set_title(f'{imu_name} Relative Roll (X) [Ideally 0]')
        axs[0].set_ylabel('deg')
        axs[0].grid(True)
        # Add limit lines if needed
        
        axs[1].plot(t_diff, diff_deg[:, 1], 'g')
        axs[1].set_title(f'{imu_name} Relative Pitch (Y) [Ideally 0]')
        axs[1].set_ylabel('deg')
        axs[1].grid(True)
        
        axs[2].plot(t_diff, diff_deg[:, 2], 'b')
        axs[2].set_title(f'{imu_name} Relative Yaw (Z) [Wheel Rotation]')
        axs[2].set_ylabel('deg')
        axs[2].grid(True)
        
        axs[2].set_xlabel('Time (s)')
        fig.suptitle(f'{imu_name}: Wheel Relative to Body (The Constraint Check)')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'ct_rel_body_wheel_{imu_name}.png')
        plt.savefig(path)
        print(f"Saved {path}")
        plt.close(fig)

        # 6. Re-Integrate Wheel Attitude using Optimized Bias
        # q_wheel_corr(t) = Integrate(omega_raw - bg_opt)
        
        print(f"Re-integrating {imu_name} with optimized bias...")
        
        # Load Raw Data again
        if not os.path.exists(raw_path): continue
        raw_data = np.loadtxt(raw_path)
        t_raw = raw_data[:, 0]
        gyro_raw = raw_data[:, 1:4] # rad/s
        
        # Load Optimized Bias
        # t, bg(3)...
        # Interpolate bg to t_raw
        t_err = err_data[:, 0]
        bg_opt = err_data[:, 1:4]
        
        bg_interp = np.zeros_like(gyro_raw)
        for k in range(3):
            bg_interp[:, k] = np.interp(t_raw, t_err, bg_opt[:, k])
            
        # Corrected Gyro
        # IMPORTANT: Raw data is Incremental (dtheta = omega * dt), not rate (rad/s)
        # But we need rate for interpolation?
        # No, integration logic:
        # q_{k+1} = q_k * exp( dtheta - b*dt )
        
        # So we should treat raw_data[:, 1:4] as dtheta (rad), not omega (rad/s).
        
        # But wait, we need to subtract bias * dt.
        # bg_interp is bias RATE (rad/s).
        
        # So: dtheta_corr = dtheta_raw - bg_interp * dt_raw
        
        dtheta_raw = gyro_raw # Since user said it is accumulated (multiplied by time)
        
        # We need dt for each step to apply bias
        dt_arr = np.diff(t_raw, prepend=t_raw[0]) # Approximation
        # Better: t_raw[i] - t_raw[i-1] inside loop
        
        # Integrate
        q_wheel_corr_list = []
        q_curr = np.array([0., 0., 0., 1.]) # Identity
        
        q_wheel_corr_list.append(q_curr)
        t_wheel_corr = [t_raw[0]]
        
        for i in range(1, len(t_raw)):
            dt = t_raw[i] - t_raw[i-1]
            if dt <= 0: dt = 1e-5 # prevent div zero or logic error
            
            dtheta = dtheta_raw[i] # This is already angle increment
            bias = bg_interp[i] # This is rate (rad/s)
            
            # Corrected increment
            dtheta_corr_val = dtheta - bias * dt
            
            # Simple integration
            angle = np.linalg.norm(dtheta_corr_val)
            if angle < 1e-8:
                dq = np.array([0., 0., 0., 1.])
            else:
                axis = dtheta_corr_val / angle
                s = np.sin(angle/2)
                dq = np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle/2)])
            
            # q_new = q_curr * dq
            # Hamilton product
            x1, y1, z1, w1 = q_curr
            x2, y2, z2, w2 = dq
            
            w_new = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x_new = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y_new = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z_new = w1*z2 + x1*y2 - y1*x2 + z1*w2
            
            q_curr = np.array([x_new, y_new, z_new, w_new])
            q_curr /= np.linalg.norm(q_curr)
            
            q_wheel_corr_list.append(q_curr)
            t_wheel_corr.append(t_raw[i])
            
        q_wheel_corr = np.array(q_wheel_corr_list)
        t_wheel_corr = np.array(t_wheel_corr)
        
        # 7. Compute Relative: Optimized Body vs Corrected Wheel
        # Interpolate Body to Wheel time (Body is 10Hz, Wheel is 100Hz+)
        # We need q_body_opt at t_wheel_corr
        
        # Slerp q_ws_opt_list
        from scipy.spatial.transform import Rotation as R
        
        # Scipy Slerp
        from scipy.spatial.transform import Slerp
        # t_opt_list might have duplicates or jitter? assuming sorted unique
        key_rots = R.from_quat(q_ws_opt_list)
        slerp_gen = Slerp(t_opt_list, key_rots)
        
        # Filter t_wheel_corr to be within t_opt_list range
        t_min = t_opt_list[0]
        t_max = t_opt_list[-1]
        mask = (t_wheel_corr >= t_min) & (t_wheel_corr <= t_max)
        
        t_eval = t_wheel_corr[mask]
        q_wheel_eval = q_wheel_corr[mask]
        
        q_body_eval = slerp_gen(t_eval).as_quat()
        
        # Compute Relative
        rel_rv_list = []
        for i in range(len(t_eval)):
            # q_rel = q_body^{-1} * q_wheel
            qb = q_body_eval[i]
            qw = q_wheel_eval[i]
            
            # initial offset removal?
            # q_rel_raw = qb_inv * qw
            # We want to remove the initial q_rel_0 so start is 0
            pass 
        
        # Vectorized q_rel
        # q_inv = [-x, -y, -z, w]
        qb_inv = q_body_eval.copy()
        qb_inv[:, 0:3] = -qb_inv[:, 0:3]
        
        # Multiply qb_inv * qw
        # Scipy doesn't support array * array easily without converting to objects
        # Manual is faster for arrays
        
        x1, y1, z1, w1 = qb_inv.T
        x2, y2, z2, w2 = q_wheel_eval.T
        
        wr = w1*w2 - x1*x2 - y1*y2 - z1*z2
        xr = w1*x2 + x1*w2 + y1*z2 - z1*y2
        yr = w1*y2 - x1*z2 + y1*w2 + z1*x2
        zr = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        # Now we have q_rel(t). 
        # But q_rel(0) is arbitrary due to integration constant.
        # We want Delta q_rel(t) = q_rel(0)^{-1} * q_rel(t)
        
        w0, x0, y0, z0 = wr[0], xr[0], yr[0], zr[0]
        # inv
        x0, y0, z0 = -x0, -y0, -z0
        
        # Final rotation q_fin = q_rel_0_inv * q_rel(t)
        wf = w0*wr - x0*xr - y0*yr - z0*zr
        xf = w0*xr + x0*wr + y0*zr - z0*yr
        yf = w0*yr - x0*zr + y0*wr + z0*xr
        zf = w0*zr + x0*yr - y0*xr + z0*wr
        
        # Convert to RV
        # Double cover check
        mask_neg = wf < 0
        wf[mask_neg] *= -1
        xf[mask_neg] *= -1
        yf[mask_neg] *= -1
        zf[mask_neg] *= -1
        
        vec_norm = np.sqrt(xf**2 + yf**2 + zf**2)
        angle = 2.0 * np.arctan2(vec_norm, wf)
        
        # Avoid div by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = angle / vec_norm
            scale[vec_norm < 1e-8] = 0
        
        rv_x = xf * scale
        rv_y = yf * scale
        rv_z = zf * scale
        
        # Unwrap Z
        rv_z = np.unwrap(rv_z)
        
        # Plot
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axs[0].plot(t_eval, rv_x * 180/np.pi, 'r')
        axs[0].set_title(f'{imu_name} Corrected Rel Roll (X) [deg]')
        axs[0].set_ylabel('Angle (deg)')
        axs[0].grid(True)
        
        axs[1].plot(t_eval, rv_y * 180/np.pi, 'g')
        axs[1].set_title(f'{imu_name} Corrected Rel Pitch (Y) [deg]')
        axs[1].set_ylabel('Angle (deg)')
        axs[1].grid(True)
        
        axs[2].plot(t_eval, rv_z * 180/np.pi, 'b')
        axs[2].set_title(f'{imu_name} Corrected Rel Yaw (Z) [deg] (Wheel Spin)')
        axs[2].set_ylabel('Angle (deg)')
        axs[2].grid(True)
        
        axs[2].set_xlabel('Time (s)')
        fig.suptitle(f'{imu_name}: Corrected Wheel vs Optimized Body')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'ct_corr_rel_body_wheel_{imu_name}.png')
        plt.savefig(path)
        print(f"Saved {path}")
        plt.close(fig)

    # --- Plot 6: Optimized Body Attitude (B-Spline Output) ---
    # Plot the RPY of the body trajectory itself
    # Check if it makes sense (e.g. should be smooth, consistent with turn)
    
    fig_body, axs_body = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # rpy_traj is in radians, convert to deg
    rpy_body_deg = rpy_traj * 180.0 / np.pi
    
    axs_body[0].plot(t_traj, rpy_body_deg[:, 0], 'k-')
    axs_body[0].set_title('Body Roll (deg)')
    axs_body[0].grid(True)
    
    axs_body[1].plot(t_traj, rpy_body_deg[:, 1], 'k-')
    axs_body[1].set_title('Body Pitch (deg)')
    axs_body[1].grid(True)
    
    axs_body[2].plot(t_traj, rpy_body_deg[:, 2], 'k-')
    axs_body[2].set_title('Body Yaw (deg)')
    axs_body[2].grid(True)
    
    axs_body[2].set_xlabel('Time (s)')
    fig_body.suptitle('Optimized Body Attitude (B-Spline)')
    plt.tight_layout()
    
    path_body = os.path.join(output_dir, 'ct_body_attitude.png')
    plt.savefig(path_body)
    print(f"Saved {path_body}")
    plt.close(fig_body)

if __name__ == "__main__":
    main()