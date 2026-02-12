import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def main():
    parser = argparse.ArgumentParser(description="Plot Wheel Phase Angle from CT results")
    parser.add_argument("--result", type=str, required=True, help="Path to ct_trajectory.txt")
    args = parser.parse_args()

    if not os.path.exists(args.result):
        print(f"Error: Result file {args.result} not found.")
        return

    output_dir = os.path.dirname(args.result)
    
    # Load Trajectory
    print(f"Loading trajectory: {args.result}")
    traj_data = np.loadtxt(args.result)
    if traj_data.ndim == 1: traj_data = traj_data.reshape(1, -1)
    t_traj = traj_data[:, 0]
    rpy_traj = traj_data[:, 7:10] * np.pi / 180.0 # to radians
    
    # Body-to-World Quaternion (q_wb)
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
    
    q_wb_arr = np.array(q_wb_list) # Nx4
    
    # Process each wheel IMU
    error_files = glob.glob(os.path.join(output_dir, "errors_*.txt"))
    for ef in error_files:
        imu_name = os.path.basename(ef).replace("errors_", "").replace(".txt", "")
        # Filter for wheel IMUs only
        if "center_imu" not in imu_name.lower(): continue 
        
        print(f"Processing Phase Angle for: {imu_name}")
        
        # 1. Load Errors File
        err_data = np.loadtxt(ef)
        if err_data.ndim == 1: err_data = err_data.reshape(1, -1)
        if err_data.shape[0] == 0: continue
        
        # Check if we have the optimized phase column
        # Old cols: 1+3+3+3+4+3+1 = 18
        # New cols: 18 + 1 = 19
        use_optimized_phase = False
        if err_data.shape[1] >= 19:
            use_optimized_phase = True
            print(f"  Found optimized phase column in {ef}")
        
        t_eval = []
        phase_deg = []

        if use_optimized_phase:
            # Directly use the saved phase
            # Col index 18 (0-based)
            t_eval = err_data[:, 0]
            phase_rad = err_data[:, 18]
            phase_deg = phase_rad * 180.0 / np.pi
            
            # Since this is sparse (control point rate), we might want to plot it as lines
            # It should be smooth enough if spline knot is 0.1s
        
        else:
            print("  Optimized phase not found, falling back to integration...")
            # Fallback to old logic
            
            # 1. Get Optimized Extrinsics q_bs (Body-to-Sensor)
            # Last available estimate for extrinsics (cols 10-13: qx,qy,qz,qw)
            q_bs_opt = err_data[-1, 10:14] # xyzw
            
            # 2. Compute q_ws_opt = q_wb(t) * q_bs_opt (Body motion projected to sensor frame)
            x2, y2, z2, w2 = q_bs_opt
            
            q_ws_opt_list = []
            t_opt_list = []
            
            for i in range(len(t_traj)):
                x1, y1, z1, w1 = q_wb_arr[i] 
                
                w = w1*w2 - x1*x2 - y1*y2 - z1*z2
                x = w1*x2 + x1*w2 + y1*z2 - z1*y2
                y = w1*y2 - x1*z2 + y1*w2 + z1*x2
                z = w1*z2 + x1*y2 - y1*x2 + z1*w2
                
                q_ws_opt_list.append(np.array([x, y, z, w]))
                t_opt_list.append(t_traj[i])

            # 3. Load Raw Data
            raw_filename = imu_name.replace("center_imu", "Center_IMU") + ".txt"
            # Try to locate raw file
            dataset_dir = os.path.abspath(os.path.join(output_dir, "../")) # .../output/.. -> .../Datasets/transformedData2/output/.. -> .../Datasets/transformedData2/
            
            raw_path = None
            # Try finding in known trial dirs
            trial_dirs = glob.glob(os.path.join(dataset_dir, "four_wheel_dataset_PassengerCar", "trial*"))
            for td in trial_dirs:
                p = os.path.join(td, raw_filename)
                if os.path.exists(p):
                    raw_path = p
                    break
            
            if not raw_path:
                 # Try simple local if in same dir
                 p = os.path.join(dataset_dir, raw_filename)
                 if os.path.exists(p): raw_path = p
            
            if not raw_path:
                print(f"Warning: Raw file for {imu_name} not found.")
                continue
                
            print(f"  Using raw data: {raw_path}")
            raw_data = np.loadtxt(raw_path)
            t_raw = raw_data[:, 0]
            gyro_raw = raw_data[:, 1:4] 
            
            # 4. Integrate Wheel with Bias Correction
            t_err = err_data[:, 0]
            bg_opt = err_data[:, 1:4] # rad/s
            
            bg_interp = np.zeros_like(gyro_raw)
            for k in range(3):
                bg_interp[:, k] = np.interp(t_raw, t_err, bg_opt[:, k])
                
            q_wheel_corr_list = []
            q_curr = np.array([0., 0., 0., 1.]) # Identity
            q_wheel_corr_list.append(q_curr)
            t_wheel_corr = [t_raw[0]]
            
            for i in range(1, len(t_raw)):
                dt = t_raw[i] - t_raw[i-1]
                if dt <= 0: dt = 1e-5
                
                # Corrected increment: dtheta - bias * dt
                dtheta = gyro_raw[i] - bg_interp[i] * dt
                
                angle = np.linalg.norm(dtheta)
                if angle < 1e-8:
                    dq = np.array([0., 0., 0., 1.])
                else:
                    axis = dtheta / angle
                    s = np.sin(angle/2)
                    dq = np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle/2)])
                
                # q_new = q_curr * dq
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
            
            # 5. Compute Relative Rotation (Wheel vs Body)
            key_rots = R.from_quat(q_ws_opt_list)
            slerp_gen = Slerp(t_opt_list, key_rots)
            
            t_min, t_max = t_opt_list[0], t_opt_list[-1]
            mask = (t_wheel_corr >= t_min) & (t_wheel_corr <= t_max)
            t_eval = t_wheel_corr[mask]
            q_wheel_eval = q_wheel_corr[mask]
            q_body_eval = slerp_gen(t_eval).as_quat()
            
            # q_rel = q_body^{-1} * q_wheel
            qb_inv = q_body_eval.copy()
            qb_inv[:, 0:3] = -qb_inv[:, 0:3]
            
            x1, y1, z1, w1 = qb_inv.T
            x2, y2, z2, w2 = q_wheel_eval.T
            
            wr = w1*w2 - x1*x2 - y1*y2 - z1*z2
            xr = w1*x2 + x1*w2 + y1*z2 - z1*y2
            yr = w1*y2 - x1*z2 + y1*w2 + z1*x2
            zr = w1*z2 + x1*y2 - y1*x2 + z1*w2
            
            # Fix Initial Offset
            w0, x0, y0, z0 = wr[0], xr[0], yr[0], zr[0]
            x0, y0, z0 = -x0, -y0, -z0
            
            wf = w0*wr - x0*xr - y0*yr - z0*zr
            xf = w0*xr + x0*wr + y0*zr - z0*yr
            yf = w0*yr - x0*zr + y0*wr + z0*xr
            zf = w0*zr + x0*yr - y0*xr + z0*wr
            
            # Double cover check
            mask_neg = wf < 0
            wf[mask_neg] *= -1
            xf[mask_neg] *= -1
            yf[mask_neg] *= -1
            zf[mask_neg] *= -1
            
            vec_norm = np.sqrt(xf**2 + yf**2 + zf**2)
            angle = 2.0 * np.arctan2(vec_norm, wf)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                scale = angle / vec_norm
                scale[vec_norm < 1e-8] = 0
                
            rv_z = zf * scale
            rv_z_unwrapped = np.unwrap(rv_z)
            phase_deg = rv_z_unwrapped * 180.0 / np.pi

        # Plot Phase
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_eval, phase_deg, 'b-', linewidth=1)
        ax.set_title(f'{imu_name} Continuous Wheel Rotation (deg)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rotation (deg)')
        # ax.set_ylim([0, 360]) # Removed limit
        ax.grid(True)
        
        path = os.path.join(output_dir, f'ct_wheel_phase_{imu_name}.png')
        plt.savefig(path)
        print(f"Saved {path}")
        plt.close(fig)

if __name__ == "__main__":
    main()
