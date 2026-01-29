import numpy as np
import yaml
import glob
import os
import argparse

def load_gnss_speed(file_path):
    # GNSS format: time, lat, lon, alt, ... (Need velocity if available, otherwise compute from pos)
    # Assuming standard format or simple txt. Let's compute speed from pos for robustness.
    # Actually, the user's GNSS file likely has velocity columns?
    # Let's try to infer from data width or compute from d_pos/d_t.
    
    data = np.loadtxt(file_path)
    
    # Simple differencing for speed
    # t, lat, lon, alt
    # Convert lat/lon to meters approx
    R = 6378137.0
    
    t = data[:, 0]
    lat = data[:, 1]
    lon = data[:, 2]
    
    dt = np.diff(t)
    dlat = np.diff(lat) * (np.pi/180)
    dlon = np.diff(lon) * (np.pi/180)
    
    dn = dlat * R
    de = dlon * R * np.cos(np.radians(lat[:-1]))
    
    v = np.sqrt(dn**2 + de**2) / dt
    
    # Pad last element
    v = np.append(v, v[-1])
    
    return t, v

def load_imu_gyro(file_path, side, rate_hz=None):
    # rate_hz is ignored, we compute from timestamps for accuracy
    
    data = np.loadtxt(file_path)
    t = data[:, 0]
    
    # Calculate real dt
    dt = np.diff(t)
    dt = np.append(dt, dt[-1]) # Pad last
    
    # Avoid division by zero
    dt[dt < 1e-6] = 1e-6 
    
    real_freq = 1.0 / np.mean(dt)

    w_norm = np.linalg.norm(data[:, 1:4], axis=1) # Norm of gx, gy, gz
    w_z_raw = data[:, 3]
    
    # Convert Delta Theta (rad) to Omega (rad/s) using real dt
    w_z = w_z_raw / dt
    w_norm = w_norm / dt
    
    if side == 'right':
        w_z = -w_z
        
    return t, np.abs(w_z), w_norm, real_freq # Use abs for speed calc

def estimate_radius(t_gnss, v_gnss, t_imu, w_imu):
    # Interpolate GNSS speed to IMU time
    v_interp = np.interp(t_imu, t_gnss, v_gnss)
    
    # Filter valid data:
    # 1. Speed > 0.5 m/s (avoid static noise)
    # 2. Rotation > 0.5 rad/s
    valid = (v_interp > 0.5) & (w_imu > 0.5)
    
    v_valid = v_interp[valid]
    w_valid = w_imu[valid]
    
    if len(v_valid) < 100:
        return None
    
    # Linear regression: v = R * w  =>  R = v / w
    # Least squares: R = sum(v*w) / sum(w*w)
    
    R = np.sum(v_valid * w_valid) / np.sum(w_valid**2)
    return R

def main():
    config_path = "CT_FGO/config/ob_gins_ct_wsl.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    gnss_file = config['gnssfile']
    print(f"Loading GNSS: {gnss_file}")
    t_gnss, v_gnss = load_gnss_speed(gnss_file)
    
    print(f"GNSS Speed Max: {np.max(v_gnss):.4f} m/s")
    
    print(f"{'Sensor':<15} | {'Side':<6} | {'Freq [Hz]':<10} | {'Est. Radius [m]':<15} | {'Samples'}")
    print("-" * 75)
    
    for key, val in config.items():
        if isinstance(val, dict) and val.get('type') == 'wheel':
            imu_file = val['file']
            side = val.get('side', 'left')
            rate = val.get('rate_hz', 100)
            
            try:
                t_imu, w_z, w_norm, freq = load_imu_gyro(imu_file, side, rate)
                
                # Estimate with w_z
                R_z = estimate_radius(t_gnss, v_gnss, t_imu, w_z)
                # Estimate with w_norm
                R_norm = estimate_radius(t_gnss, v_gnss, t_imu, w_norm)
                
                print(f"{key:<12} | {side:<6} | {freq:<6.1f} | {R_z:.4f} (Z) | {R_norm:.4f} (Norm)")
            except Exception as e:
                print(f"{key:<15} | Error: {e}")

if __name__ == "__main__":
    main()
