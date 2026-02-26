import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_and_plot():
    base_dir = "D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/"
    out_dir = "D:/Code/dataset/WID/Datasets/transformedData2/output/speed_analysis_real"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gnss_file = os.path.join(base_dir, "GNSS_use.txt")
    wheel_files = {
        "FL": os.path.join(base_dir, "Center_IMU1.txt"),
        "RL": os.path.join(base_dir, "Center_IMU2.txt"),
        "FR": os.path.join(base_dir, "Center_IMU3.txt"),
        "RR": os.path.join(base_dir, "Center_IMU4.txt")
    }
    
    gnss = np.loadtxt(gnss_file)
    t0 = gnss[0, 0]
    
    mask_gnss = (gnss[:, 0] >= t0 + 50) & (gnss[:, 0] <= t0 + 100)
    gnss_focus = gnss[mask_gnss]
    t_gnss = gnss_focus[:, 0] - t0
    
    # Calculate real GNSS speed from position derivatives
    lat = np.radians(gnss_focus[:, 1])
    lon = np.radians(gnss_focus[:, 2])
    
    # Earth radius ~ 6378137 m
    R_earth = 6378137.0
    
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    dt = np.diff(t_gnss)
    
    # Avoid div by zero
    dt[dt == 0] = 1e-6
    
    vn = dlat / dt * R_earth
    ve = dlon / dt * R_earth * np.cos(lat[:-1])
    gnss_speed = np.sqrt(vn**2 + ve**2)
    
    # Shift time for speed to be middle of interval
    t_gnss_speed = t_gnss[:-1] + dt/2.0
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("Wheel Z-axis Rotation, Wheel Speed and Corrected GNSS Speed (50s - 100s)", fontsize=16)
    
    wheel_radius = 0.3
    colors = {'FL': 'r', 'RL': 'g', 'FR': 'b', 'RR': 'c'}
    
    for name, file_path in wheel_files.items():
        if os.path.exists(file_path):
            imu = np.loadtxt(file_path)
            mask_imu = (imu[:, 0] >= t0 + 50) & (imu[:, 0] <= t0 + 100)
            imu_focus = imu[mask_imu]
            t_imu = imu_focus[:, 0] - t0
            gz = imu_focus[:, 3] 
            
            max_val = np.max(np.abs(gz))
            if max_val < 10.0:
                gz_rate = gz * 120.0
            else:
                gz_rate = gz
                
            wheel_speed = np.abs(gz_rate) * wheel_radius
            
            axes[0].plot(t_imu, gz_rate, label=f"{name} Gyro Z", color=colors[name], alpha=0.7)
            axes[1].plot(t_imu, wheel_speed, label=f"{name} Speed (|w|*R)", color=colors[name], alpha=0.7)

    # Plot smoothed GNSS speed to reduce noise from differentiation
    from scipy.ndimage import gaussian_filter1d
    gnss_speed_smooth = gaussian_filter1d(gnss_speed, sigma=2)
    
    axes[2].plot(t_gnss_speed, gnss_speed, 'k-', alpha=0.3, label="Raw GNSS Speed (diff)")
    axes[2].plot(t_gnss_speed, gnss_speed_smooth, 'r-', linewidth=2, label="Smoothed GNSS Speed")
    
    axes[0].set_ylabel("Z-axis Rotation Rate (rad/s)")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_ylabel("Calculated Wheel Speed (m/s)")
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].set_xlabel("Time (s) from t0")
    axes[2].set_ylabel("GNSS Speed (m/s)")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(out_dir, "wheel_real_gnss_speed_50_100s.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to: {save_path}")

if __name__ == "__main__":
    analyze_and_plot()
