import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_and_plot():
    # Paths
    base_dir = "D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/"
    out_dir = "D:/Code/dataset/WID/Datasets/transformedData2/output/speed_analysis_60_80"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gnss_file = os.path.join(base_dir, "GNSS_use.txt")
    wheel_files = {
        "FL": os.path.join(base_dir, "Center_IMU1.txt"),
        "RL": os.path.join(base_dir, "Center_IMU2.txt"),
        "FR": os.path.join(base_dir, "Center_IMU3.txt"),
        "RR": os.path.join(base_dir, "Center_IMU4.txt")
    }
    
    # Load GNSS truth to get t0 and compute GNSS speed
    gnss = np.loadtxt(gnss_file)
    t0 = gnss[0, 0]
    
    mask_gnss = (gnss[:, 0] >= t0 + 60) & (gnss[:, 0] <= t0 + 80)
    gnss_focus = gnss[mask_gnss]
    
    t_gnss = gnss_focus[:, 0] - t0
    vn = gnss_focus[:, 4]
    ve = gnss_focus[:, 5]
    gnss_speed = np.sqrt(vn**2 + ve**2)
    
    # Plotting setup
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("Wheel Z-axis Rotation, Wheel Speed and GNSS Speed (60s - 80s)", fontsize=16)
    
    wheel_radius = 0.3 # from yaml
    
    # Ax 0: Wheel Z-axis Rotation (rad/s)
    # Ax 1: Calculated Wheel Speed (m/s)
    # Ax 2: GNSS Speed (m/s)
    
    colors = {'FL': 'r', 'RL': 'g', 'FR': 'b', 'RR': 'c'}
    
    for name, file_path in wheel_files.items():
        if os.path.exists(file_path):
            imu = np.loadtxt(file_path)
            mask_imu = (imu[:, 0] >= t0 + 60) & (imu[:, 0] <= t0 + 80)
            imu_focus = imu[mask_imu]
            
            t_imu = imu_focus[:, 0] - t0
            # Data format: time, gx, gy, gz, ax, ay, az...
            # Note: in some formats, if it is incremental, it might be scaled. 
            # Assuming gz is in column 3 (rad/s or incremental rad)
            gz = imu_focus[:, 3] 
            
            # Since IMU frequency is 120Hz, if they are increments (rad), rate = gz * 120
            # Let's check max value to decide if it's rate or increment.
            max_val = np.max(np.abs(gz))
            if max_val < 0.5: 
                # highly likely to be increments if wheel is rotating fast, 
                # at 100km/h (30m/s), w = v/R = 100 rad/s.
                # Increment at 120Hz = 100/120 = 0.8 rad.
                # Actually, let's just multiply by 120 if it seems like an increment. 
                # But let's plot raw for now or rate. I'll just assume they are rates initially
                pass
                
            # If it is indeed rate (rad/s), speed = abs(gz) * R.
            # Actually OB_GINS usually stores increments. dt = 1/120.0
            # Let's read the first value
            # Let's just plot what's in column 3, and label as raw value. We will convert it to rate assuming it's increment if max_val is small.
            if max_val < 10.0:
                gz_rate = gz * 120.0 # Convert increment to rate (rad/s)
            else:
                gz_rate = gz # already rate
                
            wheel_speed = np.abs(gz_rate) * wheel_radius
            
            axes[0].plot(t_imu, gz_rate, label=f"{name} Gyro Z", color=colors[name], alpha=0.7)
            axes[1].plot(t_imu, wheel_speed, label=f"{name} Speed (|w|*R)", color=colors[name], alpha=0.7)

    axes[2].plot(t_gnss, gnss_speed, 'k-', label="GNSS Speed", linewidth=2)
    
    # Settings for axes
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
    
    save_path = os.path.join(out_dir, "wheel_gnss_speed_60_80s.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to: {save_path}")

if __name__ == "__main__":
    analyze_and_plot()
