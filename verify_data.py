import numpy as np
import os

base_dir = "D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/"
gnss_file = os.path.join(base_dir, "GNSS_use.txt")
wheel_files = {
    "FL": os.path.join(base_dir, "Center_IMU1.txt"),
    "RL": os.path.join(base_dir, "Center_IMU2.txt"),
    "FR": os.path.join(base_dir, "Center_IMU3.txt"),
    "RR": os.path.join(base_dir, "Center_IMU4.txt")
}

gnss = np.loadtxt(gnss_file)
t0 = gnss[0, 0]

mask_gnss = (gnss[:, 0] >= t0 + 60) & (gnss[:, 0] <= t0 + 80)
gnss_focus = gnss[mask_gnss]
vn = gnss_focus[:, 4]
ve = gnss_focus[:, 5]
gnss_speed = np.sqrt(vn**2 + ve**2)

print("--- 60s to 80s Data Verification ---")
print(f"GNSS Speed: min = {np.min(gnss_speed):.2f} m/s, max = {np.max(gnss_speed):.2f} m/s, mean = {np.mean(gnss_speed):.2f} m/s")

wheel_radius = 0.3

for name, file_path in wheel_files.items():
    if os.path.exists(file_path):
        imu = np.loadtxt(file_path)
        mask_imu = (imu[:, 0] >= t0 + 60) & (imu[:, 0] <= t0 + 80)
        imu_focus = imu[mask_imu]
        gz = imu_focus[:, 3]
        max_val = np.max(np.abs(gz))
        
        # Check if incremental or rate based on max value
        if max_val < 10.0:
            gz_rate = gz * 120.0
        else:
            gz_rate = gz
            
        wheel_speed = np.abs(gz_rate) * wheel_radius
        print(f"{name} Wheel Speed: min = {np.min(wheel_speed):.2f} m/s, max = {np.max(wheel_speed):.2f} m/s, mean = {np.mean(wheel_speed):.2f} m/s")
        print(f"  {name} Gyro Z rate: min = {np.min(np.abs(gz_rate)):.2f} rad/s, max = {np.max(np.abs(gz_rate)):.2f} rad/s")
