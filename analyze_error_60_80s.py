import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    return np.loadtxt(file_path)

result_path = "D:/Code/dataset/WID/Datasets/transformedData2/output/ct_trajectory.txt"
truth_path = "D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/GNSS_use.txt"
imu1_path = "D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/Center_IMU1.txt"

res = load_data(result_path)
truth = load_data(truth_path)
try:
    imu1 = load_data(imu1_path)
    has_imu = True
except Exception as e:
    print(f"Could not load IMU data: {e}")
    has_imu = False

# Columns in result: Time, Lat(deg), Lon(deg), Alt(m), Vn, Ve, Vd, Roll, Pitch, Yaw
t_res = res[:, 0]
lat_res = res[:, 1]
lon_res = res[:, 2]
alt_res = res[:, 3]
vel_res = res[:, 4:7]

# Columns in truth: Time, Lat(deg), Lon(deg), Alt(m), std...
t_truth = truth[:, 0]
lat_truth = truth[:, 1]
lon_truth = truth[:, 2]
alt_truth = truth[:, 3]

# Calculate GNSS Velocity from differences
# v = dx / dt. dx = (lat2 - lat1) * R * pi/180
R_earth = 6378137.0
dt_truth = np.diff(t_truth)
dlat_truth = np.diff(lat_truth) * np.pi / 180.0 * R_earth
dlon_truth = np.diff(lon_truth) * np.pi / 180.0 * R_earth * np.cos(np.mean(lat_truth) * np.pi / 180.0)

vn_truth = dlat_truth / dt_truth
ve_truth = dlon_truth / dt_truth
v2d_truth = np.sqrt(vn_truth**2 + ve_truth**2)
t_v_truth = t_truth[:-1] + dt_truth / 2.0

# Filter 60s to 80s for result
mask_res = (t_res >= 60) & (t_res <= 80)
t_res_60_80 = t_res[mask_res]
lat_res_60_80 = lat_res[mask_res]
lon_res_60_80 = lon_res[mask_res]
vel_res_60_80 = vel_res[mask_res]
speed2d_res_60_80 = np.linalg.norm(vel_res_60_80[:, :2], axis=1)

# Filter 60s to 80s for truth positions
mask_truth = (t_truth >= 60) & (t_truth <= 80)
t_truth_60_80 = t_truth[mask_truth]
lat_truth_60_80 = lat_truth[mask_truth]
lon_truth_60_80 = lon_truth[mask_truth]

# Filter 60s to 80s for truth velocities
mask_v_truth = (t_v_truth >= 60) & (t_v_truth <= 80)
t_v_truth_60_80 = t_v_truth[mask_v_truth]
speed2d_truth_60_80 = v2d_truth[mask_v_truth]

# Convert Lat/Lon error to meters (approx)
lat_err_m = (lat_res_60_80 - np.interp(t_res_60_80, t_truth_60_80, lat_truth_60_80)) * np.pi / 180.0 * R_earth
lon_err_m = (lon_res_60_80 - np.interp(t_res_60_80, t_truth_60_80, lon_truth_60_80)) * np.pi / 180.0 * R_earth * np.cos(lat_res_60_80.mean() * np.pi / 180.0)

err_pos = np.sqrt(lat_err_m**2 + lon_err_m**2)

max_err_idx = np.argmax(err_pos)
max_err_time = t_res_60_80[max_err_idx]
max_err_val = err_pos[max_err_idx]

print(f"==================================================")
print(f"Max 2D Position Error in 60-80s: {max_err_val:.4f} m at time {max_err_time:.4f} s")
print(f"At this time (t={max_err_time:.4f}):")
print(f"  Estimated Velocity (N, E, D): {vel_res_60_80[max_err_idx]} m/s")
print(f"  Estimated Speed 2D: {speed2d_res_60_80[max_err_idx]:.4f} m/s")
gnss_speed_at_max = np.interp(max_err_time, t_v_truth_60_80, speed2d_truth_60_80)
print(f"  Implied GNSS Speed 2D: {gnss_speed_at_max:.4f} m/s")

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axs[0].plot(t_res_60_80, err_pos, label='2D Position Error')
axs[0].axvline(max_err_time, color='r', linestyle='--', label=f'Max Error at {max_err_time:.2f}s')
axs[0].set_ylabel('Error (m)')
axs[0].set_title('Position Error (60s - 80s)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_res_60_80, speed2d_res_60_80, label='Estimated Speed 2D')
axs[1].plot(t_v_truth_60_80, speed2d_truth_60_80, label='Implied GNSS Speed 2D', alpha=0.7)
axs[1].axvline(max_err_time, color='r', linestyle='--')
axs[1].set_ylabel('Speed (m/s)')
axs[1].set_title('Speed Profiles')
axs[1].legend()
axs[1].grid(True)

if has_imu:
    t_imu1 = imu1[:, 0]
    # IMU format is likely t, gx, gy, gz, ax, ay, az. Wheel angular velocity is usually gy or gz. 
    # Let's plot gy and gz
    gy_imu1 = imu1[:, 2]
    gz_imu1 = imu1[:, 3]
    mask_imu = (t_imu1 >= 60) & (t_imu1 <= 80)
    t_imu_60_80 = t_imu1[mask_imu]
    gy_60_80 = gy_imu1[mask_imu]
    gz_60_80 = gz_imu1[mask_imu]
    
    wheel_speed_approx = np.abs(gy_60_80) * 0.3 # V = omega * r
    
    axs[2].plot(t_imu_60_80, gy_60_80, label='IMU1 Gyro Y (rad/s)')
    axs[2].plot(t_imu_60_80, gz_60_80, label='IMU1 Gyro Z (rad/s)', alpha=0.5)
    axs[2].axvline(max_err_time, color='r', linestyle='--')
    axs[2].set_ylabel('Angular Vel (rad/s)')
    axs[2].set_title('Wheel IMU1 Angular Velocity')
    axs[2].legend()
    axs[2].grid(True)
    
    wheel_speed_at_max = np.interp(max_err_time, t_imu_60_80, wheel_speed_approx)
    print(f"  Approx Wheel Speed at max error (|Gy| * 0.3): {wheel_speed_at_max:.4f} m/s")

axs[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('error_speed_60_80s.png')
print("Saved plot to error_speed_60_80s.png")
print(f"==================================================")
