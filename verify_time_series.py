import numpy as np
import os

base_dir = "D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/"
gnss_file = os.path.join(base_dir, "GNSS_use.txt")
rl_file = os.path.join(base_dir, "Center_IMU2.txt")

gnss = np.loadtxt(gnss_file)
t0 = gnss[0, 0]

mask_gnss = (gnss[:, 0] >= t0 + 55) & (gnss[:, 0] <= t0 + 85)
gnss_focus = gnss[mask_gnss]
t_g = gnss_focus[:, 0] - t0
vn = gnss_focus[:, 4]
ve = gnss_focus[:, 5]
v_g = np.sqrt(vn**2 + ve**2)

rl = np.loadtxt(rl_file)
mask_rl = (rl[:, 0] >= t0 + 55) & (rl[:, 0] <= t0 + 85)
rl_focus = rl[mask_rl]
t_r = rl_focus[:, 0] - t0
gz = rl_focus[:, 3]
v_r = np.abs(gz * 120.0) * 0.3

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(t_g, v_g, label='GNSS Speed')
plt.plot(t_r, v_r, label='RL Wheel Speed', alpha=0.7)
plt.xlabel('Time (s) from t0')
plt.ylabel('Speed (m/s)')
plt.title('GNSS vs Wheel Speed (55s - 85s)')
plt.legend()
plt.grid(True)
plt.savefig('D:/Code/dataset/WID/Datasets/transformedData2/output/speed_check.png')
print("Plot saved to D:/Code/dataset/WID/Datasets/transformedData2/output/speed_check.png")

# Also print max speeds in 10s bins
for start in range(50, 90, 10):
    m_g = (gnss[:, 0] >= t0 + start) & (gnss[:, 0] < t0 + start + 10)
    v_g_bin = np.sqrt(gnss[m_g, 4]**2 + gnss[m_g, 5]**2) if np.sum(m_g) > 0 else [0]
    
    m_r = (rl[:, 0] >= t0 + start) & (rl[:, 0] < t0 + start + 10)
    v_r_bin = np.abs(rl[m_r, 3] * 120.0) * 0.3 if np.sum(m_r) > 0 else [0]
    
    print(f"[{start}s - {start+10}s] GNSS Max Speed: {np.max(v_g_bin):.2f} m/s, Wheel RL Max Speed: {np.max(v_r_bin):.2f} m/s")
