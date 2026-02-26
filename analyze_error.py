import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_data():
    res_file = "D:/Code/dataset/WID/Datasets/transformedData2/output/ct_trajectory.txt"
    truth_file = "D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/GNSS_use.txt"
    imu_main_file = "D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/Body_IMU.txt"
    
    print("--- Analysis of 60s-80s Error Spike ---")
    
    # Load GNSS truth to check for outages or jumps
    truth = np.loadtxt(truth_file)
    t_truth = truth[:, 0]
    
    # Filter 60s to 80s
    # Note: Global time vs relative time. 
    # Let's check the start time of GNSS.
    t0 = t_truth[0]
    mask = (t_truth >= t0 + 55) & (t_truth <= t0 + 85)
    t_focus = t_truth[mask]
    
    # Check GNSS gaps
    if len(t_focus) > 1:
        dts = np.diff(t_focus)
        max_dt = np.max(dts)
        print(f"Max GNSS gap between 55s-85s: {max_dt:.3f} s")
    else:
        print("Not enough GNSS data in this window.")
    
    # Load Result to check velocity and turns
    res = np.loadtxt(res_file)
    t_res = res[:, 0]
    mask_res = (t_res >= t0 + 55) & (t_res <= t0 + 85)
    
    if np.sum(mask_res) > 0:
        res_focus = res[mask_res]
        v_n, v_e, v_d = res_focus[:, 4], res_focus[:, 5], res_focus[:, 6]
        speed = np.sqrt(v_n**2 + v_e**2)
        print(f"Speed in 55s-85s: min = {np.min(speed):.2f} m/s, max = {np.max(speed):.2f} m/s")
        
        # Check yaw rate (turn)
        yaw = res_focus[:, 9] # degrees
        yaw_unwrapped = np.unwrap(yaw * np.pi / 180.0) * 180.0 / np.pi
        yaw_rate = np.diff(yaw_unwrapped) / np.diff(t_res[mask_res])
        print(f"Max absolute yaw rate: {np.max(np.abs(yaw_rate)):.2f} deg/s")
    
    # Load IMU to check raw dynamics (acceleration/turn)
    try:
        imu = np.loadtxt(imu_main_file)
        t_imu = imu[:, 0]
        mask_imu = (t_imu >= t0 + 55) & (t_imu <= t0 + 85)
        imu_focus = imu[mask_imu]
        
        # Gyro Z
        gz = imu_focus[:, 3] * 180.0 / np.pi # assuming rad/s
        ax, ay, az = imu_focus[:, 4], imu_focus[:, 5], imu_focus[:, 6]
        
        print(f"Max raw Gyro Z: {np.max(np.abs(gz)):.2f} deg/s")
        print(f"Max raw Lateral Accel (Ay): {np.max(np.abs(ay)):.2f} m/s^2")
        print(f"Max raw Longitudinal Accel (Ax): {np.max(np.abs(ax)):.2f} m/s^2")
    except Exception as e:
        print("Could not load IMU data for analysis:", e)
        
    print("---------------------------------------")

if __name__ == "__main__":
    analyze_data()
