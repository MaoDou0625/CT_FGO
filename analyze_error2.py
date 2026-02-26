import numpy as np

res_file = "D:/Code/dataset/WID/Datasets/transformedData2/output/ct_trajectory.txt"
truth_file = "D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/GNSS_use.txt"

res = np.loadtxt(res_file)
truth = np.loadtxt(truth_file)

t0 = truth[0, 0]
mask_res = (res[:, 0] >= t0 + 60) & (res[:, 0] <= t0 + 80)
res_focus = res[mask_res]

mask_truth = (truth[:, 0] >= t0 + 60) & (truth[:, 0] <= t0 + 80)
truth_focus = truth[mask_truth]

if len(res_focus) > 0 and len(truth_focus) > 0:
    # Interpolate truth
    t_res = res_focus[:, 0]
    t_truth_f = truth_focus[:, 0]
    
    # We just want a rough idea of the error
    lat_res = res_focus[:, 1]
    lon_res = res_focus[:, 2]
    
    lat_truth = np.interp(t_res, t_truth_f, truth_focus[:, 1])
    lon_truth = np.interp(t_res, t_truth_f, truth_focus[:, 2])
    
    # Approx error in meters
    err_lat = (lat_res - lat_truth) * 111320.0
    err_lon = (lon_res - lon_truth) * 111320.0 * np.cos(np.radians(lat_truth))
    
    err_mag = np.sqrt(err_lat**2 + err_lon**2)
    print(f"Max horizontal error 60s-80s: {np.max(err_mag):.3f} m")
    
    # Print max accel from res
    v_n = res_focus[:, 4]
    v_e = res_focus[:, 5]
    speed = np.sqrt(v_n**2 + v_e**2)
    accel = np.diff(speed) / np.diff(t_res)
    print(f"Max computed acceleration: {np.max(np.abs(accel)):.3f} m/s^2")
