import numpy as np
import os

base_dir = "D:/Code/dataset/WID/Datasets/transformedData2/four_wheel_dataset_PassengerCar/trial01/"
gnss_file = os.path.join(base_dir, "GNSS_use.txt")

gnss = np.loadtxt(gnss_file)
t0 = gnss[0, 0]

mask = (gnss[:, 0] >= t0 + 50) & (gnss[:, 0] <= t0 + 90)
gf = gnss[mask]

# lat, lon are col 1, 2
import matplotlib.pyplot as plt

plt.figure()
plt.plot(gf[:, 2], gf[:, 1], '.-')
plt.title('GNSS Position (Lon, Lat) 50s-90s')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.savefig('D:/Code/dataset/WID/Datasets/transformedData2/output/gnss_pos_check.png')
print("Position plot saved.")

# Print distance between 50s and 90s
if len(gf) > 0:
    lat0, lon0 = gf[0, 1], gf[0, 2]
    lat1, lon1 = gf[-1, 1], gf[-1, 2]
    
    # Very rough approx distance
    dlat = (lat1 - lat0) * 111320
    dlon = (lon1 - lon0) * 111320 * np.cos(np.radians(lat0))
    dist = np.sqrt(dlat**2 + dlon**2)
    print(f"GNSS Position Moved: ~{dist:.2f} meters between 50s and 90s")
    
    print(f"GNSS velocity values at start: vn={gf[0, 4]:.4f}, ve={gf[0, 5]:.4f}, vd={gf[0, 6]:.4f}")
    print(f"GNSS velocity values at end: vn={gf[-1, 4]:.4f}, ve={gf[-1, 5]:.4f}, vd={gf[-1, 6]:.4f}")
