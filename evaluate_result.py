import sys
import numpy as np
import argparse

def load_trajectory(file_path):
    # 10 columns: time, lat, lon, alt, vx, vy, vz, roll, pitch, yaw
    # We only care about time, lat, lon, alt for now
    try:
        data = np.loadtxt(file_path, delimiter=',')
    except:
        data = np.loadtxt(file_path) # try default whitespace
    return data

def load_gnss(file_path):
    # GNSS format: time, lat, lon, alt, ...
    try:
        data = np.loadtxt(file_path, delimiter=',')
    except:
        data = np.loadtxt(file_path)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True)
    parser.add_argument('--truth', required=True)
    args = parser.parse_args()

    traj = load_trajectory(args.result)
    gnss = load_gnss(args.truth)

    # Simple nearest neighbor matching based on time
    # Assumes gnss frequency is lower or equal to traj
    
    errors_enu = []
    
    # Earth radius approximation
    R = 6378137.0
    
    for g in gnss:
        t_g = g[0]
        # Find closest point in traj
        idx = np.argmin(np.abs(traj[:, 0] - t_g))
        t_t = traj[idx, 0]
        
        if abs(t_t - t_g) > 0.05: # Max 50ms sync error allowed
            continue
            
        # Calc error (approximation for small area)
        lat_g, lon_g, alt_g = g[1], g[2], g[3]
        lat_t, lon_t, alt_t = traj[idx, 1], traj[idx, 2], traj[idx, 3]
        
        d_lat = (lat_t - lat_g) * (np.pi / 180.0)
        d_lon = (lon_t - lon_g) * (np.pi / 180.0)
        d_alt = alt_t - alt_g
        
        d_n = d_lat * R
        d_e = d_lon * R * np.cos(lat_g * np.pi / 180.0)
        d_u = d_alt
        
        errors_enu.append([d_e, d_n, d_u])
        
    errors_enu = np.array(errors_enu)
    rmse = np.sqrt(np.mean(errors_enu**2, axis=0))
    
    print(f"Evaluation Results:")
    print(f"Time Sync Pairs: {len(errors_enu)}")
    print(f"RMSE East:  {rmse[0]:.4f} m")
    print(f"RMSE North: {rmse[1]:.4f} m")
    print(f"RMSE Up:    {rmse[2]:.4f} m")
    print(f"RMSE 3D:    {np.linalg.norm(rmse):.4f} m")

if __name__ == "__main__":
    main()
