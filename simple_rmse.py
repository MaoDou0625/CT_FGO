import math
import sys

def parse_line(line):
    parts = line.strip().split()
    return [float(x) for x in parts]

def read_file(filepath):
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#') or line.startswith('%'):
                    continue
                data.append(parse_line(line))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)
    return data

def calc_rmse(result_path, truth_path):
    print(f"Comparing:\nResult: {result_path}\nTruth:  {truth_path}")
    
    res_data = read_file(result_path)
    gt_data = read_file(truth_path)

    sum_sq_n = 0.0 # North (Lat)
    sum_sq_e = 0.0 # East (Lon)
    sum_sq_u = 0.0 # Up (Alt)
    count = 0
    
    gt_idx = 0
    
    # WGS84 constants
    Re = 6378137.0
    
    for r_row in res_data:
        t_r = r_row[0]
        
        # Find closest GT within 50ms
        best_gt = None
        min_dt = 1e9
        
        while gt_idx < len(gt_data) - 1 and gt_data[gt_idx+1][0] < t_r:
            gt_idx += 1
            
        # Check current and next
        candidates = []
        if gt_idx < len(gt_data): candidates.append(gt_data[gt_idx])
        if gt_idx + 1 < len(gt_data): candidates.append(gt_data[gt_idx+1])
        
        for cand in candidates:
            dt = abs(cand[0] - t_r)
            if dt < min_dt and dt < 0.05:
                min_dt = dt
                best_gt = cand
        
        if best_gt:
            lat_r, lon_r, alt_r = r_row[1], r_row[2], r_row[3]
            lat_g, lon_g, alt_g = best_gt[1], best_gt[2], best_gt[3]
            
            # Diff in meters
            # Lat -> North
            d_n = (lat_r - lat_g) * (math.pi / 180.0) * Re
            # Lon -> East (adjust for latitude)
            d_e = (lon_r - lon_g) * (math.pi / 180.0) * Re * math.cos(lat_g * math.pi / 180.0)
            # Alt -> Up
            d_u = alt_r - alt_g
            
            sum_sq_n += d_n**2
            sum_sq_e += d_e**2
            sum_sq_u += d_u**2
            count += 1

    if count == 0:
        print("No matches found within tolerance!")
        return

    rmse_n = math.sqrt(sum_sq_n / count)
    rmse_e = math.sqrt(sum_sq_e / count)
    rmse_u = math.sqrt(sum_sq_u / count)
    
    rmse_2d = math.sqrt((sum_sq_n + sum_sq_e) / count)
    rmse_3d = math.sqrt((sum_sq_n + sum_sq_e + sum_sq_u) / count)

    print(f"\nMatched {count} points.")
    print("-" * 30)
    print(f"RMSE North: {rmse_n:.4f} m")
    print(f"RMSE East:  {rmse_e:.4f} m")
    print(f"RMSE Up:    {rmse_u:.4f} m")
    print("-" * 30)
    print(f"RMSE 2D:    {rmse_2d:.4f} m")
    print(f"RMSE 3D:    {rmse_3d:.4f} m")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 simple_rmse.py <result_file> <truth_file>")
    else:
        calc_rmse(sys.argv[1], sys.argv[2])
