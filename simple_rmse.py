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

    # Simple nearest neighbor matching by timestamp (Column 0)
    # Assumes both are sorted by time
    
    error_sum_sq = 0.0
    count = 0
    
    gt_idx = 0
    matched_count = 0

    # GNSS/Truth format: time, lat, lon, alt, ...
    # Result format: time, lat, lon, alt, ...
    
    # WGS84 constants for simple meter conversion
    Re = 6378137.0
    
    for r_row in res_data:
        t_r = r_row[0]
        
        # Find closest GT
        best_gt = None
        min_dt = 1e9
        
        # Move gt_idx forward
        while gt_idx < len(gt_data) - 1 and gt_data[gt_idx+1][0] < t_r:
            gt_idx += 1
            
        # Check gt_idx and gt_idx+1
        if gt_idx < len(gt_data):
            dt = abs(gt_data[gt_idx][0] - t_r)
            if dt < 0.05: # 50ms tolerance
                best_gt = gt_data[gt_idx]
                min_dt = dt
        
        if gt_idx + 1 < len(gt_data):
            dt_next = abs(gt_data[gt_idx+1][0] - t_r)
            if dt_next < min_dt and dt_next < 0.05:
                best_gt = gt_data[gt_idx+1]
        
        if best_gt:
            # Lat/Lon/Alt diff in meters
            lat_r, lon_r, alt_r = r_row[1], r_row[2], r_row[3]
            lat_g, lon_g, alt_g = best_gt[1], best_gt[2], best_gt[3]
            
            d_lat = (lat_r - lat_g) * (math.pi / 180.0) * Re
            d_lon = (lon_r - lon_g) * (math.pi / 180.0) * Re * math.cos(lat_g * math.pi / 180.0)
            d_alt = alt_r - alt_g
            
            dist_sq = d_lat**2 + d_lon**2 + d_alt**2
            error_sum_sq += dist_sq
            count += 1

    if count == 0:
        print("No matches found within tolerance!")
        return

    rmse = math.sqrt(error_sum_sq / count)
    print(f"\nMatched {count} points.")
    print(f"Position RMSE: {rmse:.4f} meters")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 simple_rmse.py <result_file> <truth_file>")
    else:
        calc_rmse(sys.argv[1], sys.argv[2])
