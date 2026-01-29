import sys
import numpy as np
import glob
import os
import yaml

def quat2rpy(q):
    # q: [x, y, z, w]
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)
        
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])

def get_initial_values(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    init_vals = {}
    for key, val in config.items():
        if isinstance(val, dict) and 'type' in val:
            # Parse extrinsic rotation (degrees)
            rpy_init = [0, 0, 0]
            if 'extrinsic_rotation' in val:
                rpy_init = val['extrinsic_rotation']
            
            # Parse lever arm
            lever_init = [0, 0, 0]
            if 'odolever' in val: # Wheel lever arm
                lever_init = val['odolever']
            
            # Parse radius
            rad_init = 0.3
            if 'wheel_radius' in val:
                rad_init = val['wheel_radius']

            init_vals[key] = {
                'rpy': np.array(rpy_init),
                'lever': np.array(lever_init),
                'radius': rad_init
            }
    return init_vals

def main():
    output_dir = "/mnt/d/Code/dataset/WID/Datasets/transformedData2/output"
    config_path = "CT_FGO/config/ob_gins_ct_wsl.yaml"
    
    try:
        init_vals = get_initial_values(config_path)
    except Exception as e:
        print(f"Could not load config: {e}")
        init_vals = {}

    error_files = glob.glob(os.path.join(output_dir, "errors_*.txt"))
    
    print(f"{'Sensor':<12} | {'Lever Arm Delta [mm]':<22} | {'Install Angle Delta [deg]':<26} | {'Radius [m]':<12}")
    print("-" * 80)
    
    for f in sorted(error_files):
        name = os.path.basename(f).replace("errors_", "").replace(".txt", "")
        if name not in init_vals: continue # Skip if not in config
        
        try:
            data = np.loadtxt(f, delimiter=None)
            if data.ndim == 1: last_row = data
            else: last_row = data[-1]
            
            # Format: t(1), bg(3), ba(3), lever(3), q(4)
            # CAUTION: The saved lever arm in errors_*.txt is l_body_sensor (Main Lever), not l_sensor_odopoint (Wheel Lever)!
            # Wait, StandardImuProcessor saves l_body_sensor.
            # WheelImuProcessor saves l_body_sensor too?
            # Let's check imu_processor.cc SaveErrors.
            # It saves l_body_sensor_.
            
            # WheelImuProcessor has TWO levers: l_body_sensor_ (Extrinsics) and l_sensor_odopoint_ (Optimized).
            # The optimization target was l_sensor_odopoint_.
            # BUT SaveErrors only saves l_body_sensor_.
            # I need to update SaveErrors to save l_sensor_odopoint_ for Wheel units!
            
            # Current script is useless for optimized lever arm check without code change.
            print(f"Skipping {name}: Output file does not contain optimized wheel parameters yet.")
            
        except Exception as e:
            print(f"Error reading {name}: {e}")

if __name__ == "__main__":
    main()
