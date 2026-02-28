import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


R_EARTH = 6378137.0


def load_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Invalid file format: {path}")
    return data


def ll_to_local_m(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    north = (lat - lat0) * R_EARTH
    east = (lon - lon0) * R_EARTH * np.cos(lat0)
    return east, north


def main():
    parser = argparse.ArgumentParser(description="Analyze 60-80s navigation/velocity performance.")
    parser.add_argument("--result", required=True, help="Path to ct_trajectory.txt")
    parser.add_argument("--truth", required=True, help="Path to GNSS truth file")
    parser.add_argument("--t0", type=float, default=60.0, help="Start time (s)")
    parser.add_argument("--t1", type=float, default=80.0, help="End time (s)")
    parser.add_argument("--outdir", default="", help="Output directory (default: result dir)")
    args = parser.parse_args()

    result_path = Path(args.result)
    truth_path = Path(args.truth)
    outdir = Path(args.outdir) if args.outdir else result_path.parent / "analysis_60_80"
    outdir.mkdir(parents=True, exist_ok=True)

    res = load_txt(result_path)
    truth = load_txt(truth_path)

    # Result columns: t, lat, lon, alt, Vn, Ve, Vd, ...
    t_res = res[:, 0]
    lat_res = res[:, 1]
    lon_res = res[:, 2]
    alt_res = res[:, 3]
    vn_res = res[:, 4]
    ve_res = res[:, 5]
    vd_res = res[:, 6]

    # Truth columns: t, lat, lon, alt, ...
    t_truth = truth[:, 0]
    lat_truth = truth[:, 1]
    lon_truth = truth[:, 2]
    alt_truth = truth[:, 3]

    # Compute truth velocity from finite differences
    dt_truth = np.diff(t_truth)
    dlat_m = np.diff(lat_truth) * np.pi / 180.0 * R_EARTH
    dlon_m = np.diff(lon_truth) * np.pi / 180.0 * R_EARTH * np.cos(np.deg2rad(np.mean(lat_truth)))
    dalt_m = np.diff(alt_truth)
    vn_truth = dlat_m / dt_truth
    ve_truth = dlon_m / dt_truth
    vd_truth = -dalt_m / dt_truth
    t_v_truth = t_truth[:-1] + 0.5 * dt_truth

    # 60-80s masks
    m_res = (t_res >= args.t0) & (t_res <= args.t1)
    m_truth = (t_truth >= args.t0) & (t_truth <= args.t1)
    m_v_truth = (t_v_truth >= args.t0) & (t_v_truth <= args.t1)

    t = t_res[m_res]
    lat_r = lat_res[m_res]
    lon_r = lon_res[m_res]
    alt_r = alt_res[m_res]
    vn_r = vn_res[m_res]
    ve_r = ve_res[m_res]
    vd_r = vd_res[m_res]

    t_truth_seg = t_truth[m_truth]
    lat_truth_seg = lat_truth[m_truth]
    lon_truth_seg = lon_truth[m_truth]
    alt_truth_seg = alt_truth[m_truth]

    # Interpolate truth to result timestamps
    lat_t = np.interp(t, t_truth_seg, lat_truth_seg)
    lon_t = np.interp(t, t_truth_seg, lon_truth_seg)
    alt_t = np.interp(t, t_truth_seg, alt_truth_seg)
    vn_t = np.interp(t, t_v_truth[m_v_truth], vn_truth[m_v_truth])
    ve_t = np.interp(t, t_v_truth[m_v_truth], ve_truth[m_v_truth])
    vd_t = np.interp(t, t_v_truth[m_v_truth], vd_truth[m_v_truth])

    # Errors
    lat_err_m = (lat_r - lat_t) * np.pi / 180.0 * R_EARTH
    lon_err_m = (lon_r - lon_t) * np.pi / 180.0 * R_EARTH * np.cos(np.deg2rad(np.mean(lat_r)))
    alt_err_m = alt_r - alt_t
    pos2d_err = np.sqrt(lat_err_m**2 + lon_err_m**2)

    speed_r = np.sqrt(vn_r**2 + ve_r**2)
    speed_t = np.sqrt(vn_t**2 + ve_t**2)
    speed_err = speed_r - speed_t

    # Local trajectory
    lat0, lon0 = float(lat_t[0]), float(lon_t[0])
    e_r, n_r = ll_to_local_m(lat_r, lon_r, lat0, lon0)
    e_t, n_t = ll_to_local_m(lat_t, lon_t, lat0, lon0)

    # Plot 1: XY trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(e_t, n_t, label="Truth")
    plt.plot(e_r, n_r, label="Result", alpha=0.85)
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title(f"Trajectory ({args.t0:.0f}-{args.t1:.0f}s)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "nav_60_80_traj_xy.png", dpi=150)
    plt.close()

    # Plot 2: Position error
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(t, lat_err_m)
    axs[0].set_ylabel("Lat Err (m)")
    axs[0].grid(True)
    axs[1].plot(t, lon_err_m)
    axs[1].set_ylabel("Lon Err (m)")
    axs[1].grid(True)
    axs[2].plot(t, alt_err_m)
    axs[2].set_ylabel("Alt Err (m)")
    axs[2].grid(True)
    axs[3].plot(t, pos2d_err)
    axs[3].set_ylabel("2D Err (m)")
    axs[3].set_xlabel("Time (s)")
    axs[3].grid(True)
    fig.suptitle(f"Position Errors ({args.t0:.0f}-{args.t1:.0f}s)")
    fig.tight_layout()
    fig.savefig(outdir / "nav_60_80_position_errors.png", dpi=150)
    plt.close(fig)

    # Plot 3: Velocity components
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t, vn_r, label="Result")
    axs[0].plot(t, vn_t, label="Truth", alpha=0.8)
    axs[0].set_ylabel("Vn (m/s)")
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(t, ve_r, label="Result")
    axs[1].plot(t, ve_t, label="Truth", alpha=0.8)
    axs[1].set_ylabel("Ve (m/s)")
    axs[1].grid(True)
    axs[2].plot(t, vd_r, label="Result")
    axs[2].plot(t, vd_t, label="Truth", alpha=0.8)
    axs[2].set_ylabel("Vd (m/s)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)
    fig.suptitle(f"Velocity Components ({args.t0:.0f}-{args.t1:.0f}s)")
    fig.tight_layout()
    fig.savefig(outdir / "nav_60_80_velocity_components.png", dpi=150)
    plt.close(fig)

    # Plot 4: 2D speed
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(t, speed_r, label="Result")
    axs[0].plot(t, speed_t, label="Truth", alpha=0.8)
    axs[0].set_ylabel("Speed 2D (m/s)")
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(t, speed_err)
    axs[1].set_ylabel("Speed Err (m/s)")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(True)
    fig.suptitle(f"2D Speed Analysis ({args.t0:.0f}-{args.t1:.0f}s)")
    fig.tight_layout()
    fig.savefig(outdir / "nav_60_80_speed.png", dpi=150)
    plt.close(fig)

    # Summary
    max_idx = int(np.argmax(pos2d_err))
    pos_rmse = float(np.sqrt(np.mean(pos2d_err**2)))
    pos_mae = float(np.mean(np.abs(pos2d_err)))
    speed_rmse = float(np.sqrt(np.mean(speed_err**2)))
    speed_mae = float(np.mean(np.abs(speed_err)))
    summary = [
        f"Window: {args.t0:.3f} - {args.t1:.3f} s",
        f"Samples: {t.size}",
        f"2D Pos RMSE: {pos_rmse:.6f} m",
        f"2D Pos MAE:  {pos_mae:.6f} m",
        f"2D Pos Max:  {pos2d_err[max_idx]:.6f} m @ t={t[max_idx]:.6f} s",
        f"Speed RMSE:  {speed_rmse:.6f} m/s",
        f"Speed MAE:   {speed_mae:.6f} m/s",
    ]
    (outdir / "nav_60_80_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("Analysis complete.")
    print(f"Output directory: {outdir}")
    for name in [
        "nav_60_80_traj_xy.png",
        "nav_60_80_position_errors.png",
        "nav_60_80_velocity_components.png",
        "nav_60_80_speed.png",
        "nav_60_80_summary.txt",
    ]:
        print(str(outdir / name))


if __name__ == "__main__":
    main()
