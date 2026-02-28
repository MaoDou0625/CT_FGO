import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


R_EARTH = 6378137.0


def load_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Invalid nav/truth file: {path}")
    return data


def compute_truth_velocity(truth: np.ndarray):
    t_truth = truth[:, 0]
    lat_truth = truth[:, 1]
    lon_truth = truth[:, 2]
    alt_truth = truth[:, 3]

    dt = np.diff(t_truth)
    dlat_m = np.diff(lat_truth) * np.pi / 180.0 * R_EARTH
    dlon_m = np.diff(lon_truth) * np.pi / 180.0 * R_EARTH * np.cos(np.deg2rad(np.mean(lat_truth)))
    dalt_m = np.diff(alt_truth)
    vn = dlat_m / dt
    ve = dlon_m / dt
    vd = -dalt_m / dt
    t_mid = t_truth[:-1] + 0.5 * dt
    return t_mid, vn, ve, vd


def parse_segments(text: str):
    segments = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        t0_s, t1_s = item.split("-")
        t0, t1 = float(t0_s), float(t1_s)
        if t1 <= t0:
            raise ValueError(f"Invalid segment: {item}")
        segments.append((t0, t1))
    if not segments:
        raise ValueError("No valid segments provided")
    return segments


def segment_metrics(res: np.ndarray, truth: np.ndarray, seg, t_v_truth, vn_truth, ve_truth):
    t0, t1 = seg
    t_res = res[:, 0]
    lat_res = res[:, 1]
    lon_res = res[:, 2]
    vn_res = res[:, 4]
    ve_res = res[:, 5]

    t_truth = truth[:, 0]
    lat_truth = truth[:, 1]
    lon_truth = truth[:, 2]

    m_res = (t_res >= t0) & (t_res <= t1)
    m_truth = (t_truth >= t0) & (t_truth <= t1)
    m_v_truth = (t_v_truth >= t0) & (t_v_truth <= t1)
    if m_res.sum() < 2 or m_truth.sum() < 2 or m_v_truth.sum() < 2:
        return None

    t = t_res[m_res]
    lat_r = lat_res[m_res]
    lon_r = lon_res[m_res]
    vn_r = vn_res[m_res]
    ve_r = ve_res[m_res]

    lat_t = np.interp(t, t_truth[m_truth], lat_truth[m_truth])
    lon_t = np.interp(t, t_truth[m_truth], lon_truth[m_truth])
    vn_t = np.interp(t, t_v_truth[m_v_truth], vn_truth[m_v_truth])
    ve_t = np.interp(t, t_v_truth[m_v_truth], ve_truth[m_v_truth])

    lat_err_m = (lat_r - lat_t) * np.pi / 180.0 * R_EARTH
    lon_err_m = (lon_r - lon_t) * np.pi / 180.0 * R_EARTH * np.cos(np.deg2rad(np.mean(lat_r)))
    pos2d_err = np.sqrt(lat_err_m**2 + lon_err_m**2)

    speed_r = np.sqrt(vn_r**2 + ve_r**2)
    speed_t = np.sqrt(vn_t**2 + ve_t**2)
    speed_err = speed_r - speed_t

    max_idx = int(np.argmax(pos2d_err))
    return {
        "segment": f"{t0:.0f}-{t1:.0f}",
        "samples": int(t.size),
        "pos2d_rmse_m": float(np.sqrt(np.mean(pos2d_err**2))),
        "pos2d_mae_m": float(np.mean(np.abs(pos2d_err))),
        "pos2d_max_m": float(pos2d_err[max_idx]),
        "pos2d_max_t_s": float(t[max_idx]),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_err**2))),
        "speed_mae_mps": float(np.mean(np.abs(speed_err))),
    }


def save_csv(path: Path, rows):
    fields = [
        "segment",
        "samples",
        "pos2d_rmse_m",
        "pos2d_mae_m",
        "pos2d_max_m",
        "pos2d_max_t_s",
        "speed_rmse_mps",
        "speed_mae_mps",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_text(path: Path, rows):
    lines = []
    for r in rows:
        lines.append(
            f"{r['segment']}s | N={r['samples']} | "
            f"PosRMSE={r['pos2d_rmse_m']:.4f}m MAE={r['pos2d_mae_m']:.4f}m Max={r['pos2d_max_m']:.4f}m@{r['pos2d_max_t_s']:.3f}s | "
            f"SpeedRMSE={r['speed_rmse_mps']:.4f}m/s MAE={r['speed_mae_mps']:.4f}m/s"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(path: Path, rows):
    labels = [r["segment"] for r in rows]
    pos_rmse = [r["pos2d_rmse_m"] for r in rows]
    speed_rmse = [r["speed_rmse_mps"] for r in rows]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].bar(labels, pos_rmse)
    axs[0].set_ylabel("Pos2D RMSE (m)")
    axs[0].grid(True, axis="y", alpha=0.3)
    axs[1].bar(labels, speed_rmse, color="tab:orange")
    axs[1].set_ylabel("Speed RMSE (m/s)")
    axs[1].set_xlabel("Segment (s)")
    axs[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Segment Benchmark Summary")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run multi-segment benchmark for nav result.")
    parser.add_argument("--result", required=True, help="Path to ct_trajectory.txt")
    parser.add_argument("--truth", required=True, help="Path to GNSS_use.txt")
    parser.add_argument(
        "--segments",
        default="0-20,20-40,40-60,60-80,80-90",
        help="Comma-separated segments, e.g. 0-20,20-40,40-60,60-80,80-90",
    )
    parser.add_argument("--outdir", default="", help="Output directory")
    args = parser.parse_args()

    result_path = Path(args.result)
    truth_path = Path(args.truth)
    outdir = Path(args.outdir) if args.outdir else result_path.parent / "segment_benchmark"
    outdir.mkdir(parents=True, exist_ok=True)

    res = load_txt(result_path)
    truth = load_txt(truth_path)
    segments = parse_segments(args.segments)
    t_v_truth, vn_truth, ve_truth, _ = compute_truth_velocity(truth)

    rows = []
    for seg in segments:
        m = segment_metrics(res, truth, seg, t_v_truth, vn_truth, ve_truth)
        if m is not None:
            rows.append(m)

    if not rows:
        raise RuntimeError("No segment produced valid metrics")

    csv_path = outdir / "segment_metrics.csv"
    txt_path = outdir / "segment_metrics.txt"
    png_path = outdir / "segment_metrics.png"
    save_csv(csv_path, rows)
    save_text(txt_path, rows)
    save_plot(png_path, rows)

    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
