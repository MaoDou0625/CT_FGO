from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_ROOT = Path(r"D:\Code\dataset\Railway-Precise-Localization-data")
DEFAULT_IMU_MANIFEST = DEFAULT_DATA_ROOT / "ct_fgo_imu" / "manifest.csv"
DEFAULT_GNSS_MANIFEST = DEFAULT_DATA_ROOT / "ct_fgo_gnss" / "manifest.csv"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "ct_fgo_railway_nav"
DEFAULT_CONFIG_ROOT = DEFAULT_OUTPUT_ROOT / "configs"


@dataclass
class RunRecord:
    run_name: str
    imu_txt: Path
    gnss_txt: Path
    rtk_txt: Path
    imu_rate_hz: float
    start_time_s: float
    end_time_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run CT_FGO on Railway-Precise-Localization runs and plot altitude comparisons.")
    parser.add_argument("--imu-manifest", type=Path, default=DEFAULT_IMU_MANIFEST)
    parser.add_argument("--gnss-manifest", type=Path, default=DEFAULT_GNSS_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--binary", type=Path, default=None, help="Path to ob_gins_ct executable. Auto-detected when omitted.")
    parser.add_argument("--runs", nargs="*", default=None, help="Optional subset of run names.")
    parser.add_argument("--skip-run", action="store_true", help="Only generate plots from existing CT_FGO outputs.")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_run_records(imu_manifest: Path, gnss_manifest: Path, run_filter: set[str] | None) -> list[RunRecord]:
    imu_rows = {row["run_name"]: row for row in load_manifest(imu_manifest)}
    gnss_rows = {row["run_name"]: row for row in load_manifest(gnss_manifest)}

    records: list[RunRecord] = []
    for run_name in sorted(set(imu_rows) & set(gnss_rows)):
        if run_filter and run_name not in run_filter:
            continue

        imu_row = imu_rows[run_name]
        gnss_row = gnss_rows[run_name]
        start_time_s = max(float(imu_row["first_time_s"]), float(gnss_row["first_time_s"]))
        end_time_s = min(float(imu_row["last_time_s"]), float(gnss_row["last_time_s"]))
        if end_time_s <= start_time_s:
            continue

        records.append(
            RunRecord(
                run_name=run_name,
                imu_txt=Path(imu_row["output_txt"]),
                gnss_txt=Path(gnss_row["gnss_txt"]),
                rtk_txt=Path(gnss_row["rtk_txt"]),
                imu_rate_hz=float(imu_row["avg_rate_hz"]),
                start_time_s=start_time_s,
                end_time_s=end_time_s,
            )
        )
    return records


def find_binary(repo_root: Path) -> Path:
    candidates = [
        repo_root / "bin" / "ob_gins_ct.exe",
        repo_root / "bin" / "Release" / "ob_gins_ct.exe",
        repo_root / "build_auto" / "Release" / "ob_gins_ct.exe",
        repo_root / "build_railway" / "Release" / "ob_gins_ct.exe",
        repo_root / "build_vcpkg" / "Release" / "ob_gins_ct.exe",
        repo_root / "build_scan" / "Release" / "ob_gins_ct.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find ob_gins_ct.exe in bin/ or build_* release directories.")


def as_yaml_path(path: Path) -> str:
    return path.resolve().as_posix()


def write_config(record: RunRecord, config_path: Path, output_dir: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = f"""# Auto-generated for {record.run_name}
gnssfile: "{as_yaml_path(record.gnss_txt)}"
outputpath: "{as_yaml_path(output_dir)}"
save_multi_imu: false

imu_main:
  type: "standard"
  file: "{as_yaml_path(record.imu_txt)}"
  columns: 7
  rate_hz: {record.imu_rate_hz:.6f}
  antlever: [0.0, 0.0, 0.0]
  nhc:
    enable: true
    weight: 12.0
    interval_sec: 0.05
    odopoint: [0.0, 0.0, 0.0]
  imunoise:
    accel_noise: 80.0
    gyro_noise: 0.015
    accel_bias_rw: 30.0
    gyro_bias_rw: 10.0
    accel_corr_time: 3600.0
    gyro_corr_time: 3600.0

starttime: {record.start_time_s:.6f}
endtime: {record.end_time_s:.6f}
aligntime: 3.0
kf_interval_sec: 1.0
num_iterations: 10
isearth: true

gnss_quality:
  enable: true
  gap_threshold_sec: 2.5
  jump_distance_threshold_m: 40.0
  jump_speed_threshold_mps: 35.0
  horizontal_std_threshold_m: 8.0
  vertical_std_threshold_m: 12.0
  gap_scale: 0.15
  jump_scale: 0.05
  std_scale: 0.25
  min_scale: 0.01

gnss_bias:
  enable: false
  random_walk_sigma: 0.5
  correlation_time_sec: 120.0
  initial_bias_std: 5.0

gnss_innovation_gate:
  enable: true
  only_anomaly_context: true
  horizontal_threshold_m: 12.0
  vertical_threshold_m: 6.0
  sigma_threshold: 4.0
  cooldown_sec: 8.0
  reacquire_consecutive: 2
  reacquire_horizontal_threshold_m: 6.0
  reacquire_vertical_threshold_m: 3.0
  reacquire_sigma_threshold: 2.5
  rejected_scale: 0.05
  warmup_iterations: 4
  anomaly_context_samples: 3

comparison:
  enable: false
"""
    config_path.write_text(text, encoding="utf-8")


def run_solver(binary: Path, config_path: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            [str(binary), str(config_path)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


def load_matrix(path: Path, expected_cols: int | None = None) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, expected_cols or 0), dtype=float)
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if expected_cols is not None and data.shape[1] != expected_cols:
        raise ValueError(f"{path} expected {expected_cols} columns, got {data.shape[1]}")
    return data


def blh_to_enu(blh_deg: np.ndarray, origin_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(blh_deg[:, 0])
    lon = np.radians(blh_deg[:, 1])
    h = blh_deg[:, 2]

    lat0 = np.radians(origin_deg[0])
    lon0 = np.radians(origin_deg[1])
    h0 = origin_deg[2]

    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    def ecef(lat_rad: np.ndarray | float, lon_rad: np.ndarray | float, height_m: np.ndarray | float) -> np.ndarray:
        v = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
        x = (v + height_m) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (v + height_m) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (v * (1.0 - e2) + height_m) * np.sin(lat_rad)
        if np.isscalar(lat_rad) or np.ndim(lat_rad) == 0:
            return np.array([x, y, z], dtype=float)
        return np.column_stack((x, y, z))

    p = ecef(lat, lon, h)
    p0 = ecef(lat0, lon0, h0)
    dp = p - p0

    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)
    sin_lon0 = np.sin(lon0)
    cos_lon0 = np.cos(lon0)
    transform = np.array(
        [
            [-sin_lon0, cos_lon0, 0.0],
            [-sin_lat0 * cos_lon0, -sin_lat0 * sin_lon0, cos_lat0],
            [cos_lat0 * cos_lon0, cos_lat0 * sin_lon0, sin_lat0],
        ]
    )
    return dp @ transform.T


def find_low_weight_spans(weight_profile: np.ndarray) -> list[tuple[float, float]]:
    if weight_profile.size == 0:
        return []

    mask = weight_profile[:, 7] < 0.999
    if not np.any(mask):
        return []

    times = weight_profile[:, 0]
    spans: list[tuple[float, float]] = []
    start = None
    end = None

    for i, flagged in enumerate(mask):
        if not flagged:
            if start is not None and end is not None:
                spans.append((start, end))
                start = None
                end = None
            continue

        left_dt = times[i] - times[i - 1] if i > 0 else 0.5
        right_dt = times[i + 1] - times[i] if i + 1 < len(times) else 0.5
        left = times[i] - max(left_dt * 0.5, 0.5)
        right = times[i] + max(right_dt * 0.5, 0.5)

        if start is None:
            start = left
            end = right
        elif left <= end + 0.5:
            end = right
        else:
            spans.append((start, end))
            start = left
            end = right

    if start is not None and end is not None:
        spans.append((start, end))
    return spans


def plot_altitude(record: RunRecord, output_dir: Path) -> tuple[int, int]:
    nav = load_matrix(output_dir / "ct_trajectory.txt", expected_cols=10)
    gnss = load_matrix(record.gnss_txt, expected_cols=7)
    rtk = load_matrix(record.rtk_txt, expected_cols=4)
    weight_profile = load_matrix(output_dir / "gnss_weight_profile.txt", expected_cols=11)

    if nav.size == 0:
        raise FileNotFoundError(f"Missing CT_FGO trajectory for {record.run_name}")

    fig, ax = plt.subplots(figsize=(14, 6))
    spans = find_low_weight_spans(weight_profile)
    for start, end in spans:
        ax.axvspan(start, end, color="#f8c8c8", alpha=0.35)

    ax.plot(nav[:, 0], nav[:, 3], color="#1f4e79", linewidth=1.6, label="CT_FGO nav")
    if gnss.size:
        ax.plot(gnss[:, 0], gnss[:, 3], color="#6b7280", linewidth=1.0, alpha=0.9, label="GNSS")
    if rtk.size:
        ax.plot(rtk[:, 0], rtk[:, 3], color="#0b7a46", linewidth=1.3, label="RTK fix")

    ax.set_title(f"{record.run_name} altitude comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "altitude_comparison.png", dpi=160)
    plt.close(fig)

    return len(spans), int(weight_profile.shape[0])


def plot_horizontal_comparison(record: RunRecord, output_dir: Path) -> None:
    nav = load_matrix(output_dir / "ct_trajectory.txt", expected_cols=10)
    gnss = load_matrix(record.gnss_txt, expected_cols=7)
    rtk = load_matrix(record.rtk_txt, expected_cols=4)

    if nav.size == 0:
        raise FileNotFoundError(f"Missing CT_FGO trajectory for {record.run_name}")

    candidates = [nav[:, 1:4]]
    if gnss.size:
        candidates.append(gnss[:, 1:4])
    if rtk.size:
        candidates.append(rtk[:, 1:4])
    origin = candidates[0][0]

    nav_enu = blh_to_enu(nav[:, 1:4], origin)
    gnss_enu = blh_to_enu(gnss[:, 1:4], origin) if gnss.size else np.empty((0, 3))
    rtk_enu = blh_to_enu(rtk[:, 1:4], origin) if rtk.size else np.empty((0, 3))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(nav_enu[:, 0], nav_enu[:, 1], color="#1f4e79", linewidth=1.6, label="CT_FGO nav")
    if gnss_enu.size:
        ax.plot(gnss_enu[:, 0], gnss_enu[:, 1], color="#6b7280", linewidth=1.0, alpha=0.85, label="GNSS")
    if rtk_enu.size:
        ax.plot(rtk_enu[:, 0], rtk_enu[:, 1], color="#0b7a46", linewidth=1.3, label="RTK fix")
    ax.set_title(f"{record.run_name} horizontal comparison")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_comparison_2d.png", dpi=160)
    plt.close(fig)

    center = [float(origin[0]), float(origin[1])]
    fmap = folium.Map(location=center, zoom_start=15, control_scale=True)

    def add_track(points: np.ndarray, color: str, name: str) -> None:
        if points.size == 0:
            return
        latlon = points[:, :2].tolist()
        group = folium.FeatureGroup(name=name, show=True)
        folium.PolyLine(latlon, color=color, weight=3, opacity=0.9).add_to(group)
        folium.CircleMarker(latlon[0], radius=4, color=color, fill=True, fill_opacity=1.0, popup=f"{name} start").add_to(group)
        folium.CircleMarker(latlon[-1], radius=4, color=color, fill=True, fill_opacity=1.0, popup=f"{name} end").add_to(group)
        group.add_to(fmap)

    add_track(nav[:, 1:4], "#1f4e79", "CT_FGO nav")
    add_track(gnss[:, 1:4], "#6b7280", "GNSS")
    add_track(rtk[:, 1:4], "#0b7a46", "RTK fix")
    folium.LayerControl(collapsed=False).add_to(fmap)
    all_tracks = [nav[:, 1:3]]
    if gnss.size:
        all_tracks.append(gnss[:, 1:3])
    if rtk.size:
        all_tracks.append(rtk[:, 1:3])
    all_points = np.vstack(all_tracks)
    fmap.fit_bounds(
        [
            [float(np.min(all_points[:, 0])), float(np.min(all_points[:, 1]))],
            [float(np.max(all_points[:, 0])), float(np.max(all_points[:, 1]))],
        ]
    )
    fmap.save(output_dir / "trajectory_comparison_map.html")


def build_overview(records: list[RunRecord], output_root: Path) -> None:
    existing = [record for record in records if (output_root / record.run_name / "ct_trajectory.txt").exists()]
    if not existing:
        return

    cols = 2
    rows = int(np.ceil(len(existing) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows), squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    for idx, record in enumerate(existing):
        ax = axes[idx // cols][idx % cols]
        ax.set_visible(True)

        run_dir = output_root / record.run_name
        nav = load_matrix(run_dir / "ct_trajectory.txt", expected_cols=10)
        gnss = load_matrix(record.gnss_txt, expected_cols=7)
        rtk = load_matrix(record.rtk_txt, expected_cols=4)
        weight_profile = load_matrix(run_dir / "gnss_weight_profile.txt", expected_cols=11)

        for start, end in find_low_weight_spans(weight_profile):
            ax.axvspan(start, end, color="#f8c8c8", alpha=0.3)

        ax.plot(nav[:, 0], nav[:, 3], color="#1f4e79", linewidth=1.2)
        if gnss.size:
            ax.plot(gnss[:, 0], gnss[:, 3], color="#6b7280", linewidth=0.8, alpha=0.85)
        if rtk.size:
            ax.plot(rtk[:, 0], rtk[:, 3], color="#0b7a46", linewidth=1.0)

        ax.set_title(record.run_name, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Alt (m)")

    fig.tight_layout()
    fig.savefig(output_root / "altitude_overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    binary = args.binary.resolve() if args.binary else find_binary(repo_root)
    run_filter = set(args.runs) if args.runs else None

    records = build_run_records(args.imu_manifest, args.gnss_manifest, run_filter)
    if not records:
        raise SystemExit("No runs available after applying manifests/filter.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.config_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str | int | float]] = []
    for record in records:
        run_dir = args.output_root / record.run_name
        config_path = args.config_root / f"{record.run_name}.yaml"
        write_config(record, config_path, run_dir)

        return_code = 0
        if not args.skip_run:
            print(f"Running CT_FGO for {record.run_name}")
            return_code = run_solver(binary, config_path, run_dir / "run.log")
            if return_code != 0:
                print(f"  failed with exit code {return_code}")

        degraded_spans = 0
        profile_rows = 0
        if return_code == 0 and (run_dir / "ct_trajectory.txt").exists():
            degraded_spans, profile_rows = plot_altitude(record, run_dir)
            plot_horizontal_comparison(record, run_dir)

        summary_rows.append(
            {
                "run_name": record.run_name,
                "config_path": str(config_path),
                "output_dir": str(run_dir),
                "return_code": return_code,
                "degraded_spans": degraded_spans,
                "gnss_quality_rows": profile_rows,
            }
        )

    summary_path = args.output_root / "run_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_name", "config_path", "output_dir", "return_code", "degraded_spans", "gnss_quality_rows"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    build_overview(records, args.output_root)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
