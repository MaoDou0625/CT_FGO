from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from pyubx2 import UBXReader
from pyubx2.exceptions import UBXParseError, UBXTypeError


DATASET_README_URL = "https://github.com/ETH-PBL/Railway-Precise-Localization/blob/main/README.md"
EARTH_RADIUS_M = 6378137.0
IMU_ACC_SENSITIVITY_MG_PER_LSB = 0.061
IMU_GYR_SENSITIVITY_MDPS_PER_LSB = 4.37
ALT_JUMP_THRESHOLD_M = 2.0
DT_SAMPLE_LIMIT = 20000
MAX_CONTIGUOUS_RTK_DT_S = 2.5


@dataclass
class RunSummary:
    run_name: str
    csv_path: Path
    row_count: int
    duration_s: float
    raw_type_counts: Counter
    ubx_type_counts: Counter
    imu_median_dt_s: float | None
    imu_est_hz: float | None
    imu_avg_hz: float | None
    pvt_median_dt_s: float | None
    pvt_est_hz: float | None
    pvt_avg_hz: float | None
    cov_median_dt_s: float | None
    cov_est_hz: float | None
    cov_avg_hz: float | None
    pvt_count: int
    cov_count: int
    rtk_point_count: int
    rtk_fix_count: int
    rtk_float_count: int
    rtk_coverage: float | None
    rtk_alt_min_m: float | None
    rtk_alt_max_m: float | None
    rtk_alt_span_m: float | None
    rtk_alt_max_step_m: float | None
    rtk_alt_p95_step_m: float | None
    rtk_alt_jump_count_gt_2m: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ETH-PBL Railway Precise Localization RTK data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\Code\dataset\Railway-Precise-Localization-data\data"),
        help="Directory containing per-run subdirectories with CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs") / "railway_rtk",
        help="Directory for plots and CSV summaries.",
    )
    return parser.parse_args()


def parse_rtc_seconds(text: str) -> float | None:
    try:
        hh, mm, sec = text.split(":")
        ss, micros = sec.split(".")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(micros) / 1_000_000.0
    except ValueError:
        return None


def parse_gnss_datetime(msg) -> datetime | None:
    try:
        if not getattr(msg, "validDate", False) or not getattr(msg, "validTime", False):
            return None
        base = datetime(
            int(msg.year),
            int(msg.month),
            int(msg.day),
            int(msg.hour),
            int(msg.min),
            int(msg.second),
        )
        return base + timedelta(microseconds=round(int(msg.nano) / 1000))
    except (AttributeError, TypeError, ValueError):
        return None


def estimate_rate_hz(dt_values: list[float]) -> tuple[float | None, float | None]:
    positive = [dt for dt in dt_values if dt > 0]
    if not positive:
        return None, None
    median_dt = statistics.median(positive)
    if median_dt <= 0:
        return median_dt, None
    return median_dt, 1.0 / median_dt


def local_en_m(lat_deg: Iterable[float], lon_deg: Iterable[float], ref_lat_deg: float, ref_lon_deg: float) -> tuple[list[float], list[float]]:
    lat0 = math.radians(ref_lat_deg)
    lats = []
    lons = []
    for lat, lon in zip(lat_deg, lon_deg):
        north = math.radians(lat - ref_lat_deg) * EARTH_RADIUS_M
        east = math.radians(lon - ref_lon_deg) * EARTH_RADIUS_M * math.cos(lat0)
        lats.append(north)
        lons.append(east)
    return lons, lats


def analyze_run(csv_path: Path) -> tuple[RunSummary, list[dict]]:
    raw_type_counts: Counter = Counter()
    ubx_type_counts: Counter = Counter()
    imu_dt_samples: list[float] = []
    pvt_dt_samples: list[float] = []
    cov_dt_samples: list[float] = []
    rtk_records: list[dict] = []

    row_count = 0
    first_rtc_s = None
    last_rtc_s = None
    last_imu_rtc_s = None
    last_pvt_gnss_dt = None
    last_cov_gnss_dt = None
    alt_diffs: list[float] = []
    last_rtk_alt = None
    last_rtk_gnss_dt = None
    pvt_count = 0
    cov_count = 0
    rtk_fix_count = 0
    rtk_float_count = 0
    rtk_alt_min_m = None
    rtk_alt_max_m = None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            rtc_s = parse_rtc_seconds(row[0])
            raw_type = row[1]
            payload = row[2]

            if rtc_s is None:
                continue

            row_count += 1
            raw_type_counts[raw_type] += 1
            if first_rtc_s is None:
                first_rtc_s = rtc_s
            last_rtc_s = rtc_s

            if raw_type == "I":
                if last_imu_rtc_s is not None and len(imu_dt_samples) < DT_SAMPLE_LIMIT:
                    dt = rtc_s - last_imu_rtc_s
                    if dt > 0:
                        imu_dt_samples.append(dt)
                last_imu_rtc_s = rtc_s
                continue

            if raw_type != "U":
                continue

            try:
                message = UBXReader.parse(bytes.fromhex(payload))
            except (ValueError, UBXParseError, UBXTypeError):
                continue

            identity = getattr(message, "identity", "UNKNOWN")
            ubx_type_counts[identity] += 1

            if identity == "NAV-PVT":
                pvt_count += 1
                gnss_dt = parse_gnss_datetime(message)
                pvt_dt_s = None
                if gnss_dt is not None and last_pvt_gnss_dt is not None and len(pvt_dt_samples) < DT_SAMPLE_LIMIT:
                    pvt_dt_s = (gnss_dt - last_pvt_gnss_dt).total_seconds()
                    if pvt_dt_s > 0:
                        pvt_dt_samples.append(pvt_dt_s)
                if gnss_dt is not None:
                    last_pvt_gnss_dt = gnss_dt

                fix_ok = bool(getattr(message, "gnssFixOk", False))
                invalid_llh = bool(getattr(message, "invalidLlh", False))
                carr_soln = int(getattr(message, "carrSoln", 0))
                if fix_ok and not invalid_llh and carr_soln > 0:
                    alt_m = float(getattr(message, "hMSL")) / 1000.0
                    lat = float(getattr(message, "lat"))
                    lon = float(getattr(message, "lon"))
                    hacc_m = float(getattr(message, "hAcc")) / 1000.0
                    vacc_m = float(getattr(message, "vAcc")) / 1000.0
                    rel_min = (rtc_s - first_rtc_s) / 60.0 if first_rtc_s is not None else None
                    rtk_records.append(
                        {
                            "run_name": csv_path.parent.name,
                            "rtc_seconds": rtc_s,
                            "rtc_minutes": rel_min,
                            "gnss_time": gnss_dt.isoformat() if gnss_dt is not None else "",
                            "gnss_dt_s": pvt_dt_s,
                            "lat_deg": lat,
                            "lon_deg": lon,
                            "alt_m": alt_m,
                            "hacc_m": hacc_m,
                            "vacc_m": vacc_m,
                            "num_sv": int(getattr(message, "numSV", 0)),
                            "carr_soln": carr_soln,
                            "carr_label": "RTK Fix" if carr_soln == 2 else "RTK Float",
                        }
                    )
                    rtk_alt_min_m = alt_m if rtk_alt_min_m is None else min(rtk_alt_min_m, alt_m)
                    rtk_alt_max_m = alt_m if rtk_alt_max_m is None else max(rtk_alt_max_m, alt_m)
                    if last_rtk_alt is not None and gnss_dt is not None and last_rtk_gnss_dt is not None:
                        alt_dt = (gnss_dt - last_rtk_gnss_dt).total_seconds()
                        if 0 < alt_dt <= MAX_CONTIGUOUS_RTK_DT_S:
                            alt_diffs.append(abs(alt_m - last_rtk_alt))
                    last_rtk_alt = alt_m
                    if gnss_dt is not None:
                        last_rtk_gnss_dt = gnss_dt
                    if carr_soln == 2:
                        rtk_fix_count += 1
                    elif carr_soln == 1:
                        rtk_float_count += 1
                continue

            if identity == "NAV-COV":
                cov_count += 1
                gnss_dt = parse_gnss_datetime(message)
                if gnss_dt is not None and last_cov_gnss_dt is not None and len(cov_dt_samples) < DT_SAMPLE_LIMIT:
                    dt = (gnss_dt - last_cov_gnss_dt).total_seconds()
                    if dt > 0:
                        cov_dt_samples.append(dt)
                if gnss_dt is not None:
                    last_cov_gnss_dt = gnss_dt

    duration_s = (last_rtc_s - first_rtc_s) if (first_rtc_s is not None and last_rtc_s is not None) else 0.0
    imu_median_dt_s, imu_est_hz = estimate_rate_hz(imu_dt_samples)
    pvt_median_dt_s, pvt_est_hz = estimate_rate_hz(pvt_dt_samples)
    cov_median_dt_s, cov_est_hz = estimate_rate_hz(cov_dt_samples)
    imu_avg_hz = (raw_type_counts.get("I", 0) / duration_s) if duration_s > 0 and raw_type_counts.get("I", 0) > 0 else None
    pvt_avg_hz = (pvt_count / duration_s) if duration_s > 0 and pvt_count > 0 else None
    cov_avg_hz = (cov_count / duration_s) if duration_s > 0 and cov_count > 0 else None
    rtk_point_count = len(rtk_records)
    rtk_coverage = (rtk_point_count / pvt_count) if pvt_count else None
    alt_span = (rtk_alt_max_m - rtk_alt_min_m) if (rtk_alt_min_m is not None and rtk_alt_max_m is not None) else None
    alt_max_step = max(alt_diffs) if alt_diffs else None
    alt_p95_step = statistics.quantiles(alt_diffs, n=20)[18] if len(alt_diffs) >= 20 else (statistics.median(alt_diffs) if alt_diffs else None)
    alt_jump_count = sum(1 for diff in alt_diffs if diff > ALT_JUMP_THRESHOLD_M)

    summary = RunSummary(
        run_name=csv_path.parent.name,
        csv_path=csv_path,
        row_count=row_count,
        duration_s=duration_s,
        raw_type_counts=raw_type_counts,
        ubx_type_counts=ubx_type_counts,
        imu_median_dt_s=imu_median_dt_s,
        imu_est_hz=imu_est_hz,
        imu_avg_hz=imu_avg_hz,
        pvt_median_dt_s=pvt_median_dt_s,
        pvt_est_hz=pvt_est_hz,
        pvt_avg_hz=pvt_avg_hz,
        cov_median_dt_s=cov_median_dt_s,
        cov_est_hz=cov_est_hz,
        cov_avg_hz=cov_avg_hz,
        pvt_count=pvt_count,
        cov_count=cov_count,
        rtk_point_count=rtk_point_count,
        rtk_fix_count=rtk_fix_count,
        rtk_float_count=rtk_float_count,
        rtk_coverage=rtk_coverage,
        rtk_alt_min_m=rtk_alt_min_m,
        rtk_alt_max_m=rtk_alt_max_m,
        rtk_alt_span_m=alt_span,
        rtk_alt_max_step_m=alt_max_step,
        rtk_alt_p95_step_m=alt_p95_step,
        rtk_alt_jump_count_gt_2m=alt_jump_count,
    )
    return summary, rtk_records


def build_raw_format_table() -> pd.DataFrame:
    rows = [
        {
            "level": "raw_csv",
            "column": "timestamp",
            "dtype": "string",
            "unit": "hh:mm:ss.uuuuuu",
            "frequency": "all rows; board RTC clock",
            "description": "Board-relative RTC timestamp from power-up.",
        },
        {
            "level": "raw_csv",
            "column": "data_type",
            "dtype": "categorical string",
            "unit": "",
            "frequency": "all rows",
            "description": "Record type: I=IMU, U=UBX GNSS, ARR/DEP/BRK=event markers.",
        },
        {
            "level": "raw_csv",
            "column": "data",
            "dtype": "string",
            "unit": "hex or plain text",
            "frequency": "all rows",
            "description": "Payload. IMU rows are 12-byte hex, U rows are UBX hex, event rows are text.",
        },
        {
            "level": "decoded_imu",
            "column": "gyro_x / gyro_y / gyro_z",
            "dtype": "int16 decoded from hex",
            "unit": "deg/s raw scale, 4.37 mdps per LSB",
            "frequency": "about 1667 Hz",
            "description": "Gyroscope axes from 12-byte IMU payload, little-endian pairs per README parser.",
        },
        {
            "level": "decoded_imu",
            "column": "acc_x / acc_y / acc_z",
            "dtype": "int16 decoded from hex",
            "unit": "g raw scale, 0.061 mg per LSB",
            "frequency": "about 1667 Hz",
            "description": "Accelerometer axes from 12-byte IMU payload, little-endian pairs per README parser.",
        },
        {
            "level": "decoded_nav_pvt",
            "column": "lat / lon",
            "dtype": "float",
            "unit": "degrees",
            "frequency": "about 1 Hz on 2022-10-19, about 2 Hz on 2022-10-20",
            "description": "GNSS position from UBX NAV-PVT.",
        },
        {
            "level": "decoded_nav_pvt",
            "column": "alt_m",
            "dtype": "float",
            "unit": "meters above mean sea level",
            "frequency": "same as NAV-PVT",
            "description": "Computed from UBX hMSL / 1000.",
        },
        {
            "level": "decoded_nav_pvt",
            "column": "hacc_m / vacc_m",
            "dtype": "float",
            "unit": "meters",
            "frequency": "same as NAV-PVT",
            "description": "Horizontal and vertical accuracy from UBX hAcc/vAcc / 1000.",
        },
        {
            "level": "decoded_nav_pvt",
            "column": "carr_soln",
            "dtype": "int",
            "unit": "0=no RTK, 1=float, 2=fix",
            "frequency": "same as NAV-PVT",
            "description": "Carrier solution state used to select RTK samples.",
        },
        {
            "level": "decoded_nav_cov",
            "column": "covariance message",
            "dtype": "UBX NAV-COV",
            "unit": "",
            "frequency": "about 1 Hz on 2022-10-20 only",
            "description": "Available on day 2 according to upstream README and local parse counts.",
        },
    ]
    return pd.DataFrame(rows)


def plot_repeatability_map(rtk_df: pd.DataFrame, output_path: Path) -> None:
    if rtk_df.empty:
        return
    ref_lat = rtk_df["lat_deg"].mean()
    ref_lon = rtk_df["lon_deg"].mean()
    east_m, north_m = local_en_m(rtk_df["lat_deg"], rtk_df["lon_deg"], ref_lat, ref_lon)
    rtk_df = rtk_df.copy()
    rtk_df["east_m"] = east_m
    rtk_df["north_m"] = north_m

    runs = sorted(rtk_df["run_name"].unique())
    cmap = plt.get_cmap("tab20", len(runs))
    fig, ax = plt.subplots(figsize=(11, 8))
    for idx, run_name in enumerate(runs):
        sub = rtk_df[rtk_df["run_name"] == run_name].sort_values("gnss_time")
        ax.plot(sub["east_m"], sub["north_m"], lw=1.2, alpha=0.85, color=cmap(idx), label=run_name)
        if not sub.empty:
            ax.scatter(sub["east_m"].iloc[0], sub["north_m"].iloc[0], color=cmap(idx), marker="o", s=18)
            ax.scatter(sub["east_m"].iloc[-1], sub["north_m"].iloc[-1], color=cmap(idx), marker="x", s=18)

    ax.set_title("RTK Track Repeatability Across All Runs")
    ax.set_xlabel("East [m] relative to global RTK centroid")
    ax.set_ylabel("North [m] relative to global RTK centroid")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_altitude_panels(rtk_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    if rtk_df.empty:
        return
    runs = list(summary_df["run_name"])
    ncols = 3
    nrows = math.ceil(len(runs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, run_name in zip(axes, runs):
        sub = rtk_df[rtk_df["run_name"] == run_name].sort_values("rtc_minutes").reset_index(drop=True)
        if sub.empty:
            ax.set_title(f"{run_name}\nNo RTK points")
            ax.axis("off")
            continue
        alt_diff = sub["alt_m"].diff().abs()
        contiguous = sub["gnss_dt_s"].fillna(float("inf")) <= MAX_CONTIGUOUS_RTK_DT_S
        jumps = sub[(alt_diff > ALT_JUMP_THRESHOLD_M) & contiguous]
        fix = sub[sub["carr_soln"] == 2]
        flt = sub[sub["carr_soln"] == 1]
        ax.plot(sub["rtc_minutes"], sub["alt_m"], color="#9aa0a6", lw=0.8, alpha=0.7)
        if not flt.empty:
            ax.scatter(flt["rtc_minutes"], flt["alt_m"], s=6, color="#f59e0b", alpha=0.7, label="RTK float")
        if not fix.empty:
            ax.scatter(fix["rtc_minutes"], fix["alt_m"], s=6, color="#1f9d55", alpha=0.7, label="RTK fix")
        if not jumps.empty:
            ax.scatter(jumps["rtc_minutes"], jumps["alt_m"], s=22, color="#d11a2a", marker="x", label=">|2m| jump")
        max_step = alt_diff[contiguous].max()
        subtitle = f"max step={max_step:.2f} m" if pd.notna(max_step) else "max step=n/a"
        ax.set_title(f"{run_name}\n{subtitle}", fontsize=9)
        ax.set_xlabel("RTC time [min]")
        ax.set_ylabel("RTK altitude [m]")
        ax.grid(True, alpha=0.25)

    for ax in axes[len(runs):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("RTK Altitude per Run", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_pairwise_repeatability(rtk_df: pd.DataFrame) -> pd.DataFrame:
    if rtk_df.empty:
        return pd.DataFrame(columns=["group", "run_a", "run_b", "median_nn_m", "p95_nn_m", "max_nn_m"])

    enriched = rtk_df.copy()
    enriched["group"] = enriched["run_name"].str.split("_").str[3]
    ref_lat = enriched["lat_deg"].mean()
    ref_lon = enriched["lon_deg"].mean()
    east_m, north_m = local_en_m(enriched["lat_deg"], enriched["lon_deg"], ref_lat, ref_lon)
    enriched["east_m"] = east_m
    enriched["north_m"] = north_m

    rows: list[dict] = []
    for group, group_df in enriched.groupby("group"):
        run_names = sorted(group_df["run_name"].unique())
        for idx, run_a in enumerate(run_names):
            a_pts = group_df[group_df["run_name"] == run_a][["east_m", "north_m"]].to_numpy()
            if len(a_pts) == 0:
                continue
            for run_b in run_names[idx + 1 :]:
                b_pts = group_df[group_df["run_name"] == run_b][["east_m", "north_m"]].to_numpy()
                if len(b_pts) == 0:
                    continue
                d2 = ((a_pts[:, None, :] - b_pts[None, :, :]) ** 2).sum(axis=2)
                nearest = d2.min(axis=1) ** 0.5
                rows.append(
                    {
                        "group": group,
                        "run_a": run_a,
                        "run_b": run_b,
                        "median_nn_m": float(pd.Series(nearest).median()),
                        "p95_nn_m": float(pd.Series(nearest).quantile(0.95)),
                        "max_nn_m": float(nearest.max()),
                    }
                )
    return pd.DataFrame(rows).sort_values(["group", "median_nn_m"])


def write_report(output_dir: Path, summary_df: pd.DataFrame, repeatability_df: pd.DataFrame) -> None:
    total_runs = len(summary_df)
    total_rtk = int(summary_df["rtk_point_count"].sum())
    total_fix = int(summary_df["rtk_fix_count"].sum())
    total_float = int(summary_df["rtk_float_count"].sum())
    mean_imu_hz = summary_df["imu_est_hz"].dropna().mean()
    mean_imu_avg_hz = summary_df["imu_avg_hz"].dropna().mean()
    mean_pvt_hz = summary_df["pvt_avg_hz"].dropna().mean()
    mean_cov_hz = summary_df["cov_avg_hz"].dropna().mean()
    max_alt_step_row = summary_df.sort_values("rtk_alt_max_step_m", ascending=False).iloc[0]
    repeatability_lines = []
    if not repeatability_df.empty:
        grouped = repeatability_df.groupby("group")[["median_nn_m", "p95_nn_m"]].median().reset_index()
        for _, row in grouped.iterrows():
            repeatability_lines.append(
                f"- {row['group']}: pairwise median nearest-neighbor distance {row['median_nn_m']:.2f} m, pairwise p95 {row['p95_nn_m']:.2f} m"
            )

    lines = [
        "# Railway RTK Analysis",
        "",
        f"Source README: {DATASET_README_URL}",
        "",
        "## Key Findings",
        "",
        f"- Runs analyzed: {total_runs}",
        f"- Total RTK points (float + fix): {total_rtk}",
        f"- RTK fix points: {total_fix}",
        f"- RTK float points: {total_float}",
        f"- IMU average rate across runs: {mean_imu_avg_hz:.1f} Hz" if pd.notna(mean_imu_avg_hz) else "- IMU average rate across runs: n/a",
        f"- IMU median-dt implied rate across runs: {mean_imu_hz:.1f} Hz" if pd.notna(mean_imu_hz) else "- IMU median-dt implied rate across runs: n/a",
        f"- NAV-PVT average rate across runs: {mean_pvt_hz:.2f} Hz" if pd.notna(mean_pvt_hz) else "- NAV-PVT average rate across runs: n/a",
        f"- NAV-COV average rate across runs: {mean_cov_hz:.2f} Hz" if pd.notna(mean_cov_hz) else "- NAV-COV average rate across runs: n/a",
        f"- Largest RTK altitude step: {max_alt_step_row['rtk_alt_max_step_m']:.2f} m in {max_alt_step_row['run_name']}" if pd.notna(max_alt_step_row["rtk_alt_max_step_m"]) else "- Largest RTK altitude step: n/a",
        "",
        "## Repeatability",
        "",
        *repeatability_lines,
        "",
        "## Output Files",
        "",
        "- rtk_repeatability_map.png",
        "- rtk_altitude_panels.png",
        "- run_summary.csv",
        "- repeatability_pairwise.csv",
        "- raw_format_summary.csv",
        "- rtk_points.csv",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_root.glob("*/*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found under {data_root}")

    summaries: list[RunSummary] = []
    all_rtk_records: list[dict] = []
    for csv_path in csv_files:
        summary, rtk_records = analyze_run(csv_path)
        summaries.append(summary)
        all_rtk_records.extend(rtk_records)
        print(f"Analyzed {csv_path.parent.name}: rows={summary.row_count}, rtk={summary.rtk_point_count}")

    summary_df = pd.DataFrame(
        [
            {
                "run_name": s.run_name,
                "csv_path": str(s.csv_path),
                "row_count": s.row_count,
                "duration_min": s.duration_s / 60.0 if s.duration_s else None,
                "raw_I_count": s.raw_type_counts.get("I", 0),
                "raw_U_count": s.raw_type_counts.get("U", 0),
                "raw_ARR_count": s.raw_type_counts.get("ARR", 0),
                "raw_DEP_count": s.raw_type_counts.get("DEP", 0),
                "raw_BRK_count": s.raw_type_counts.get("BRK", 0),
                "nav_pvt_count": s.pvt_count,
                "nav_cov_count": s.cov_count,
                "imu_median_dt_ms": s.imu_median_dt_s * 1000.0 if s.imu_median_dt_s is not None else None,
                "imu_est_hz": s.imu_est_hz,
                "imu_avg_hz": s.imu_avg_hz,
                "pvt_median_dt_s": s.pvt_median_dt_s,
                "pvt_est_hz": s.pvt_est_hz,
                "pvt_avg_hz": s.pvt_avg_hz,
                "cov_median_dt_s": s.cov_median_dt_s,
                "cov_est_hz": s.cov_est_hz,
                "cov_avg_hz": s.cov_avg_hz,
                "rtk_point_count": s.rtk_point_count,
                "rtk_fix_count": s.rtk_fix_count,
                "rtk_float_count": s.rtk_float_count,
                "rtk_coverage": s.rtk_coverage,
                "rtk_alt_min_m": s.rtk_alt_min_m,
                "rtk_alt_max_m": s.rtk_alt_max_m,
                "rtk_alt_span_m": s.rtk_alt_span_m,
                "rtk_alt_max_step_m": s.rtk_alt_max_step_m,
                "rtk_alt_p95_step_m": s.rtk_alt_p95_step_m,
                "rtk_alt_jump_count_gt_2m": s.rtk_alt_jump_count_gt_2m,
                "ubx_types": ";".join(f"{k}:{v}" for k, v in sorted(s.ubx_type_counts.items())),
            }
            for s in summaries
        ]
    ).sort_values("run_name")

    rtk_df = pd.DataFrame(all_rtk_records).sort_values(["run_name", "rtc_seconds"])
    repeatability_df = compute_pairwise_repeatability(rtk_df)

    build_raw_format_table().to_csv(output_dir / "raw_format_summary.csv", index=False)
    summary_df.to_csv(output_dir / "run_summary.csv", index=False)
    rtk_df.to_csv(output_dir / "rtk_points.csv", index=False)
    repeatability_df.to_csv(output_dir / "repeatability_pairwise.csv", index=False)
    plot_repeatability_map(rtk_df, output_dir / "rtk_repeatability_map.png")
    plot_altitude_panels(rtk_df, summary_df, output_dir / "rtk_altitude_panels.png")
    write_report(output_dir, summary_df, repeatability_df)

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
