from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


G_MPS2 = 9.80665
ACC_SENSITIVITY_MG_PER_LSB = 0.061
GYR_SENSITIVITY_MDPS_PER_LSB = 4.37


@dataclass
class ConversionSummary:
    run_name: str
    source_csv: Path
    output_txt: Path
    output_rows: int
    first_time_s: float | None
    last_time_s: float | None
    duration_s: float | None
    avg_rate_hz: float | None
    skipped_non_imu: int
    skipped_invalid_time: int
    skipped_invalid_payload: int
    skipped_nonpositive_dt: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Railway-Precise-Localization IMU data into CT-FGO text format.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\Code\dataset\Railway-Precise-Localization-data\data"),
        help="Directory containing raw per-run CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\Code\dataset\Railway-Precise-Localization-data\ct_fgo_imu"),
        help="Directory where CT-FGO IMU text files will be written.",
    )
    return parser.parse_args()


def parse_rtc_seconds(text: str) -> float | None:
    try:
        hh, mm, sec = text.split(":")
        ss, micros = sec.split(".")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(micros) / 1_000_000.0
    except ValueError:
        return None


def twos_complement(hex_str: str, bits: int = 16) -> int:
    value = int(hex_str, 16)
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def decode_imu_rates_and_accels(payload_hex: str) -> tuple[float, float, float, float, float, float] | None:
    if len(payload_hex) < 24:
        return None

    # Same byte order as the upstream sample parser.
    gyr_x = twos_complement(payload_hex[2:4] + payload_hex[0:2]) * (GYR_SENSITIVITY_MDPS_PER_LSB / 1000.0) * math.pi / 180.0
    gyr_y = twos_complement(payload_hex[6:8] + payload_hex[4:6]) * (GYR_SENSITIVITY_MDPS_PER_LSB / 1000.0) * math.pi / 180.0
    gyr_z = twos_complement(payload_hex[10:12] + payload_hex[8:10]) * (GYR_SENSITIVITY_MDPS_PER_LSB / 1000.0) * math.pi / 180.0

    acc_x = twos_complement(payload_hex[14:16] + payload_hex[12:14]) * (ACC_SENSITIVITY_MG_PER_LSB / 1000.0) * G_MPS2
    acc_y = twos_complement(payload_hex[18:20] + payload_hex[16:18]) * (ACC_SENSITIVITY_MG_PER_LSB / 1000.0) * G_MPS2
    acc_z = twos_complement(payload_hex[22:24] + payload_hex[20:22]) * (ACC_SENSITIVITY_MG_PER_LSB / 1000.0) * G_MPS2
    return gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z


def convert_one_run(source_csv: Path, output_txt: Path) -> ConversionSummary:
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    output_rows = 0
    first_time_s = None
    last_time_s = None
    prev_time_s = None
    skipped_non_imu = 0
    skipped_invalid_time = 0
    skipped_invalid_payload = 0
    skipped_nonpositive_dt = 0

    with source_csv.open("r", encoding="utf-8-sig", newline="") as src, output_txt.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, delimiter=" ")

        for row in reader:
            if len(row) < 3:
                skipped_invalid_payload += 1
                continue

            if row[1] != "I":
                skipped_non_imu += 1
                continue

            time_s = parse_rtc_seconds(row[0])
            if time_s is None:
                skipped_invalid_time += 1
                continue

            decoded = decode_imu_rates_and_accels(row[2])
            if decoded is None:
                skipped_invalid_payload += 1
                continue

            if prev_time_s is None:
                prev_time_s = time_s
                first_time_s = time_s
                continue

            dt = time_s - prev_time_s
            prev_time_s = time_s
            if dt <= 0:
                skipped_nonpositive_dt += 1
                continue

            gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z = decoded
            dtheta_x = gyr_x * dt
            dtheta_y = gyr_y * dt
            dtheta_z = gyr_z * dt
            dvel_x = acc_x * dt
            dvel_y = acc_y * dt
            dvel_z = acc_z * dt

            writer.writerow(
                [
                    f"{time_s:.6f}",
                    f"{dtheta_x:.12e}",
                    f"{dtheta_y:.12e}",
                    f"{dtheta_z:.12e}",
                    f"{dvel_x:.12e}",
                    f"{dvel_y:.12e}",
                    f"{dvel_z:.12e}",
                ]
            )
            output_rows += 1
            last_time_s = time_s

    duration_s = None
    avg_rate_hz = None
    if first_time_s is not None and last_time_s is not None and last_time_s > first_time_s:
        duration_s = last_time_s - first_time_s
        if output_rows > 0:
            avg_rate_hz = output_rows / duration_s

    return ConversionSummary(
        run_name=source_csv.parent.name,
        source_csv=source_csv,
        output_txt=output_txt,
        output_rows=output_rows,
        first_time_s=first_time_s,
        last_time_s=last_time_s,
        duration_s=duration_s,
        avg_rate_hz=avg_rate_hz,
        skipped_non_imu=skipped_non_imu,
        skipped_invalid_time=skipped_invalid_time,
        skipped_invalid_payload=skipped_invalid_payload,
        skipped_nonpositive_dt=skipped_nonpositive_dt,
    )


def main() -> None:
    args = parse_args()
    csv_files = sorted(args.data_root.glob("*/*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found under {args.data_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"
    summaries: list[ConversionSummary] = []

    for source_csv in csv_files:
        run_name = source_csv.parent.name
        output_txt = args.output_root / run_name / "Body_IMU.txt"
        summary = convert_one_run(source_csv, output_txt)
        summaries.append(summary)
        print(f"Converted {run_name}: rows={summary.output_rows}, rate={summary.avg_rate_hz:.3f} Hz" if summary.avg_rate_hz else f"Converted {run_name}: rows={summary.output_rows}")

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_name",
                "source_csv",
                "output_txt",
                "output_rows",
                "first_time_s",
                "last_time_s",
                "duration_s",
                "avg_rate_hz",
                "skipped_non_imu",
                "skipped_invalid_time",
                "skipped_invalid_payload",
                "skipped_nonpositive_dt",
            ]
        )
        for s in summaries:
            writer.writerow(
                [
                    s.run_name,
                    str(s.source_csv),
                    str(s.output_txt),
                    s.output_rows,
                    f"{s.first_time_s:.6f}" if s.first_time_s is not None else "",
                    f"{s.last_time_s:.6f}" if s.last_time_s is not None else "",
                    f"{s.duration_s:.6f}" if s.duration_s is not None else "",
                    f"{s.avg_rate_hz:.6f}" if s.avg_rate_hz is not None else "",
                    s.skipped_non_imu,
                    s.skipped_invalid_time,
                    s.skipped_invalid_payload,
                    s.skipped_nonpositive_dt,
                ]
            )

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
