from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from pyubx2 import UBXReader
from pyubx2.exceptions import UBXParseError, UBXTypeError


@dataclass
class ConversionSummary:
    run_name: str
    source_csv: Path
    gnss_txt: Path
    rtk_txt: Path
    nav_pvt_count: int
    nav_cov_count: int
    gnss_rows: int
    rtk_fix_rows: int
    rtk_float_rows: int
    first_time_s: float | None
    last_time_s: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Railway-Precise-Localization GNSS/RTK data into CT-FGO text format.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\Code\dataset\Railway-Precise-Localization-data\data"),
        help="Directory containing raw per-run CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\Code\dataset\Railway-Precise-Localization-data\ct_fgo_gnss"),
        help="Directory where GNSS_low.txt and RTK_truth.txt will be written.",
    )
    return parser.parse_args()


def parse_rtc_seconds(text: str) -> float | None:
    try:
        hh, mm, sec = text.split(":")
        ss, micros = sec.split(".")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(micros) / 1_000_000.0
    except ValueError:
        return None


def convert_one_run(source_csv: Path, output_dir: Path) -> ConversionSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    gnss_txt = output_dir / "GNSS_low.txt"
    rtk_txt = output_dir / "RTK_truth.txt"

    nav_pvt_count = 0
    nav_cov_count = 0
    gnss_rows = 0
    rtk_fix_rows = 0
    rtk_float_rows = 0
    first_time_s = None
    last_time_s = None

    with source_csv.open("r", encoding="utf-8-sig", newline="") as src, \
        gnss_txt.open("w", encoding="utf-8", newline="") as gnss_dst, \
        rtk_txt.open("w", encoding="utf-8", newline="") as rtk_dst:

        reader = csv.reader(src)
        gnss_writer = csv.writer(gnss_dst, delimiter=" ")
        rtk_writer = csv.writer(rtk_dst, delimiter=" ")

        for row in reader:
            if len(row) < 3 or row[1] != "U":
                continue

            time_s = parse_rtc_seconds(row[0])
            if time_s is None:
                continue

            try:
                msg = UBXReader.parse(bytes.fromhex(row[2]))
            except (ValueError, UBXParseError, UBXTypeError):
                continue

            identity = getattr(msg, "identity", "")
            if identity == "NAV-COV":
                nav_cov_count += 1
                continue
            if identity != "NAV-PVT":
                continue

            nav_pvt_count += 1
            fix_ok = bool(getattr(msg, "gnssFixOk", False))
            invalid_llh = bool(getattr(msg, "invalidLlh", False))
            if not fix_ok or invalid_llh:
                continue

            lat_deg = float(getattr(msg, "lat"))
            lon_deg = float(getattr(msg, "lon"))
            h_m = float(getattr(msg, "hMSL")) / 1000.0
            hacc_m = float(getattr(msg, "hAcc")) / 1000.0
            vacc_m = float(getattr(msg, "vAcc")) / 1000.0
            carr_soln = int(getattr(msg, "carrSoln", 0))

            gnss_writer.writerow(
                [
                    f"{time_s:.6f}",
                    f"{lat_deg:.10f}",
                    f"{lon_deg:.10f}",
                    f"{h_m:.4f}",
                    f"{hacc_m:.4f}",
                    f"{hacc_m:.4f}",
                    f"{vacc_m:.4f}",
                ]
            )
            gnss_rows += 1
            if first_time_s is None:
                first_time_s = time_s
            last_time_s = time_s

            if carr_soln == 1:
                rtk_float_rows += 1
            elif carr_soln == 2:
                rtk_fix_rows += 1
                rtk_writer.writerow(
                    [
                        f"{time_s:.6f}",
                        f"{lat_deg:.10f}",
                        f"{lon_deg:.10f}",
                        f"{h_m:.4f}",
                    ]
                )

    return ConversionSummary(
        run_name=source_csv.parent.name,
        source_csv=source_csv,
        gnss_txt=gnss_txt,
        rtk_txt=rtk_txt,
        nav_pvt_count=nav_pvt_count,
        nav_cov_count=nav_cov_count,
        gnss_rows=gnss_rows,
        rtk_fix_rows=rtk_fix_rows,
        rtk_float_rows=rtk_float_rows,
        first_time_s=first_time_s,
        last_time_s=last_time_s,
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
        output_dir = args.output_root / run_name
        summary = convert_one_run(source_csv, output_dir)
        summaries.append(summary)
        print(
            f"Converted {run_name}: GNSS={summary.gnss_rows}, RTK_fix={summary.rtk_fix_rows}, RTK_float={summary.rtk_float_rows}"
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_name",
                "source_csv",
                "gnss_txt",
                "rtk_txt",
                "nav_pvt_count",
                "nav_cov_count",
                "gnss_rows",
                "rtk_fix_rows",
                "rtk_float_rows",
                "first_time_s",
                "last_time_s",
            ]
        )
        for s in summaries:
            writer.writerow(
                [
                    s.run_name,
                    str(s.source_csv),
                    str(s.gnss_txt),
                    str(s.rtk_txt),
                    s.nav_pvt_count,
                    s.nav_cov_count,
                    s.gnss_rows,
                    s.rtk_fix_rows,
                    s.rtk_float_rows,
                    f"{s.first_time_s:.6f}" if s.first_time_s is not None else "",
                    f"{s.last_time_s:.6f}" if s.last_time_s is not None else "",
                ]
            )

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
