from __future__ import annotations

import argparse
from pathlib import Path

import folium
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one HTML map per Railway run.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\Code\dataset\Railway-Precise-Localization-data\ct_fgo_gnss"),
        help="Root directory that contains per-run folders.",
    )
    parser.add_argument(
        "--file-name",
        default="RTK_truth.txt",
        choices=["RTK_truth.txt", "GNSS_low.txt"],
        help="Which per-run file to load and plot.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output HTML name inside each run folder. Defaults to <file-stem>_map.html",
    )
    return parser.parse_args()


def load_track(track_path: Path) -> pd.DataFrame:
    df = pd.read_csv(track_path, sep=r"\s+", header=None)
    if df.shape[1] < 3:
        raise ValueError(f"{track_path} has fewer than 3 columns")
    df = df.rename(columns={1: "latitude", 2: "longitude"})
    return df.dropna(subset=["latitude", "longitude"])


def build_single_map(track_path: Path, output_path: Path) -> None:
    df = load_track(track_path)
    if df.empty:
        print(f"Skip empty track: {track_path}")
        return

    coords = list(zip(df["latitude"], df["longitude"]))
    fmap = folium.Map(location=coords[0], zoom_start=15, control_scale=True)

    folium.PolyLine(
        locations=coords,
        color="blue",
        weight=2.5,
        opacity=1.0,
        tooltip=track_path.parent.name,
    ).add_to(fmap)

    folium.Marker(
        location=coords[0],
        popup=f"{track_path.parent.name} start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fmap)

    folium.Marker(
        location=coords[-1],
        popup=f"{track_path.parent.name} end",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fmap)

    fmap.fit_bounds(coords)
    fmap.save(str(output_path))
    print(f"Map saved: {output_path}")


def main() -> None:
    args = parse_args()
    output_name = args.output_name or f"{Path(args.file_name).stem}_map.html"

    run_dirs = sorted([p for p in args.root.iterdir() if p.is_dir()])
    for run_dir in run_dirs:
        track_path = run_dir / args.file_name
        if not track_path.exists():
            print(f"Missing file, skip: {track_path}")
            continue
        output_path = run_dir / output_name
        build_single_map(track_path, output_path)


if __name__ == "__main__":
    main()
