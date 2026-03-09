from __future__ import annotations

import argparse
from pathlib import Path

import folium
import pandas as pd


COLORS = [
    "blue",
    "red",
    "green",
    "purple",
    "orange",
    "darkred",
    "lightred",
    "beige",
    "darkblue",
    "darkgreen",
    "cadetblue",
    "darkpurple",
    "white",
    "pink",
    "lightblue",
    "lightgreen",
    "gray",
    "black",
    "lightgray",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all Railway-Precise-Localization runs on one Folium map.")
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
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <root>/all_runs_<file-stem>_map.html",
    )
    return parser.parse_args()


def load_track(track_path: Path) -> pd.DataFrame:
    df = pd.read_csv(track_path, sep=r"\s+", header=None)
    if df.shape[1] < 3:
        raise ValueError(f"{track_path} has fewer than 3 columns")
    df = df.rename(columns={1: "latitude", 2: "longitude"})
    df = df.dropna(subset=["latitude", "longitude"])
    return df


def build_map(root: Path, file_name: str, output_path: Path) -> None:
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    tracks: list[tuple[str, pd.DataFrame]] = []
    bounds: list[tuple[float, float]] = []

    for run_dir in run_dirs:
        track_path = run_dir / file_name
        if not track_path.exists():
            continue
        df = load_track(track_path)
        if df.empty:
            continue
        tracks.append((run_dir.name, df))
        bounds.extend(list(zip(df["latitude"], df["longitude"])))

    if not tracks:
        raise SystemExit(f"No valid {file_name} tracks found under {root}")

    start_lat = tracks[0][1]["latitude"].iloc[0]
    start_lon = tracks[0][1]["longitude"].iloc[0]
    fmap = folium.Map(location=[start_lat, start_lon], zoom_start=13, control_scale=True)

    for idx, (run_name, df) in enumerate(tracks):
        color = COLORS[idx % len(COLORS)]
        coords = list(zip(df["latitude"], df["longitude"]))
        group = folium.FeatureGroup(name=run_name, show=True)

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=2.5,
            opacity=0.9,
            tooltip=run_name,
        ).add_to(group)

        folium.CircleMarker(
            location=coords[0],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=1.0,
            popup=f"{run_name} start",
        ).add_to(group)

        folium.CircleMarker(
            location=coords[-1],
            radius=4,
            color=color,
            fill=False,
            popup=f"{run_name} end",
        ).add_to(group)

        group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds(bounds)
    fmap.save(str(output_path))
    print(f"Map successfully saved to: {output_path}")


def main() -> None:
    args = parse_args()
    output_path = args.output or (args.root / f"all_runs_{Path(args.file_name).stem}_map.html")
    build_map(args.root, args.file_name, output_path)


if __name__ == "__main__":
    main()
