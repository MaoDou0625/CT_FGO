from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import folium
import pandas as pd


ROOT = Path(r"D:\Code\dataset\Railway-Precise-Localization-data\ct_fgo_gnss")
OUTPUT_HTML = ROOT / "highlighted_regions_map.html"


def load_tracks(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gnss_rows = []
    rtk_rows = []
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        gnss_path = run_dir / "GNSS_low.txt"
        rtk_path = run_dir / "RTK_truth.txt"
        if gnss_path.exists():
            g = pd.read_csv(gnss_path, sep=r"\s+", header=None, names=["t", "lat", "lon", "h", "sx", "sy", "sz"])
            g["run"] = run_dir.name
            gnss_rows.append(g[["run", "t", "lat", "lon", "h"]])
        if rtk_path.exists():
            r = pd.read_csv(rtk_path, sep=r"\s+", header=None, names=["t", "lat", "lon", "h"])
            r["run"] = run_dir.name
            rtk_rows.append(r[["run", "t", "lat", "lon", "h"]])
    return pd.concat(gnss_rows, ignore_index=True), pd.concat(rtk_rows, ignore_index=True)


def add_local_xy(df: pd.DataFrame, lat_col: str, lon_col: str, ref_lat: float, ref_lon: float) -> pd.DataFrame:
    df = df.copy()
    lat0 = math.radians(ref_lat)
    df["east"] = (df[lon_col] - ref_lon).apply(math.radians) * 6378137.0 * math.cos(lat0)
    df["north"] = (df[lat_col] - ref_lat).apply(math.radians) * 6378137.0
    return df


def find_shared_region(gnss: pd.DataFrame, rtk: pd.DataFrame, ref_lat: float, ref_lon: float, cell_m: float = 300.0) -> dict:
    gnss_xy = add_local_xy(gnss, "lat", "lon", ref_lat, ref_lon)
    rtk_xy = add_local_xy(rtk, "lat", "lon", ref_lat, ref_lon)

    cells = defaultdict(lambda: {"gnss": set(), "rtk": set()})
    for _, row in gnss_xy.iterrows():
        key = (int(row["east"] // cell_m), int(row["north"] // cell_m))
        cells[key]["gnss"].add(row["run"])
    for _, row in rtk_xy.iterrows():
        key = (int(row["east"] // cell_m), int(row["north"] // cell_m))
        cells[key]["rtk"].add(row["run"])

    best_key = None
    best_intersection = set()
    best_score = (-1, -1)
    for key, value in cells.items():
        inter = value["gnss"] & value["rtk"]
        score = (len(inter), len(value["rtk"]))
        if score > best_score:
            best_key = key
            best_intersection = inter
            best_score = score

    east = (best_key[0] + 0.5) * cell_m
    north = (best_key[1] + 0.5) * cell_m
    lat = ref_lat + math.degrees(north / 6378137.0)
    lon = ref_lon + math.degrees(east / (6378137.0 * math.cos(math.radians(ref_lat))))
    return {
        "label": "Shared GNSS + RTK region",
        "lat": lat,
        "lon": lon,
        "radius_m": cell_m / 2,
        "runs": sorted(best_intersection),
        "color": "green",
    }


def find_gnss_anomaly_region(gnss: pd.DataFrame, ref_lat: float, ref_lon: float, cell_m: float = 300.0) -> dict:
    anomaly_rows = []
    for run, g in gnss.groupby("run"):
        g = g.sort_values("t").reset_index(drop=True)
        if len(g) < 2:
            continue
        dt = g["t"].diff()
        lat1 = g["lat"].shift(1)
        lon1 = g["lon"].shift(1)
        lat2 = g["lat"]
        lon2 = g["lon"]
        r = 6371000.0
        x = (lon2 - lon1).apply(math.radians) * ((lat1 + lat2) / 2).apply(math.radians).apply(math.cos) * r
        y = (lat2 - lat1).apply(math.radians) * r
        dist = (x**2 + y**2) ** 0.5
        bad = ((dt > 2.5) | (dist > 80)).fillna(False)
        if not bad.any():
            continue
        ev = g[bad].copy()
        prev = g.shift(1).loc[bad]
        ev["lat_mid"] = (ev["lat"].to_numpy() + prev["lat"].to_numpy()) / 2
        ev["lon_mid"] = (ev["lon"].to_numpy() + prev["lon"].to_numpy()) / 2
        ev["run"] = run
        anomaly_rows.append(ev[["run", "lat_mid", "lon_mid"]])

    anomalies = pd.concat(anomaly_rows, ignore_index=True)
    anomalies = add_local_xy(anomalies, "lat_mid", "lon_mid", ref_lat, ref_lon)

    cells = defaultdict(set)
    for _, row in anomalies.iterrows():
        key = (int(row["east"] // cell_m), int(row["north"] // cell_m))
        cells[key].add(row["run"])

    best_key, best_runs = max(cells.items(), key=lambda item: len(item[1]))
    east = (best_key[0] + 0.5) * cell_m
    north = (best_key[1] + 0.5) * cell_m
    lat = ref_lat + math.degrees(north / 6378137.0)
    lon = ref_lon + math.degrees(east / (6378137.0 * math.cos(math.radians(ref_lat))))
    return {
        "label": "GNSS anomaly hotspot",
        "lat": lat,
        "lon": lon,
        "radius_m": cell_m / 2,
        "runs": sorted(best_runs),
        "color": "red",
    }


def main() -> None:
    gnss, rtk = load_tracks(ROOT)
    ref_lat = pd.concat([gnss["lat"], rtk["lat"]]).mean()
    ref_lon = pd.concat([gnss["lon"], rtk["lon"]]).mean()

    shared = find_shared_region(gnss, rtk, ref_lat, ref_lon, cell_m=300.0)
    anomaly = find_gnss_anomaly_region(gnss, ref_lat, ref_lon, cell_m=300.0)

    fmap = folium.Map(location=[ref_lat, ref_lon], zoom_start=13, control_scale=True)

    gnss_group = folium.FeatureGroup(name="GNSS_low", show=True)
    rtk_group = folium.FeatureGroup(name="RTK_truth", show=True)

    for run, g in gnss.groupby("run"):
        coords = list(zip(g["lat"], g["lon"]))
        if coords:
            folium.PolyLine(coords, color="#808080", weight=1.5, opacity=0.45, tooltip=f"GNSS {run}").add_to(gnss_group)

    for run, r in rtk.groupby("run"):
        coords = list(zip(r["lat"], r["lon"]))
        if coords:
            folium.PolyLine(coords, color="#0066cc", weight=2.0, opacity=0.7, tooltip=f"RTK {run}").add_to(rtk_group)

    gnss_group.add_to(fmap)
    rtk_group.add_to(fmap)

    for region in [shared, anomaly]:
        popup = "<br>".join([region["label"], f"Runs: {len(region['runs'])}"] + region["runs"])
        folium.Circle(
            location=[region["lat"], region["lon"]],
            radius=region["radius_m"],
            color=region["color"],
            weight=3,
            fill=True,
            fill_opacity=0.12,
            popup=popup,
            tooltip=region["label"],
        ).add_to(fmap)
        folium.Marker(
            location=[region["lat"], region["lon"]],
            popup=popup,
            icon=folium.Icon(color=region["color"], icon="info-sign"),
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(str(OUTPUT_HTML))
    print(f"Saved highlighted map: {OUTPUT_HTML}")
    print("Shared region:", shared)
    print("Anomaly region:", anomaly)


if __name__ == "__main__":
    main()
