from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import folium
import numpy as np
import pandas as pd


ROOT = Path(r"D:\Code\dataset\Railway-Precise-Localization-data\ct_fgo_gnss")
OUTPUT_HTML = ROOT / "highlighted_segments_map.html"
BIN_M = 250.0


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


def add_track_coordinate(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    xy = df[["east", "north"]].to_numpy()
    centered = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    if axis[0] < 0:
        axis = -axis
    out = df.copy()
    out["s"] = xy.dot(axis)
    return out, axis


def compute_gnss_anomalies(gnss: pd.DataFrame) -> pd.DataFrame:
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
        anomaly_rows.append(ev[["run", "t", "lat_mid", "lon_mid"]])
    return pd.concat(anomaly_rows, ignore_index=True)


def select_top_bins(items: dict[int, tuple], limit: int) -> list[tuple]:
    chosen = []
    used_bins = []
    for bin_id, payload in sorted(items.items(), key=lambda item: item[1][0], reverse=True):
        if any(abs(bin_id - other) <= 1 for other in used_bins):
            continue
        chosen.append((bin_id, *payload))
        used_bins.append(bin_id)
        if len(chosen) >= limit:
            break
    return chosen


def shared_segment_from_bin(bin_id: int, gnss: pd.DataFrame, rtk: pd.DataFrame, runs: list[str], label: str, color: str) -> dict:
    s0 = bin_id * BIN_M
    s1 = (bin_id + 1) * BIN_M
    hotspot = rtk[(rtk["s"] >= s0) & (rtk["s"] < s1)]
    if hotspot.empty:
        raise ValueError(f"No hotspot points found for bin {bin_id}")
    center_e = hotspot["east"].mean()
    center_n = hotspot["north"].mean()
    seg = gnss[((gnss["east"] - center_e) ** 2 + (gnss["north"] - center_n) ** 2) <= 150.0**2].sort_values("s")
    if seg.empty:
        seg = gnss[(gnss["s"] >= s0) & (gnss["s"] < s1)].sort_values("s")
    if seg.empty:
        raise ValueError(f"No GNSS points found for bin {bin_id}")
    start = seg.iloc[0]
    end = seg.iloc[-1]
    coords = list(zip(seg["lat"], seg["lon"]))
    return {
        "label": label,
        "bin_id": bin_id,
        "color": color,
        "runs": runs,
        "start": (float(start["lat"]), float(start["lon"])),
        "end": (float(end["lat"]), float(end["lon"])),
        "coords": coords,
    }


def anomaly_segment_from_bin(bin_id: int, anomalies: pd.DataFrame, runs: list[str], label: str, color: str) -> dict:
    s0 = bin_id * BIN_M
    s1 = (bin_id + 1) * BIN_M
    seg = anomalies[(anomalies["s"] >= s0) & (anomalies["s"] < s1)].sort_values("s")
    if seg.empty:
        raise ValueError(f"No anomaly points found for bin {bin_id}")
    start = seg.iloc[0]
    end = seg.iloc[-1]
    coords = list(zip(seg["lat_mid"], seg["lon_mid"]))
    if len(coords) == 1:
        coords = coords * 2
    return {
        "label": label,
        "bin_id": bin_id,
        "color": color,
        "runs": runs,
        "start": (float(start["lat_mid"]), float(start["lon_mid"])),
        "end": (float(end["lat_mid"]), float(end["lon_mid"])),
        "coords": coords,
    }


def main() -> None:
    gnss, rtk = load_tracks(ROOT)
    ref_lat = pd.concat([gnss["lat"], rtk["lat"]]).mean()
    ref_lon = pd.concat([gnss["lon"], rtk["lon"]]).mean()

    gnss = add_local_xy(gnss, "lat", "lon", ref_lat, ref_lon)
    rtk = add_local_xy(rtk, "lat", "lon", ref_lat, ref_lon)
    gnss, axis = add_track_coordinate(gnss)
    rtk = rtk.copy()
    rtk["s"] = rtk[["east", "north"]].to_numpy().dot(axis)

    anomalies = compute_gnss_anomalies(gnss)
    anomalies = add_local_xy(anomalies, "lat_mid", "lon_mid", ref_lat, ref_lon)
    anomalies["s"] = anomalies[["east", "north"]].to_numpy().dot(axis)

    shared_bins = {}
    s_min = int(math.floor(min(gnss["s"].min(), rtk["s"].min()) / BIN_M)) - 1
    s_max = int(math.ceil(max(gnss["s"].max(), rtk["s"].max()) / BIN_M)) + 1
    for bin_id in range(s_min, s_max + 1):
        s0 = bin_id * BIN_M
        s1 = (bin_id + 1) * BIN_M
        g_runs = set(gnss[(gnss["s"] >= s0) & (gnss["s"] < s1)]["run"])
        r_runs = set(rtk[(rtk["s"] >= s0) & (rtk["s"] < s1)]["run"])
        inter = g_runs & r_runs
        if inter:
            shared_bins[bin_id] = (len(inter), sorted(inter))

    anomaly_bins = {}
    for bin_id in range(int(math.floor(anomalies["s"].min() / BIN_M)) - 1, int(math.ceil(anomalies["s"].max() / BIN_M)) + 1):
        s0 = bin_id * BIN_M
        s1 = (bin_id + 1) * BIN_M
        runs = set(anomalies[(anomalies["s"] >= s0) & (anomalies["s"] < s1)]["run"])
        if runs:
            anomaly_bins[bin_id] = (len(runs), sorted(runs))

    shared_top = select_top_bins(shared_bins, limit=3)
    anomaly_top = select_top_bins(anomaly_bins, limit=3)

    segments = []
    for idx, (bin_id, count, runs) in enumerate(shared_top, start=1):
        segments.append(shared_segment_from_bin(bin_id, gnss, rtk, runs, f"Shared Segment {idx} ({count} runs)", "#1f9d55"))
    for idx, (bin_id, count, runs) in enumerate(anomaly_top, start=1):
        segments.append(anomaly_segment_from_bin(bin_id, anomalies, runs, f"Anomaly Segment {idx} ({count} runs)", "#d11a2a"))

    fmap = folium.Map(location=[ref_lat, ref_lon], zoom_start=13, control_scale=True)
    base_gnss = folium.FeatureGroup(name="GNSS_low", show=True)
    base_rtk = folium.FeatureGroup(name="RTK_truth", show=True)
    highlight_group = folium.FeatureGroup(name="Selected segments", show=True)

    for run, g in gnss.groupby("run"):
        folium.PolyLine(list(zip(g["lat"], g["lon"])), color="#808080", weight=1.2, opacity=0.35, tooltip=f"GNSS {run}").add_to(base_gnss)
    for run, r in rtk.groupby("run"):
        folium.PolyLine(list(zip(r["lat"], r["lon"])), color="#0066cc", weight=1.8, opacity=0.55, tooltip=f"RTK {run}").add_to(base_rtk)

    for seg in segments:
        popup = "<br>".join([seg["label"], f"Runs: {len(seg['runs'])}"] + seg["runs"])
        folium.PolyLine(seg["coords"], color=seg["color"], weight=5, opacity=0.95, tooltip=seg["label"]).add_to(highlight_group)
        folium.Marker(
            location=seg["start"],
            popup=f"START<br>{popup}",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(highlight_group)
        folium.Marker(
            location=seg["end"],
            popup=f"END<br>{popup}",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(highlight_group)

    base_gnss.add_to(fmap)
    base_rtk.add_to(fmap)
    highlight_group.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(str(OUTPUT_HTML))

    print(f"Saved highlighted segment map: {OUTPUT_HTML}")
    for seg in segments:
        print(seg["label"], "start", seg["start"], "end", seg["end"], "runs", len(seg["runs"]))


if __name__ == "__main__":
    main()
