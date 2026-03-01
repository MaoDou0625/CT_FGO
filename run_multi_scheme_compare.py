import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_ROOT = Path(r"D:\Code\dataset\WID\Datasets\transformedData2")
DEFAULT_TRUTH = DEFAULT_DATA_ROOT / "four_wheel_dataset_PassengerCar" / "trial01" / "GNSS_use.txt"
DEFAULT_OUT = DEFAULT_DATA_ROOT / "five_scheme_eval_20260301"
DEFAULT_TRUTH_ROOT = DEFAULT_DATA_ROOT / "four_wheel_dataset_PassengerCar"


@dataclass
class Scheme:
    run_name: str
    result_path: Path
    truth_path: Path
    status: str = "DONE"


def load_table(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2:
        raise ValueError(f"Invalid table shape for {path}")
    return data


def convert_kf_gins_nav_to_ct_like(src_nav: Path, dst_txt: Path) -> None:
    raw = load_table(src_nav)
    if raw.shape[1] < 11:
        raise ValueError(f"KF-GINS nav format unexpected: {src_nav}, columns={raw.shape[1]}")
    out = np.column_stack(
        [
            raw[:, 1],   # time
            raw[:, 2],   # lat
            raw[:, 3],   # lon
            raw[:, 4],   # alt
            raw[:, 5],   # vn
            raw[:, 6],   # ve
            raw[:, 7],   # vd
            raw[:, 8],   # roll
            raw[:, 9],   # pitch
            raw[:, 10],  # yaw
        ]
    )
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst_txt, out, fmt="%.9f")


def convert_wheel_gins_traj_to_ct_like(src_traj: Path, dst_txt: Path) -> None:
    raw = load_table(src_traj)
    if raw.shape[1] < 10:
        raise ValueError(f"Wheel-GINS traj format unexpected: {src_traj}, columns={raw.shape[1]}")
    out = raw[:, :10]
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst_txt, out, fmt="%.9f")


def write_manifest(path: Path, schemes: List[Scheme]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "config_path",
                "result_path",
                "truth_path",
                "log_path",
                "status",
                "runtime_sec",
                "return_code",
            ],
        )
        w.writeheader()
        for s in schemes:
            w.writerow(
                {
                    "run_name": s.run_name,
                    "config_path": "",
                    "result_path": str(s.result_path),
                    "truth_path": str(s.truth_path),
                    "log_path": "",
                    "status": s.status,
                    "runtime_sec": "",
                    "return_code": "0",
                }
            )


def run_unified_evaluate(manifest: Path, out_dir: Path, baseline: str, segments: str) -> None:
    cmd = [
        sys.executable,
        "experiment_60_80.py",
        "evaluate",
        "--manifest",
        str(manifest),
        "--out-dir",
        str(out_dir),
        "--baseline-run",
        baseline,
        "--segments",
        segments,
        "--target-segment",
        "60-80",
        "--min-improve-pct",
        "15",
        "--max-non-target-degrade-pct",
        "5",
        "--max-speed-degrade-pct",
        "5",
    ]
    subprocess.run(cmd, check=True)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def render_charts(summary_rows: List[dict], out_dir: Path) -> None:
    names = [r["run_name"] for r in summary_rows]
    pos_60_80 = [float(r["pos_rmse_60_80_m"]) for r in summary_rows]
    speed_60_80 = [float(r["speed_rmse_60_80_mps"]) for r in summary_rows]
    pos_global = [float(r["pos_rmse_global_w_m"]) for r in summary_rows]
    speed_global = [float(r["speed_rmse_global_w_mps"]) for r in summary_rows]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, pos_60_80, width, label="Pos2D_RMSE_60_80 (m)")
    ax.bar(x + width / 2, speed_60_80, width, label="Speed2D_RMSE_60_80 (m/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_title("Multi-Scheme 60-80 Segment Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_60_80.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(names, pos_global, marker="o", label="Pos2D_RMSE_GlobalWeighted (m)")
    ax.plot(names, speed_global, marker="s", label="Speed2D_RMSE_GlobalWeighted (m/s)")
    ax.set_title("Multi-Scheme Global Weighted Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_global_weighted.png", dpi=150)
    plt.close(fig)


def latlonalt_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray, lat0_deg: float, lon0_deg: float, alt0_m: float):
    r_earth = 6378137.0
    dlat = np.deg2rad(lat_deg - lat0_deg)
    dlon = np.deg2rad(lon_deg - lon0_deg)
    north = dlat * r_earth
    east = dlon * r_earth * np.cos(np.deg2rad(lat0_deg))
    up = alt_m - alt0_m
    return east, north, up


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse_trials_arg(trials_arg: str, truth_root: Path) -> List[str]:
    if trials_arg.strip().lower() == "all":
        trials = sorted(p.name for p in truth_root.glob("trial*") if p.is_dir())
        if not trials:
            raise RuntimeError(f"No trial folders found under: {truth_root}")
        return trials
    trials = [x.strip() for x in trials_arg.split(",") if x.strip()]
    if not trials:
        raise ValueError("--trials is empty")
    return trials


def format_template(path_tpl: str, trial: str) -> Path:
    return Path(path_tpl.format(trial=trial))


def pick_first_existing(candidates: List[Path]) -> Path:
    for c in candidates:
        if c.exists():
            return c
    joined = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No valid input found. Tried:\n{joined}")


def resolve_default_scheme_path(data_root: Path, scheme: str, trial: str) -> Path:
    trial_suffix = trial.replace("trial", "")
    candidates = {
        "A": [
            data_root / "output_schemeA_main_allwheel_20260228" / trial / "ct_trajectory.txt",
            data_root / "output_schemeA_main_allwheel_20260228" / f"ct_trajectory_{trial}.txt",
            data_root / "output_schemeA_main_allwheel_20260228" / f"ct_trajectory_trial{trial_suffix}.txt",
            data_root / "output_schemeA_main_allwheel_20260228" / "ct_trajectory.txt",
        ],
        "B": [
            data_root / "output_subtask3_schemeB_main_plus_rear_right" / trial / "ct_trajectory.txt",
            data_root / "output_subtask3_schemeB_main_plus_rear_right" / f"ct_trajectory_{trial}.txt",
            data_root / "output_subtask3_schemeB_main_plus_rear_right" / f"ct_trajectory_trial{trial_suffix}.txt",
            data_root / "output_subtask3_schemeB_main_plus_rear_right" / "ct_trajectory.txt",
        ],
        "C": [
            data_root / "output_scheme_c_front_left_20260228" / trial / "ct_trajectory.txt",
            data_root / "output_scheme_c_front_left_20260228" / f"ct_trajectory_{trial}.txt",
            data_root / "output_scheme_c_front_left_20260228" / f"ct_trajectory_trial{trial_suffix}.txt",
            data_root / "output_scheme_c_front_left_20260228" / "ct_trajectory.txt",
        ],
        "F": [
            data_root / "output_schemeF_main_only_20260301" / trial / "ct_trajectory.txt",
            data_root / "output_schemeF_main_only_20260301" / f"ct_trajectory_{trial}.txt",
            data_root / "output_schemeF_main_only_20260301" / f"ct_trajectory_trial{trial_suffix}.txt",
            data_root / "output_schemeF_main_only_20260301" / "ct_trajectory.txt",
        ],
        "D": [
            data_root / f"kf_gins_D_mainimu_{trial}_direct_20260228_01" / "KF_GINS_Navresult.nav",
            data_root / f"kf_gins_D_mainimu_{trial}_20260228_01" / "KF_GINS_Navresult.nav",
        ],
        "E": [
            data_root / f"wheel_gins_single_rear2_{trial}_20260228_2145" / "traj.txt",
            data_root / f"wheel_gins_schemeE_{trial}_20260228_2128" / "traj.txt",
            data_root / "wheel_gins_single_rear2_20260228_2145" / "traj.txt",
            data_root / "wheel_gins_schemeE_trial01_20260228_2128" / "traj.txt",
        ],
    }
    return pick_first_existing(candidates[scheme])


def resolve_input(
    data_root: Path,
    scheme: str,
    trial: str,
    template: str,
) -> Path:
    if template:
        path = format_template(template, trial)
        if not path.exists():
            raise FileNotFoundError(f"Missing file for scheme {scheme}, trial {trial}: {path}")
        return path
    return resolve_default_scheme_path(data_root, scheme, trial)


def render_detailed_trajectory_plots(manifest_rows: List[dict], out_dir: Path) -> None:
    rows_by_trial = defaultdict(list)
    truth_by_trial = {}
    for row in manifest_rows:
        truth_path = Path(row["truth_path"])
        trial = truth_path.parent.name
        rows_by_trial[trial].append(row)
        truth_by_trial[trial] = truth_path

    for trial, rows in sorted(rows_by_trial.items()):
        render_trial_detailed_plots(rows, truth_by_trial[trial], out_dir / "detailed_plots" / trial)


def render_trial_detailed_plots(manifest_rows: List[dict], truth_path: Path, detailed_dir: Path) -> None:
    truth = load_table(truth_path)
    if truth.shape[1] < 4:
        raise ValueError(f"Truth format unexpected: {truth_path}")

    t_truth = truth[:, 0]
    lat_truth = truth[:, 1]
    lon_truth = truth[:, 2]
    alt_truth = truth[:, 3]

    lat0 = float(lat_truth[0])
    lon0 = float(lon_truth[0])
    alt0 = float(alt_truth[0])
    truth_e_all, truth_n_all, truth_u_all = latlonalt_to_enu(lat_truth, lon_truth, alt_truth, lat0, lon0, alt0)

    per_scheme_dir = detailed_dir / "per_scheme"
    detailed_dir.mkdir(parents=True, exist_ok=True)
    per_scheme_dir.mkdir(parents=True, exist_ok=True)

    fig_2d_all, ax_2d_all = plt.subplots(figsize=(8, 6))
    ax_2d_all.plot(truth_e_all, truth_n_all, "k-", linewidth=2.0, label="truth")
    ax_2d_all.set_title(f"All Schemes 2D Trajectory vs Truth ({truth_path.parent.name})")
    ax_2d_all.set_xlabel("East (m)")
    ax_2d_all.set_ylabel("North (m)")
    ax_2d_all.grid(True, alpha=0.3)

    fig_enu_all, axs_enu_all = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t_truth_rel = t_truth - t_truth[0]
    axs_enu_all[0].plot(t_truth_rel, truth_e_all, "k-", linewidth=2.0, label="truth")
    axs_enu_all[1].plot(t_truth_rel, truth_n_all, "k-", linewidth=2.0, label="truth")
    axs_enu_all[2].plot(t_truth_rel, truth_u_all, "k-", linewidth=2.0, label="truth")
    axs_enu_all[0].set_ylabel("E (m)")
    axs_enu_all[1].set_ylabel("N (m)")
    axs_enu_all[2].set_ylabel("U (m)")
    axs_enu_all[2].set_xlabel("Time (s)")
    axs_enu_all[0].set_title(f"All Schemes ENU Displacement vs Truth ({truth_path.parent.name})")
    for ax in axs_enu_all:
        ax.grid(True, alpha=0.3)

    for row in manifest_rows:
        run_name = row["run_name"]
        result_path = Path(row["result_path"])
        if not result_path.exists():
            continue
        res = load_table(result_path)
        if res.shape[1] < 4:
            continue

        t_res = res[:, 0]
        lat_res = res[:, 1]
        lon_res = res[:, 2]
        alt_res = res[:, 3]

        valid = (t_res >= t_truth[0]) & (t_res <= t_truth[-1])
        if valid.sum() < 2:
            continue

        t = t_res[valid]
        lat_r = lat_res[valid]
        lon_r = lon_res[valid]
        alt_r = alt_res[valid]

        lat_t = np.interp(t, t_truth, lat_truth)
        lon_t = np.interp(t, t_truth, lon_truth)
        alt_t = np.interp(t, t_truth, alt_truth)

        lat_ref = float(lat_t[0])
        lon_ref = float(lon_t[0])
        alt_ref = float(alt_t[0])
        e_r, n_r, u_r = latlonalt_to_enu(lat_r, lon_r, alt_r, lat_ref, lon_ref, alt_ref)
        e_t, n_t, u_t = latlonalt_to_enu(lat_t, lon_t, alt_t, lat_ref, lon_ref, alt_ref)
        t_rel = t - t[0]

        fig2d, ax2d = plt.subplots(figsize=(8, 6))
        ax2d.plot(e_t, n_t, "k-", linewidth=2.0, label="truth")
        ax2d.plot(e_r, n_r, "-", linewidth=1.5, label=run_name)
        ax2d.set_title(f"2D Trajectory vs Truth: {run_name}")
        ax2d.set_xlabel("East (m)")
        ax2d.set_ylabel("North (m)")
        ax2d.grid(True, alpha=0.3)
        ax2d.legend()
        fig2d.tight_layout()
        fig2d.savefig(per_scheme_dir / f"{safe_name(run_name)}_2d_vs_truth.png", dpi=150)
        plt.close(fig2d)

        figenu, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(t_rel, e_t, "k-", linewidth=2.0, label="truth")
        axs[0].plot(t_rel, e_r, "-", linewidth=1.4, label=run_name)
        axs[1].plot(t_rel, n_t, "k-", linewidth=2.0, label="truth")
        axs[1].plot(t_rel, n_r, "-", linewidth=1.4, label=run_name)
        axs[2].plot(t_rel, u_t, "k-", linewidth=2.0, label="truth")
        axs[2].plot(t_rel, u_r, "-", linewidth=1.4, label=run_name)
        axs[0].set_ylabel("E (m)")
        axs[1].set_ylabel("N (m)")
        axs[2].set_ylabel("U (m)")
        axs[2].set_xlabel("Time (s)")
        axs[0].set_title(f"ENU Displacement vs Truth: {run_name}")
        for ax in axs:
            ax.grid(True, alpha=0.3)
            ax.legend()
        figenu.tight_layout()
        figenu.savefig(per_scheme_dir / f"{safe_name(run_name)}_enu_vs_truth.png", dpi=150)
        plt.close(figenu)

        e_r_all, n_r_all, u_r_all = latlonalt_to_enu(lat_res, lon_res, alt_res, lat0, lon0, alt0)
        ax_2d_all.plot(e_r_all, n_r_all, linewidth=1.2, label=run_name)
        t_rel_all = t_res - t_truth[0]
        axs_enu_all[0].plot(t_rel_all, e_r_all, linewidth=1.0, label=run_name)
        axs_enu_all[1].plot(t_rel_all, n_r_all, linewidth=1.0, label=run_name)
        axs_enu_all[2].plot(t_rel_all, u_r_all, linewidth=1.0, label=run_name)

    ax_2d_all.legend()
    fig_2d_all.tight_layout()
    fig_2d_all.savefig(detailed_dir / "all_schemes_2d_vs_truth.png", dpi=150)
    plt.close(fig_2d_all)

    axs_enu_all[0].legend()
    fig_enu_all.tight_layout()
    fig_enu_all.savefig(detailed_dir / "all_schemes_enu_vs_truth.png", dpi=150)
    plt.close(fig_enu_all)


def write_final_report(
    out_dir: Path,
    manifest_path: Path,
    summary_rows: List[dict],
    gate_rows: List[dict],
    kf_converted: Path,
    wheel_converted: Path,
) -> None:
    gate_map = {r["run_name"]: r for r in gate_rows}
    lines = [
        "# Multi-Scheme Unified Comparison Report",
        "",
        "## Inputs",
        f"- Manifest: `{manifest_path}`",
        f"- Converted KF-GINS result: `{kf_converted}`",
        f"- Converted Wheel-GINS result: `{wheel_converted}`",
        "",
        "## Results",
    ]
    for s in summary_rows:
        g = gate_map.get(s["run_name"], {})
        lines.append(
            "- "
            + f"{s['run_name']}: pos60-80={float(s['pos_rmse_60_80_m']):.4f} m, "
            + f"speed60-80={float(s['speed_rmse_60_80_mps']):.4f} m/s, "
            + f"posGlobalW={float(s['pos_rmse_global_w_m']):.4f} m, "
            + f"overall_pass={g.get('overall_pass', 'NA')}"
        )

    lines.extend(
        [
            "",
            "## Files",
            "- `summary_metrics.csv`, `gate_results.csv`, `report.md` from unified evaluator",
            "- `compare_60_80.png`, `compare_global_weighted.png`",
            "- `detailed_plots/all_schemes_2d_vs_truth.png`",
            "- `detailed_plots/all_schemes_enu_vs_truth.png`",
            "- `detailed_plots/per_scheme/*_2d_vs_truth.png`",
            "- `detailed_plots/per_scheme/*_enu_vs_truth.png`",
            "",
            "## Notes",
            "- Unified segments: `0-20,20-40,40-60,60-80,80-90`",
            "- Baseline is applied per trial: `schemeA_ct_main_allwheel_{trial}`",
        ]
    )
    (out_dir / "FINAL_REPORT_5SCHEMES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_command_list(out_dir: Path, manifest_path: Path) -> None:
    txt = f"""# Commands For Reproduction

## 1) Build and run unified multi-scheme comparison
python run_multi_scheme_compare.py --out-dir "{out_dir}" --trials all

## 2) Re-run evaluate for one trial only (example: trial01)
python experiment_60_80.py evaluate --manifest "{out_dir / 'trial01' / 'manifest_multischemes.csv'}" --out-dir "{out_dir / 'trial01'}" --baseline-run schemeA_ct_main_allwheel_trial01 --segments 0-20,20-40,40-60,60-80,80-90 --target-segment 60-80
"""
    (out_dir / "RUN_COMMANDS_5SCHEMES.md").write_text(txt, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified multi-scheme comparison for CT_FGO vs KF-GINS vs Wheel-GINS.")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--truth-root", type=Path, default=DEFAULT_TRUTH_ROOT)
    p.add_argument("--trials", default="all", help="Comma separated trial names (e.g. trial01,trial02) or 'all'.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--segments", default="0-20,20-40,40-60,60-80,80-90")
    p.add_argument("--check-only", action="store_true", help="Only validate all input paths and conversions, no evaluation.")

    p.add_argument(
        "--scheme-a-ct-template",
        default="",
        help="Path template with {trial}, e.g. D:/.../output_schemeA/.../{trial}/ct_trajectory.txt",
    )
    p.add_argument(
        "--scheme-b-ct-template",
        default="",
        help="Path template with {trial} for scheme B ct_trajectory.txt",
    )
    p.add_argument(
        "--scheme-c-ct-template",
        default="",
        help="Path template with {trial} for scheme C ct_trajectory.txt",
    )
    p.add_argument(
        "--scheme-f-ct-template",
        default="",
        help="Path template with {trial} for scheme F ct_trajectory.txt",
    )
    p.add_argument(
        "--scheme-d-kf-nav-template",
        default="",
        help="Path template with {trial} for KF-GINS nav file",
    )
    p.add_argument(
        "--scheme-e-wheel-traj-template",
        default="",
        help="Path template with {trial} for Wheel-GINS traj file",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials = parse_trials_arg(args.trials, args.truth_root)

    schemes = []
    kf_converted_list = []
    wheel_converted_list = []
    for trial in trials:
        truth_path = args.truth_root / trial / "GNSS_use.txt"
        if not truth_path.exists():
            raise FileNotFoundError(f"Missing truth file: {truth_path}")

        scheme_a_ct = resolve_input(args.data_root, "A", trial, args.scheme_a_ct_template)
        scheme_b_ct = resolve_input(args.data_root, "B", trial, args.scheme_b_ct_template)
        scheme_c_ct = resolve_input(args.data_root, "C", trial, args.scheme_c_ct_template)
        scheme_f_ct = resolve_input(args.data_root, "F", trial, args.scheme_f_ct_template)
        scheme_d_kf_nav = resolve_input(args.data_root, "D", trial, args.scheme_d_kf_nav_template)
        scheme_e_wheel_traj = resolve_input(args.data_root, "E", trial, args.scheme_e_wheel_traj_template)

        kf_out_dir = args.data_root / f"transformedforKF_GINS_schemeD_{trial}"
        wheel_out_dir = args.data_root / f"transformedforWheel_GINS_schemeE_{trial}"
        kf_converted = kf_out_dir / "ct_trajectory.txt"
        wheel_converted = wheel_out_dir / "ct_trajectory.txt"
        convert_kf_gins_nav_to_ct_like(scheme_d_kf_nav, kf_converted)
        convert_wheel_gins_traj_to_ct_like(scheme_e_wheel_traj, wheel_converted)
        kf_converted_list.append(kf_converted)
        wheel_converted_list.append(wheel_converted)

        schemes.extend(
            [
                Scheme(f"schemeA_ct_main_allwheel_{trial}", scheme_a_ct, truth_path),
                Scheme(f"schemeB_ct_main_rear_right_{trial}", scheme_b_ct, truth_path),
                Scheme(f"schemeC_ct_main_front_left_{trial}", scheme_c_ct, truth_path),
                Scheme(f"schemeF_ct_main_only_{trial}", scheme_f_ct, truth_path),
                Scheme(f"schemeD_kf_gins_main_imu_{trial}", kf_converted, truth_path),
                Scheme(f"schemeE_wheel_gins_main_rear_{trial}", wheel_converted, truth_path),
            ]
        )

    manifest_path = args.out_dir / "manifest_multischemes.csv"
    write_manifest(manifest_path, schemes)
    if args.check_only:
        print(f"Check passed. trials={len(trials)}, runs={len(schemes)}")
        print(f"Manifest: {manifest_path}")
        return

    all_summary_rows: List[dict] = []
    all_gate_rows: List[dict] = []
    for trial in trials:
        trial_rows = [s for s in schemes if s.truth_path.parent.name == trial]
        trial_manifest = args.out_dir / trial / "manifest_multischemes.csv"
        write_manifest(trial_manifest, trial_rows)
        trial_out = args.out_dir / trial
        run_unified_evaluate(trial_manifest, trial_out, f"schemeA_ct_main_allwheel_{trial}", args.segments)
        all_summary_rows.extend(read_csv(trial_out / "summary_metrics.csv"))
        all_gate_rows.extend(read_csv(trial_out / "gate_results.csv"))

    write_csv_rows(args.out_dir / "summary_metrics.csv", all_summary_rows)
    write_csv_rows(args.out_dir / "gate_results.csv", all_gate_rows)

    render_detailed_trajectory_plots(
        [{"run_name": s.run_name, "result_path": str(s.result_path), "truth_path": str(s.truth_path)} for s in schemes],
        args.out_dir,
    )

    render_charts(all_summary_rows, args.out_dir)
    write_final_report(
        args.out_dir,
        manifest_path,
        all_summary_rows,
        all_gate_rows,
        kf_converted_list[0] if kf_converted_list else Path(""),
        wheel_converted_list[0] if wheel_converted_list else Path(""),
    )
    write_command_list(args.out_dir, manifest_path)
    print(f"Done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
