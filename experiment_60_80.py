import argparse
import csv
import math
import re
import subprocess
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

R_EARTH = 6378137.0
DEFAULT_SEGMENTS = "0-20,20-40,40-60,60-80,80-90"


@dataclass
class SegmentMetric:
    segment: str
    samples: int
    pos2d_rmse_m: float
    pos2d_mae_m: float
    pos2d_max_m: float
    pos2d_max_t_s: float
    speed_rmse_mps: float
    speed_mae_mps: float


def parse_segments(text: str) -> List[Tuple[float, float]]:
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


def parse_float_list(text: str) -> List[float]:
    vals = []
    for item in text.split(","):
        s = item.strip()
        if s:
            vals.append(float(s))
    if not vals:
        raise ValueError(f"Empty float list: {text}")
    return vals


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


def segment_metric(
    res: np.ndarray,
    truth: np.ndarray,
    seg: Tuple[float, float],
    t_v_truth: np.ndarray,
    vn_truth: np.ndarray,
    ve_truth: np.ndarray,
) -> Optional[SegmentMetric]:
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
    return SegmentMetric(
        segment=f"{int(t0)}-{int(t1)}",
        samples=int(t.size),
        pos2d_rmse_m=float(np.sqrt(np.mean(pos2d_err**2))),
        pos2d_mae_m=float(np.mean(np.abs(pos2d_err))),
        pos2d_max_m=float(pos2d_err[max_idx]),
        pos2d_max_t_s=float(t[max_idx]),
        speed_rmse_mps=float(np.sqrt(np.mean(speed_err**2))),
        speed_mae_mps=float(np.mean(np.abs(speed_err))),
    )


def analyze_log(log_path: Path) -> Tuple[str, int]:
    if not log_path.exists():
        return "NO_LOG", 0
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    no_conv_hits = len(re.findall(r"NO_CONVERGENCE", txt))
    m = re.search(r"Termination:\s*([A-Z_]+)", txt)
    if m:
        return m.group(1), no_conv_hits
    if "NO_CONVERGENCE" in txt:
        return "NO_CONVERGENCE", no_conv_hits
    if "CONVERGENCE" in txt:
        return "CONVERGENCE", no_conv_hits
    return "UNKNOWN", no_conv_hits


def weighted_avg(values: List[float], weights: List[int]) -> float:
    wsum = sum(weights)
    if wsum <= 0:
        return float("nan")
    return float(sum(v * w for v, w in zip(values, weights)) / wsum)


def compute_run_metrics(result_path: Path, truth_path: Path, segments: List[Tuple[float, float]]) -> List[SegmentMetric]:
    res = load_txt(result_path)
    truth = load_txt(truth_path)
    t_v_truth, vn_truth, ve_truth, _ = compute_truth_velocity(truth)
    rows = []
    for seg in segments:
        row = segment_metric(res, truth, seg, t_v_truth, vn_truth, ve_truth)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No valid segment metrics for {result_path}")
    return rows


def save_segment_csv(path: Path, rows: List[SegmentMetric]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "segment",
                "samples",
                "pos2d_rmse_m",
                "pos2d_mae_m",
                "pos2d_max_m",
                "pos2d_max_t_s",
                "speed_rmse_mps",
                "speed_mae_mps",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.segment,
                    r.samples,
                    f"{r.pos2d_rmse_m:.9f}",
                    f"{r.pos2d_mae_m:.9f}",
                    f"{r.pos2d_max_m:.9f}",
                    f"{r.pos2d_max_t_s:.6f}",
                    f"{r.speed_rmse_mps:.9f}",
                    f"{r.speed_mae_mps:.9f}",
                ]
            )


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty manifest.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def update_config_text(
    text: str,
    speed_weight: float,
    nhc_weight: float,
    roll_weight: float,
    yaw_front_weight: float,
    yaw_rear_weight: float,
    output_path: str,
) -> str:
    def replace_numeric_keep_comment(src_line: str, key: str, value: float) -> str:
        m = re.match(rf'^(\s*{re.escape(key)}\s*:\s*)([^#]*)(.*)$', src_line)
        if not m:
            return src_line
        comment = m.group(3)
        if comment:
            # Ensure YAML comment starts with a separating space.
            comment = " " + comment.lstrip()
        return f"{m.group(1)}{value}{comment}"

    lines = text.splitlines()
    out = []
    current_block = ""
    for line in lines:
        m_block = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*$", line)
        if m_block:
            current_block = m_block.group(1)

        front_names = {"center_imu1", "center_imu3"}
        rear_names = {"center_imu2", "center_imu4"}
        yaw_value = yaw_front_weight if current_block in front_names else yaw_rear_weight
        if current_block not in front_names and current_block not in rear_names:
            yaw_value = yaw_front_weight

        line = replace_numeric_keep_comment(line, "speed_weight", speed_weight)
        line = replace_numeric_keep_comment(line, "nhc_weight", nhc_weight)
        line = replace_numeric_keep_comment(line, "attitude_weight_roll", roll_weight)
        line = replace_numeric_keep_comment(line, "attitude_weight_yaw", yaw_value)
        line = re.sub(r'^(\s*outputpath\s*:\s*)(["\']).*(["\'])(\s*#.*)?$', rf'\g<1>"{output_path}"', line)
        out.append(line)
    return "\n".join(out) + "\n"


def cmd_gen_grid(args):
    base_config = Path(args.base_config)
    out_dir = Path(args.out_dir)
    config_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    result_root = out_dir / "results"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    base_text = base_config.read_text(encoding="utf-8", errors="ignore")
    speed_vals = parse_float_list(args.speed_values)
    nhc_vals = parse_float_list(args.nhc_values)
    roll_vals = parse_float_list(args.roll_values)
    yaw_f_vals = parse_float_list(args.yaw_front_values)
    yaw_r_vals = parse_float_list(args.yaw_rear_values)

    rows = []
    for speed, nhc, roll, yaw_f, yaw_r in product(speed_vals, nhc_vals, roll_vals, yaw_f_vals, yaw_r_vals):
        run_name = f"sw{speed:g}_nhc{nhc:g}_roll{roll:g}_yawf{yaw_f:g}_yawr{yaw_r:g}"
        run_output = result_root / run_name
        cfg_text = update_config_text(
            base_text,
            speed_weight=speed,
            nhc_weight=nhc,
            roll_weight=roll,
            yaw_front_weight=yaw_f,
            yaw_rear_weight=yaw_r,
            output_path=str(run_output).replace("\\", "/"),
        )
        cfg_path = config_dir / f"{run_name}.yaml"
        cfg_path.write_text(cfg_text, encoding="utf-8")
        rows.append(
            {
                "run_name": run_name,
                "config_path": str(cfg_path),
                "result_path": str(run_output / "ct_trajectory.txt"),
                "truth_path": args.truth_path,
                "log_path": str(log_dir / f"{run_name}.log"),
                "status": "PENDING",
                "runtime_sec": "",
                "return_code": "",
            }
        )

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_path, rows)
    print(f"Generated {len(rows)} runs.")
    print(f"Manifest: {manifest_path}")


def cmd_run(args):
    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Empty manifest.")
    exe = Path(args.executable)
    if not exe.exists():
        raise FileNotFoundError(f"Executable not found: {exe}")

    for row in rows:
        if args.only_pending and row.get("status", "").upper() == "DONE":
            continue
        cfg = Path(row["config_path"])
        log = Path(row["log_path"])
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with log.open("w", encoding="utf-8", errors="ignore") as lf:
            proc = subprocess.run([str(exe), str(cfg)], stdout=lf, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter() - start
        row["runtime_sec"] = f"{elapsed:.3f}"
        row["return_code"] = str(proc.returncode)
        row["status"] = "DONE" if proc.returncode == 0 else "FAILED"
        print(f"[{row['status']}] {row['run_name']} rc={proc.returncode} time={elapsed:.2f}s")

    write_manifest(manifest_path, rows)
    print(f"Updated manifest: {manifest_path}")


def cmd_evaluate(args):
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = out_dir / "per_run_segments"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    segments = parse_segments(args.segments)
    seg_names = [f"{int(a)}-{int(b)}" for a, b in segments]

    rows = read_manifest(manifest_path)
    summary_rows = []
    run_to_metrics: Dict[str, Dict[str, SegmentMetric]] = {}
    for row in rows:
        run_name = row["run_name"]
        result_path = Path(row["result_path"])
        truth_path = Path(row["truth_path"] if row.get("truth_path") else args.truth_path)
        if not result_path.exists() or not truth_path.exists():
            print(f"[SKIP] {run_name}: missing result or truth.")
            continue
        metrics = compute_run_metrics(result_path, truth_path, segments)
        save_segment_csv(per_run_dir / f"{run_name}.csv", metrics)
        metric_map = {m.segment: m for m in metrics}
        run_to_metrics[run_name] = metric_map

        pos_vals = [m.pos2d_rmse_m for m in metrics]
        speed_vals = [m.speed_rmse_mps for m in metrics]
        samples = [m.samples for m in metrics]
        term, no_conv_hits = analyze_log(Path(row["log_path"])) if row.get("log_path") else ("NO_LOG", 0)

        target_seg = args.target_segment
        if target_seg not in metric_map:
            print(f"[SKIP] {run_name}: target segment {target_seg} missing.")
            continue
        target = metric_map[target_seg]
        summary_rows.append(
            {
                "run_name": run_name,
                "status": row.get("status", ""),
                "return_code": row.get("return_code", ""),
                "runtime_sec": row.get("runtime_sec", ""),
                "termination": term,
                "no_convergence_hits": str(no_conv_hits),
                "pos_rmse_60_80_m": f"{target.pos2d_rmse_m:.9f}",
                "speed_rmse_60_80_mps": f"{target.speed_rmse_mps:.9f}",
                "pos_rmse_global_w_m": f"{weighted_avg(pos_vals, samples):.9f}",
                "speed_rmse_global_w_mps": f"{weighted_avg(speed_vals, samples):.9f}",
            }
        )

    if not summary_rows:
        raise RuntimeError("No runs were evaluated.")

    baseline_name = args.baseline_run
    if baseline_name not in run_to_metrics:
        raise RuntimeError(f"Baseline run not found in evaluated metrics: {baseline_name}")
    base = run_to_metrics[baseline_name]
    base_target = base[args.target_segment]

    gate_rows = []
    for s in summary_rows:
        run_name = s["run_name"]
        cur = run_to_metrics[run_name]
        cur_target = cur[args.target_segment]
        improve_60_80_pct = (base_target.pos2d_rmse_m - cur_target.pos2d_rmse_m) / base_target.pos2d_rmse_m * 100.0

        non_target_degrades = []
        for seg in seg_names:
            if seg == args.target_segment:
                continue
            if seg in base and seg in cur and base[seg].pos2d_rmse_m > 1e-9:
                deg = (cur[seg].pos2d_rmse_m - base[seg].pos2d_rmse_m) / base[seg].pos2d_rmse_m * 100.0
                non_target_degrades.append(deg)
        max_non_target_deg = max(non_target_degrades) if non_target_degrades else 0.0

        speed_deg_60_80 = (cur_target.speed_rmse_mps - base_target.speed_rmse_mps) / max(base_target.speed_rmse_mps, 1e-9) * 100.0
        pass_target = improve_60_80_pct >= args.min_improve_pct
        pass_non_target = max_non_target_deg <= args.max_non_target_degrade_pct
        pass_speed = speed_deg_60_80 <= args.max_speed_degrade_pct
        pass_convergence = s["termination"] not in {"NO_CONVERGENCE", "FAILURE", "DID_NOT_RUN"}
        gate_rows.append(
            {
                "run_name": run_name,
                "improve_60_80_pos_rmse_pct": f"{improve_60_80_pct:.3f}",
                "max_non_target_pos_rmse_degrade_pct": f"{max_non_target_deg:.3f}",
                "speed_rmse_60_80_degrade_pct": f"{speed_deg_60_80:.3f}",
                "pass_target": str(pass_target),
                "pass_non_target": str(pass_non_target),
                "pass_speed": str(pass_speed),
                "pass_convergence": str(pass_convergence),
                "overall_pass": str(pass_target and pass_non_target and pass_speed and pass_convergence),
            }
        )

    summary_csv = out_dir / "summary_metrics.csv"
    gates_csv = out_dir / "gate_results.csv"
    write_manifest(summary_csv, summary_rows)
    write_manifest(gates_csv, gate_rows)

    report_lines = [
        "# 60-80s Experiment Evaluation",
        "",
        f"- Baseline: `{baseline_name}`",
        f"- Target segment: `{args.target_segment}`",
        f"- Gate: improve >= {args.min_improve_pct:.1f}%, non-target degrade <= {args.max_non_target_degrade_pct:.1f}%, speed degrade <= {args.max_speed_degrade_pct:.1f}%",
        "",
        "## Runs",
    ]
    for s in summary_rows:
        g = next(x for x in gate_rows if x["run_name"] == s["run_name"])
        report_lines.append(
            f"- `{s['run_name']}` | pos60-80={float(s['pos_rmse_60_80_m']):.4f}m | "
            f"speed60-80={float(s['speed_rmse_60_80_mps']):.4f}m/s | term={s['termination']} | pass={g['overall_pass']}"
        )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {gates_csv}")
    print(f"Saved: {out_dir / 'report.md'}")


def build_parser():
    p = argparse.ArgumentParser(description="Phased reproducible pipeline for 60-80s accuracy experiments.")
    sp = p.add_subparsers(dest="command", required=True)

    p_gen = sp.add_parser("gen-grid", help="Generate grid-scan configs and manifest for P1-1.")
    p_gen.add_argument("--base-config", required=True)
    p_gen.add_argument("--out-dir", required=True)
    p_gen.add_argument("--truth-path", required=True)
    p_gen.add_argument("--speed-values", default="8,10,12")
    p_gen.add_argument("--nhc-values", default="8,10,12")
    p_gen.add_argument("--roll-values", default="80,100,120")
    p_gen.add_argument("--yaw-front-values", default="0.5,1,2")
    p_gen.add_argument("--yaw-rear-values", default="60,100,140")
    p_gen.set_defaults(func=cmd_gen_grid)

    p_run = sp.add_parser("run", help="Run all configs in manifest.")
    p_run.add_argument("--manifest", required=True)
    p_run.add_argument("--executable", required=True)
    p_run.add_argument("--only-pending", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_eval = sp.add_parser("evaluate", help="Evaluate runs and apply acceptance gates.")
    p_eval.add_argument("--manifest", required=True)
    p_eval.add_argument("--out-dir", required=True)
    p_eval.add_argument("--baseline-run", required=True)
    p_eval.add_argument("--truth-path", default="")
    p_eval.add_argument("--segments", default=DEFAULT_SEGMENTS)
    p_eval.add_argument("--target-segment", default="60-80")
    p_eval.add_argument("--min-improve-pct", type=float, default=15.0)
    p_eval.add_argument("--max-non-target-degrade-pct", type=float, default=5.0)
    p_eval.add_argument("--max-speed-degrade-pct", type=float, default=5.0)
    p_eval.set_defaults(func=cmd_evaluate)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
