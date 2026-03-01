import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np


DATA_ROOT = Path(r"D:\Code\dataset\WID\Datasets\transformedData2")
TRUTH_ROOT = DATA_ROOT / "four_wheel_dataset_PassengerCar"
CT_EXE = Path(r".\bin\Release\ob_gins_ct.exe")
KF_EXE = Path(r"D:\Code\KF-GINS\bin\Release\KF-GINS.exe")
WHEEL_EXE = Path(r"D:\Code\Wheel-GINS\bin\Release\Wheel-GINS.exe")
KF_TEMPLATE = Path(r"D:\Code\KF-GINS\config\kf-gins-trial01.yaml")
WHEEL_TEMPLATE = Path(r"D:\Code\Wheel-GINS\config\wid_single_rear2.yaml")


def discover_trials(truth_root: Path, trials_arg: str) -> List[str]:
    if trials_arg.strip().lower() == "all":
        trials = sorted(p.name for p in truth_root.glob("trial*") if p.is_dir())
        if not trials:
            raise RuntimeError(f"No trials found in {truth_root}")
        return trials
    trials = [x.strip() for x in trials_arg.split(",") if x.strip()]
    if not trials:
        raise ValueError("Empty --trials")
    return trials


def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def is_valid_nonempty_file(path: Path, min_bytes: int = 64) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def load_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim != 2:
        raise ValueError(f"Invalid file shape: {path}")
    return arr


def convert_body_imu_to_kf_delta(body_imu_txt: Path, out_txt: Path) -> None:
    raw = load_txt(body_imu_txt)
    if raw.shape[1] < 7:
        raise ValueError(f"Unexpected Body_IMU columns: {body_imu_txt}")
    t = raw[:, 0]
    dt = np.diff(t, prepend=t[0])
    if dt.size > 1 and dt[0] == 0:
        dt[0] = dt[1]
    out = raw.copy()
    out[:, 1:7] = raw[:, 1:7] * dt[:, None]
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_txt, out[:, :7], fmt="%.12e")


def convert_center_imu_to_wheel_rate_bin(center_imu_txt: Path, out_bin: Path) -> None:
    raw = load_txt(center_imu_txt)
    if raw.shape[1] < 7:
        raise ValueError(f"Unexpected Center_IMU columns: {center_imu_txt}")
    t = raw[:, 0]
    dt = np.diff(t, prepend=t[0])
    if dt.size > 1 and dt[0] == 0:
        dt[0] = dt[1]
    rate = np.zeros((raw.shape[0], 7), dtype=np.float64)
    rate[:, 0] = t
    rate[:, 1:7] = raw[:, 1:7] / dt[:, None]
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    rate.tofile(out_bin)


def patch_yaml_path_line(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    out = []
    hit = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(f"{key}:"):
            indent = line[: len(line) - len(stripped)]
            out.append(f'{indent}{key}: "{value}"')
            hit = True
        else:
            out.append(line)
    if not hit:
        out.append(f'{key}: "{value}"')
    return "\n".join(out) + "\n"


def patch_yaml_vec_line(text: str, key: str, vec3: np.ndarray) -> str:
    lines = text.splitlines()
    out = []
    hit = False
    v = f"[ {vec3[0]:.10f}, {vec3[1]:.10f}, {vec3[2]:.4f} ]"
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(f"{key}:"):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}{key}: {v}")
            hit = True
        else:
            out.append(line)
    if not hit:
        out.append(f"{key}: {v}")
    return "\n".join(out) + "\n"


def patch_yaml_scalar_line(text: str, key: str, value: float) -> str:
    lines = text.splitlines()
    out = []
    hit = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(f"{key}:"):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}{key}: {value:.6f}")
            hit = True
        else:
            out.append(line)
    if not hit:
        out.append(f"{key}: {value:.6f}")
    return "\n".join(out) + "\n"


def make_ct_cfg(base_cfg: Path, trial: str, out_dir: Path, cfg_out: Path) -> None:
    s = read_text(base_cfg)
    s = s.replace("/trial01/", f"/{trial}/")
    s = s.replace("\\trial01\\", f"\\{trial}\\")
    s = patch_yaml_path_line(s, "outputpath", out_dir.as_posix())
    write_text(cfg_out, s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all 6 schemes on all WID trials.")
    ap.add_argument("--trials", default="all")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--out-root", type=Path, default=DATA_ROOT / "six_scheme_eval_20260301_all")
    args = ap.parse_args()

    trials = discover_trials(TRUTH_ROOT, args.trials)
    cfg_root = args.out_root / "generated_configs"

    ct_schemes = [
        ("A", Path("config/ob_gins_ct_schemeA_main_allwheel.yaml"), DATA_ROOT / "output_schemeA_main_allwheel_alltrials"),
        ("B", Path("config/ob_gins_ct_schemeB_main_rear_right.yaml"), DATA_ROOT / "output_schemeB_main_rear_right_alltrials"),
        ("C", Path("config/ob_gins_ct_schemeC_main_front_left.yaml"), DATA_ROOT / "output_schemeC_main_front_left_alltrials"),
        ("F", Path("config/ob_gins_ct_schemeF_main_only.yaml"), DATA_ROOT / "output_schemeF_main_only_alltrials"),
    ]

    for trial in trials:
        truth = TRUTH_ROOT / trial / "GNSS_use.txt"
        if not truth.exists():
            raise FileNotFoundError(f"Missing truth: {truth}")
        gnss = load_txt(truth)
        init_blh = gnss[0, 1:4]
        t_start = float(gnss[0, 0])
        t_end = float(gnss[-1, 0])

        for scheme_key, base_cfg, scheme_out_root in ct_schemes:
            out_dir = scheme_out_root / trial
            cfg_path = cfg_root / "ct" / f"scheme{scheme_key}_{trial}.yaml"
            make_ct_cfg(base_cfg, trial, out_dir, cfg_path)
            if args.prepare_only:
                continue
            ct_traj = out_dir / "ct_trajectory.txt"
            if args.skip_existing and is_valid_nonempty_file(ct_traj):
                print(f"SKIP existing CT {scheme_key} {trial}")
            else:
                run_cmd([str(CT_EXE), str(cfg_path)])

        # KF-GINS (Scheme D)
        kf_out = DATA_ROOT / f"kf_gins_D_mainimu_{trial}_auto"
        kf_delta = kf_out / "Body_IMU_delta_for_kfgins.txt"
        body_imu = TRUTH_ROOT / trial / "Body_IMU.txt"
        convert_body_imu_to_kf_delta(body_imu, kf_delta)
        kf_delta_arr = load_txt(kf_delta)
        if kf_delta_arr[0, 0] > 1e-9:
            first = np.zeros((1, 7), dtype=np.float64)
            first[0, 0] = 0.0
            kf_delta_arr = np.vstack([first, kf_delta_arr[:, :7]])
            np.savetxt(kf_delta, kf_delta_arr, fmt="%.12e")
        kf_cfg_text = read_text(KF_TEMPLATE)
        kf_cfg_text = patch_yaml_path_line(kf_cfg_text, "imupath", kf_delta.as_posix())
        kf_cfg_text = patch_yaml_path_line(kf_cfg_text, "gnsspath", truth.as_posix())
        kf_cfg_text = patch_yaml_path_line(kf_cfg_text, "outputpath", kf_out.as_posix())
        kf_cfg_text = patch_yaml_vec_line(kf_cfg_text, "initpos", init_blh)
        kf_cfg_text = patch_yaml_scalar_line(kf_cfg_text, "starttime", t_start)
        kf_cfg_text = patch_yaml_scalar_line(kf_cfg_text, "endtime", t_end)
        kf_cfg = cfg_root / "kf" / f"kf-gins-{trial}.yaml"
        write_text(kf_cfg, kf_cfg_text)
        if not args.prepare_only:
            kf_nav = kf_out / "KF_GINS_Navresult.nav"
            if args.skip_existing and is_valid_nonempty_file(kf_nav):
                print(f"SKIP existing KF {trial}")
            else:
                run_cmd([str(KF_EXE), str(kf_cfg)], cwd=KF_EXE.parent.parent.parent)

        # Wheel-GINS (Scheme E, rear wheel IMU2)
        wheel_out = DATA_ROOT / f"wheel_gins_single_rear2_{trial}_auto"
        wheel_rate_bin = wheel_out / "Center_IMU2_rate_7col.bin"
        center_imu2 = TRUTH_ROOT / trial / "Center_IMU2.txt"
        convert_center_imu_to_wheel_rate_bin(center_imu2, wheel_rate_bin)
        wheel_rate_arr = np.fromfile(wheel_rate_bin, dtype=np.float64).reshape(-1, 7)
        if wheel_rate_arr[0, 0] > 1e-9:
            first = np.zeros((1, 7), dtype=np.float64)
            wheel_rate_arr = np.vstack([first, wheel_rate_arr])
            wheel_rate_arr.tofile(wheel_rate_bin)
        wheel_cfg_text = read_text(WHEEL_TEMPLATE)
        wheel_cfg_text = patch_yaml_path_line(wheel_cfg_text, "imupath", wheel_rate_bin.as_posix())
        wheel_cfg_text = patch_yaml_path_line(wheel_cfg_text, "gnsspath", truth.as_posix())
        wheel_cfg_text = patch_yaml_path_line(wheel_cfg_text, "outputpath", wheel_out.as_posix())
        wheel_cfg_text = patch_yaml_vec_line(wheel_cfg_text, "initBLH", init_blh)
        wheel_cfg_text = patch_yaml_scalar_line(wheel_cfg_text, "starttime", t_start)
        wheel_cfg_text = patch_yaml_scalar_line(wheel_cfg_text, "endtime", t_end)
        wheel_cfg = cfg_root / "wheel" / f"wheel-gins-{trial}.yaml"
        write_text(wheel_cfg, wheel_cfg_text)
        if not args.prepare_only:
            wheel_traj = wheel_out / "traj.txt"
            if args.skip_existing and is_valid_nonempty_file(wheel_traj):
                print(f"SKIP existing Wheel {trial}")
            else:
                run_cmd([str(WHEEL_EXE), str(wheel_cfg)], cwd=WHEEL_EXE.parent.parent.parent)

    if args.prepare_only:
        print(f"Prepared configs and converted IMU files for trials: {', '.join(trials)}")
        return

    cmd = [
        sys.executable,
        "run_multi_scheme_compare.py",
        "--trials",
        ",".join(trials),
        "--out-dir",
        str(args.out_root),
        "--scheme-a-ct-template",
        str((DATA_ROOT / "output_schemeA_main_allwheel_alltrials" / "{trial}" / "ct_trajectory.txt").as_posix()),
        "--scheme-b-ct-template",
        str((DATA_ROOT / "output_schemeB_main_rear_right_alltrials" / "{trial}" / "ct_trajectory.txt").as_posix()),
        "--scheme-c-ct-template",
        str((DATA_ROOT / "output_schemeC_main_front_left_alltrials" / "{trial}" / "ct_trajectory.txt").as_posix()),
        "--scheme-f-ct-template",
        str((DATA_ROOT / "output_schemeF_main_only_alltrials" / "{trial}" / "ct_trajectory.txt").as_posix()),
        "--scheme-d-kf-nav-template",
        str((DATA_ROOT / "kf_gins_D_mainimu_{trial}_auto" / "KF_GINS_Navresult.nav").as_posix()),
        "--scheme-e-wheel-traj-template",
        str((DATA_ROOT / "wheel_gins_single_rear2_{trial}_auto" / "traj.txt").as_posix()),
    ]
    run_cmd(cmd, cwd=Path(__file__).resolve().parent)
    print(f"All done. Unified outputs: {args.out_root}")


if __name__ == "__main__":
    main()
