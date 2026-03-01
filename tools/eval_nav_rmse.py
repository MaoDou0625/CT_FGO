import argparse
import math
from bisect import bisect_left


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def load_txt(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.replace(",", " ").split()
            rows.append([float(x) for x in parts])
    return rows


def interp_truth(truth, t):
    ts = [r[0] for r in truth]
    i = bisect_left(ts, t)
    if i <= 0 or i >= len(truth):
        return None
    t0, t1 = truth[i - 1][0], truth[i][0]
    if t1 <= t0:
        return None
    w = (t - t0) / (t1 - t0)
    out = []
    for c in range(1, 4):
        out.append(truth[i - 1][c] * (1 - w) + truth[i][c] * w)
    return out


def radii(lat_rad):
    s = math.sin(lat_rad)
    den = math.sqrt(1.0 - WGS84_E2 * s * s)
    rn = WGS84_A / den
    rm = WGS84_A * (1.0 - WGS84_E2) / (den ** 3)
    return rm, rn


def lla_diff_to_enu_m(lat_ref_deg, dlat_deg, dlon_deg, dh):
    lat = math.radians(lat_ref_deg)
    rm, rn = radii(lat)
    de = math.radians(dlon_deg) * (rn) * math.cos(lat)
    dn = math.radians(dlat_deg) * (rm)
    du = dh
    return de, dn, du


def eval_rmse(nav, truth, nav_cols):
    # nav_cols: (time_idx, lat_idx, lon_idx, h_idx)
    et2 = en2 = eu2 = e3d2 = 0.0
    n = 0
    for r in nav:
        t = r[nav_cols[0]]
        lat = r[nav_cols[1]]
        lon = r[nav_cols[2]]
        h = r[nav_cols[3]]
        gt = interp_truth(truth, t)
        if gt is None:
            continue
        lat_t, lon_t, h_t = gt
        de, dn, du = lla_diff_to_enu_m(lat_t, lat - lat_t, lon - lon_t, h - h_t)
        et2 += de * de
        en2 += dn * dn
        eu2 += du * du
        e3d2 += de * de + dn * dn + du * du
        n += 1
    if n == 0:
        return None
    return {
        "n": n,
        "rmse_e": math.sqrt(et2 / n),
        "rmse_n": math.sqrt(en2 / n),
        "rmse_u": math.sqrt(eu2 / n),
        "rmse_2d": math.sqrt((et2 + en2) / n),
        "rmse_3d": math.sqrt(e3d2 / n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True)
    ap.add_argument("--kf", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()

    ct = load_txt(args.ct)
    kf = load_txt(args.kf)
    truth = load_txt(args.truth)

    # CT: time lat lon h ...
    ct_m = eval_rmse(ct, truth, (0, 1, 2, 3))
    # KF nav: week time lat lon h ...
    kf_m = eval_rmse(kf, truth, (1, 2, 3, 4))

    if ct_m is None or kf_m is None:
        print("EVAL_FAILED")
        return

    print(
        f"CT_D,{ct_m['n']},{ct_m['rmse_e']:.6f},{ct_m['rmse_n']:.6f},{ct_m['rmse_u']:.6f},{ct_m['rmse_2d']:.6f},{ct_m['rmse_3d']:.6f}"
    )
    print(
        f"KF,{kf_m['n']},{kf_m['rmse_e']:.6f},{kf_m['rmse_n']:.6f},{kf_m['rmse_u']:.6f},{kf_m['rmse_2d']:.6f},{kf_m['rmse_3d']:.6f}"
    )


if __name__ == "__main__":
    main()
