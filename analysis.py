#!/usr/bin/env python3
"""
race_report.py
==============
Generates a Markdown report (plus figures and CSV tables) from a ROS 2 CSV export
of a drone race (second CSV format, with twist columns).

Usage
-----
    python race_report.py run.csv
    python race_report.py run.csv --out my_report

Output folder
-------------
    <out>/report.md
    <out>/figures/*.png (+ gate_crossings.pdf)
    <out>/tables/*.csv
    <out>/config.json
"""

# =============================================================================
# 0. IMPORTS AND GLOBAL SETTINGS
# =============================================================================
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless: figures are saved, never shown
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401  (registers the 3D projection)

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices, 0/0


# =============================================================================
# 1. COMMAND-LINE ARGUMENTS
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Drone race CSV -> Markdown report")
    p.add_argument("csv", help="input CSV file (second format, with twist columns)")
    p.add_argument("--out", default=None, help="output folder (default: report_<csvname>)")
    p.add_argument("--ns", default="/uav1", help="topic namespace prefix")
    p.add_argument("--z-shift", type=float, default=0.0,
                   help="metres subtracted from the drone z (ground truth AND reference)")
    p.add_argument("--margin", type=float, default=0.05,
                   help="tolerance added to the half-aperture when validating a crossing [m]")
    p.add_argument("--window", type=float, default=0.30,
                   help="path length shown before/after each gate plane in the grid [m]")
    p.add_argument("--drone-diam", type=float, default=0.10,
                   help="drone footprint drawn in the gate grid [m] (0 to hide)")
    p.add_argument("--grid-lim", type=float, default=0.34,
                   help="half-size of each gate panel in the grid [m] (outer frame is 0.30)")
    p.add_argument("--dpi", type=int, default=250)
    return p.parse_args()


# =============================================================================
# 2. LOADING AND CLEANING THE CSV
# =============================================================================
def column_names(ns):
    """Column names are selected by name, never by position."""
    ax = "xyz"
    return dict(
        gt_pos=[f"{ns}/ground_truth/pose/pose/position/{a}" for a in ax],
        gt_vel=[f"{ns}/ground_truth/twist/twist/linear/{a}" for a in ax],
        ref_pos=[f"{ns}/control_manager/state/reference/pose/position/{a}" for a in ax],
        ref_vel=[f"{ns}/control_manager/state/reference/twist/linear/{a}" for a in ax],
    )


def interp_columns(t_new, t_src, df, cols):
    """Interpolate the columns of df (sampled at t_src) onto t_new. NaN outside the range."""
    out = np.column_stack([np.interp(t_new, t_src, df[c].to_numpy(float)) for c in cols])
    out[(t_new < t_src[0]) | (t_new > t_src[-1])] = np.nan
    return out


def load_run(csv_path, ns, z_shift):
    cols = column_names(ns)
    df = pd.read_csv(csv_path).sort_values("time").reset_index(drop=True)

    required = ["time"] + cols["gt_pos"] + cols["ref_pos"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Missing columns in the CSV:\n  " + "\n  ".join(missing))

    # --- ground truth: keep only rows where it was actually published
    gt = df.dropna(subset=cols["gt_pos"])
    t_abs = gt["time"].to_numpy(float)
    t0 = t_abs[0]
    t = t_abs - t0
    xyz = gt[cols["gt_pos"]].to_numpy(float).copy()
    xyz[:, 2] -= z_shift

    # --- speed: from twist if available, otherwise from the position derivative
    speed, speed_src = None, "ground-truth twist"
    if all(c in df.columns for c in cols["gt_vel"]):
        v = gt[cols["gt_vel"]].interpolate(limit_direction="both").to_numpy(float)
        speed = np.linalg.norm(v, axis=1)
    if speed is None or np.isnan(speed).all():
        speed = np.linalg.norm(np.gradient(xyz, t, axis=0), axis=1)
        speed_src = "derivative of the ground-truth position"

    # --- reference (controller) position/velocity, interpolated onto the ground-truth times
    rp = df.dropna(subset=cols["ref_pos"])
    rt = rp["time"].to_numpy(float) - t0
    ref = interp_columns(t, rt, rp, cols["ref_pos"])
    ref[:, 2] -= z_shift
    track_err = np.linalg.norm(xyz - ref, axis=1)

    ref_speed = None
    if all(c in df.columns for c in cols["ref_vel"]):
        rv = df.dropna(subset=cols["ref_vel"])
        if len(rv) > 1:
            ref_speed = np.linalg.norm(
                interp_columns(t, rv["time"].to_numpy(float) - t0, rv, cols["ref_vel"]), axis=1)

    return dict(t=t, xyz=xyz, ref=ref, speed=speed, speed_src=speed_src,
                ref_speed=ref_speed, track_err=track_err,
                t_start_abs=t0, n_rows=len(df), n_gt=len(gt),
                speed_scale=1.0, ref_speed_scale=1.0,
                rate=1.0 / np.median(np.diff(t)))


# =============================================================================
# 3. GATES (embedded in the script)
# =============================================================================
# Gate poses from the Gazebo world `crazyflie_map` (source: obstacles_crazyrun.obj).
# Values are the original ones from the gates file; the z correction is applied below.
# heading_rad = yaw of the crossing direction (0 = +x, pi/2 = +y). Order assumed = race order.
GATE_APERTURE_M = 0.5        # inner opening, square
GATE_OUTER_M = 0.6           # outer side of the frame
GATE_Z_OFFSET = -0.20        # [m] fixed shift applied to every gate centre z (gates moved down)

GATES_RAW = [
    dict(id=1, name="Cube",     center=[-0.117, -0.978, 1.074], heading_rad=-1.5708),
    dict(id=2, name="Cube.001", center=[-0.334, -2.774, 0.574], heading_rad=-2.1869),
    dict(id=3, name="Cube.002", center=[-1.555, -3.697, 1.074], heading_rad=-3.1416),
    dict(id=4, name="Cube.003", center=[-3.281, -3.551, 0.574], heading_rad=2.2672),
    dict(id=5, name="Cube.004", center=[-3.883, -2.147, 1.074], heading_rad=1.5708),
    dict(id=6, name="Cube.005", center=[-3.152, -0.48, 1.074],  heading_rad=0.2269),
    dict(id=7, name="Cube.006", center=[-1.742, -0.219, 0.574], heading_rad=0.0),
]


def load_gates():
    centers = np.array([g["center"] for g in GATES_RAW], float)
    centers[:, 2] += GATE_Z_OFFSET                          # manual z correction
    return dict(
        aperture=GATE_APERTURE_M,
        outer=GATE_OUTER_M,
        ids=[g["id"] for g in GATES_RAW],
        centers=centers,
        headings=np.array([g["heading_rad"] for g in GATES_RAW], float),
    )


def gate_axes(h):
    """Unit vectors: n = crossing direction, u = horizontal axis in the gate plane."""
    n = np.array([np.cos(h), np.sin(h), 0.0])
    u = np.array([-np.sin(h), np.cos(h), 0.0])
    return n, u


# =============================================================================
# 4. GATE CROSSINGS AND LAP DETECTION (gate 1 -> gate 1)
# =============================================================================
def find_crossings(t, xyz, speed, gates, margin):
    """All forward crossings of each gate plane, with an inside-aperture check."""
    half = gates["aperture"] / 2 + margin
    events = []
    for gid, c, h in zip(gates["ids"], gates["centers"], gates["headings"]):
        n, u = gate_axes(h)
        s = (xyz - c) @ n                                   # signed distance from the plane
        for i in np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0)):
            a = -s[i] / (s[i + 1] - s[i])                   # interpolation factor
            p = xyz[i] + a * (xyz[i + 1] - xyz[i])
            lat, ver = (p - c) @ u, p[2] - c[2]
            events.append(dict(gate=gid, time=t[i] + a * (t[i + 1] - t[i]),
                               lateral=lat, vertical=ver,
                               speed=speed[i] + a * (speed[i + 1] - speed[i]),
                               valid=abs(lat) <= half and abs(ver) <= half))
    return sorted(events, key=lambda e: e["time"])


def build_laps(t, xyz, speed, gates, margin):
    """
    Returns
      table      : (lap_index, gate_id) -> crossing event (used for the gate grid)
      gate_times : array (n_laps, n_gates + 1). Column j = time of the valid crossing of
                   gate j; last column = return to gate 1 (NaN for the partial last lap).
                   NaN = gate not crossed (validly) in that lap.
      rejected   : crossings that fell outside the aperture
    """
    ids = gates["ids"]
    n_g = len(ids)
    events = find_crossings(t, xyz, speed, gates, margin)
    starts = [e for e in events if e["gate"] == ids[0] and e["valid"]]
    if not starts:
        raise SystemExit("No valid crossing of gate 1: check gate frame, headings, "
                         "--z-shift or --margin")

    n_laps = len(starts)
    ends = [e["time"] for e in starts[1:]] + [np.inf]
    table, gate_times = {}, np.full((n_laps, n_g + 1), np.nan)

    for k, (e0, te) in enumerate(zip(starts, ends)):
        table[(k, ids[0])] = e0
        gate_times[k, 0] = e0["time"]
        prev = e0["time"]
        for j, gid in enumerate(ids[1:], start=1):
            cand = [e for e in events if e["gate"] == gid and prev < e["time"] < te]
            if not cand:
                continue
            valid = [e for e in cand if e["valid"]]
            e = valid[0] if valid else cand[0]              # prefer a valid crossing
            table[(k, gid)] = e
            if e["valid"]:
                gate_times[k, j] = e["time"]
                prev = e["time"]
        if k < n_laps - 1:
            gate_times[k, -1] = te
    rejected = [e for e in events if not e["valid"]]
    return table, gate_times, rejected


# =============================================================================
# 5. LAP AND SEGMENT STATISTICS
# =============================================================================
def analyze_laps(run, gate_times, gates):
    t, xyz, speed, err = run["t"], run["xyz"], run["speed"], run["track_err"]
    ids = gates["ids"]
    n_g = len(ids)
    n_laps = gate_times.shape[0]
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]   # 3D path length
    seg_cols = [f"{ids[i]}→{ids[(i + 1) % n_g]}" for i in range(n_g)]

    # ---- segments (gate i -> gate i+1, last one returns to gate 1)
    seg_t = np.diff(gate_times, axis=1)
    seg_d = np.diff(np.interp(gate_times, t, cum), axis=1)
    seg_v = seg_d / seg_t
    seg_vmax = np.full_like(seg_t, np.nan)
    seg_vmin = np.full_like(seg_t, np.nan)
    for k in range(n_laps):
        for j in range(n_g):
            a, b = gate_times[k, j], gate_times[k, j + 1]
            if np.isnan(a) or np.isnan(b):
                continue
            m = (t >= a) & (t <= b)
            if m.any():
                seg_vmax[k, j], seg_vmin[k, j] = speed[m].max(), speed[m].min()

    laps_idx = pd.Index(range(1, n_laps + 1), name="lap")
    mk = lambda a: pd.DataFrame(a, columns=seg_cols, index=laps_idx)
    segs = dict(time=mk(seg_t), dist=mk(seg_d), avg_speed=mk(seg_v),
                max_speed=mk(seg_vmax), min_speed=mk(seg_vmin))

    # ---- speed at each gate crossing (n_laps x n_gates)
    segs["gate_speed"] = pd.DataFrame(np.interp(gate_times[:, :-1], t, speed),
                                      columns=[f"gate {g}" for g in ids], index=laps_idx)

    # ---- laps
    rows = []
    for k in range(n_laps):
        ts, te = gate_times[k, 0], gate_times[k, -1]
        complete = not np.isnan(te)
        regular = complete and not np.isnan(gate_times[k]).any()
        m = (t >= ts) & (t <= (te if complete else t[-1]))
        length = np.interp(te, t, cum) - np.interp(ts, t, cum) if complete else np.nan
        time_ = te - ts if complete else np.nan
        s_m, t_m = speed[m], t[m]
        dist_twist = np.sum(0.5 * (s_m[1:] + s_m[:-1]) * np.diff(t_m)) if m.sum() > 1 else np.nan
        len_pos = cum[m][-1] - cum[m][0] if m.sum() > 1 else np.nan
        ref_ratio = np.nan                       # same check on the controller reference
        if run["ref_speed"] is not None and m.sum() > 1:
            rs, rpos = run["ref_speed"][m], run["ref"][m]
            d_twist = np.nansum(0.5 * (rs[1:] + rs[:-1]) * np.diff(t_m))
            d_pos = np.nansum(np.linalg.norm(np.diff(rpos, axis=0), axis=1))
            ref_ratio = d_twist / d_pos if d_pos > 0 else np.nan
        rows.append({
            "status": "regular" if regular else ("irregular" if complete else "partial"),
            "t_start [s]": ts, "time [s]": time_, "length [m]": length,
            "avg speed [m/s]": length / time_ if complete else np.nan,
            "mean speed [m/s]": speed[m].mean(), "max speed [m/s]": speed[m].max(),
            "min speed [m/s]": speed[m].min(),
            "twist/position ratio": dist_twist / len_pos,
            "ref twist/position ratio": ref_ratio,
            "track err mean [m]": np.nanmean(err[m]), "track err max [m]": np.nanmax(err[m]),
            "missed gates": [ids[i] for i in range(1, n_g) if np.isnan(gate_times[k, i])],
        })
    laps = pd.DataFrame(rows, index=laps_idx)
    return laps, segs


# =============================================================================
# 6. FIGURES
# =============================================================================
# ----- 6.1 Speed profile and tracking error over the whole race
def plot_speed_profile(run, gate_times, path, dpi):
    t = run["t"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1.3]))
    a1.plot(t, run["speed"], lw=0.9, label=f"ground-truth speed ({run['speed_src']})")
    if run["ref_speed"] is not None:
        a1.plot(t, run["ref_speed"], lw=0.7, color="tab:orange", alpha=0.7,
                label="reference speed")
    gt_ = gate_times[:, :-1]
    ok = ~np.isnan(gt_)
    a1.plot(gt_[ok], np.interp(gt_[ok], t, run["speed"]), "k.", ms=4, label="gate crossing")
    for k, ts in enumerate(gate_times[:, 0]):
        for a in (a1, a2):
            a.axvline(ts, color="gray", ls="--", lw=0.7)
        a1.text(ts, 0.98, f" lap {k + 1}", transform=a1.get_xaxis_transform(),
                ha="left", va="top", fontsize=8)
    a1.set_ylabel("speed [m/s]"); a1.grid(alpha=0.3); a1.legend(fontsize=8, loc="lower right")
    a1.set_title("Speed throughout the race (dashed lines = start of each lap)")
    a2.plot(t, run["track_err"], color="tab:red", lw=0.8)
    a2.set_ylabel("|reference − ground truth| [m]"); a2.set_xlabel("time [s]")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ----- 6.2 3D trajectory with the gates
def gate_corners(c, h, half):
    _, u = gate_axes(h)
    z = np.array([0, 0, 1.0])
    pts = [c - half * u - half * z, c + half * u - half * z,
           c + half * u + half * z, c - half * u + half * z]
    return np.array(pts + [pts[0]])


def plot_trajectory_3d(run, gate_times, gates, path, dpi):
    t, xyz = run["t"], run["xyz"]
    n_laps = gate_times.shape[0]
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    m0 = t <= gate_times[0, 0]
    if m0.sum() > 1:
        ax.plot(*xyz[m0].T, color="gray", ls="--", lw=1.2, label="approach")
    cmap = plt.get_cmap("tab10")
    for k in range(n_laps):
        a = gate_times[k, 0]
        b = gate_times[k, -1] if not np.isnan(gate_times[k, -1]) else t[-1]
        m = (t >= a) & (t <= b)
        ax.plot(*xyz[m].T, color=cmap(k % 10), lw=1.2,
                label=f"lap {k + 1}" + (" (partial)" if np.isnan(gate_times[k, -1]) else ""))

    all_pts = [xyz]
    for gid, c, h in zip(gates["ids"], gates["centers"], gates["headings"]):
        n, _ = gate_axes(h)
        inner = gate_corners(c, h, gates["aperture"] / 2)
        outer = gate_corners(c, h, gates["outer"] / 2)
        ax.plot(*outer.T, color="black", lw=1.5)
        ax.plot(*inner.T, color="black", lw=1.0)
        ax.quiver(*c, *(0.35 * n), color="black", arrow_length_ratio=0.3, lw=1)
        ax.text(c[0], c[1], c[2] + 0.45, f"G{gid}", fontsize=9, fontweight="bold", ha="center")
        all_pts += [outer, c[None, :] + 0.35 * n]
    P = np.vstack(all_pts)
    ax.set_box_aspect(np.maximum(np.ptp(P, axis=0), 1e-3))   # equal scale on all axes
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=28, azim=-60)
    ax.set_title("3D trajectory and gates (arrows = crossing direction)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ----- 6.3 Segment times per lap (stacked bars)
def plot_segment_bars(segs, laps, path, dpi):
    st = segs["time"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(st))
    cmap = plt.get_cmap("Set2")
    labels = [f"lap {i}" + ("*" if laps.loc[i, "status"] == "partial" else "") for i in st.index]
    for j, col in enumerate(st.columns):
        vals = np.nan_to_num(st[col].to_numpy())
        ax.bar(labels, vals, bottom=bottom, color=cmap(j % 8), edgecolor="k", label=col)
        bottom += vals
    for k, i in enumerate(st.index):
        if laps.loc[i, "status"] != "partial":
            ax.text(k, bottom[k] + 0.05, f"{bottom[k]:.1f} s", ha="center", fontsize=8)
    ax.set_ylabel("time [s]"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Lap time split by gate-to-gate segment (* = partial lap)")
    ax.legend(title="segment", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ----- 6.4 Gate grid: crossing point of every lap (rows) through every gate (columns)
def plot_gate_grid(run, table, n_laps, gates, window, drone_diam, path_stem, dpi,
                   lim=0.34):
    t, xyz = run["t"], run["xyz"]
    ids = gates["ids"]
    n_g = len(ids)
    half_in, half_out = gates["aperture"] / 2, gates["outer"] / 2
    m = int(round(half_in / 0.05))                 # 5 cm steps from the centre to the aperture edge
    red, pink = "red", "#ffa0a0"

    fig, axes = plt.subplots(n_laps, n_g, figsize=(1.8 * n_g, 2.0 * n_laps), squeeze=False)
    for k in range(n_laps):
        for j, (gid, c, h) in enumerate(zip(ids, gates["centers"], gates["headings"])):
            ax = axes[k, j]
            n, u = gate_axes(h)

            # white gate with black contour (outer frame and inner aperture)
            ax.add_patch(Rectangle((-half_out, -half_out), 2 * half_out, 2 * half_out,
                                   fc="white", ec="black", lw=1.5, zorder=2))
            ax.add_patch(Rectangle((-half_in, -half_in), 2 * half_in, 2 * half_in,
                                   fc="white", ec="black", lw=1.5, zorder=2))
            # grid only inside the aperture (10 cm major, 5 cm minor)
            for i in range(-m + 1, m):
                major = (i % 2 == 0)
                kw = dict(color="gray" if major else "lightgray",
                          lw=0.6 if major else 0.3, zorder=2.5)
                v = i * 0.05
                ax.plot([v, v], [-half_in, half_in], **kw)
                ax.plot([-half_in, half_in], [v, v], **kw)

            e = table.get((k, gid))
            if e is None:
                ax.text(0, 0, "no crossing", ha="center", va="center",
                        color="gray", fontsize=8, zorder=5)
                ax.set_xlabel("–", fontsize=7, labelpad=2)
            else:
                # path near the gate plane, seen from behind (as the pilot sees it)
                s = (xyz - c) @ n
                sel = (np.abs(s) <= window) & (np.abs(t - e["time"]) < 1.0)
                P = xyz[sel] - c
                x, z = -(P @ u), P[:, 2]
                before = t[sel] <= e["time"]
                cx, cz = -e["lateral"], e["vertical"]
                ax.plot(np.r_[x[before], cx], np.r_[z[before], cz], "-", color=red, lw=0.7, zorder=3)
                ax.plot(np.r_[cx, x[~before]], np.r_[cz, z[~before]], "-", color=pink, lw=0.7, zorder=3)
                ax.plot(x[before], z[before], ".", color=red, ms=2, zorder=3)
                ax.plot(x[~before], z[~before], ".", color=pink, ms=2, zorder=3)
                if drone_diam:
                    ax.add_patch(Circle((cx, cz), drone_diam / 2, fc=(1, 0, 0, 0.2),
                                        ec=red, lw=0.8, zorder=4))
                ax.plot(cx, cz, "o", color=red, ms=5, zorder=6)
                lab = f"x {cx * 100:+.1f}, z {cz * 100:+.1f} cm" + ("" if e["valid"] else " (outside)")
                ax.set_xlabel(lab, fontsize=7, labelpad=2, color="black" if e["valid"] else red)

            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k == 0:
                ax.set_title(f"Gate {gid}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Lap {k + 1}", fontsize=10)

    fig.suptitle("Crossing points relative to each gate (view in the flight direction; "
                 "grid 10 cm, minor 5 cm)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975), w_pad=0.2, h_pad=0.5)
    fig.savefig(f"{path_stem}.png", dpi=dpi)
    fig.savefig(f"{path_stem}.pdf")                # vector version
    plt.close(fig)


# =============================================================================
# 7. MARKDOWN REPORT
# =============================================================================
def df_to_md(df, fmt="{:.2f}", index_label=""):
    """Minimal DataFrame -> Markdown table (no extra dependencies)."""
    def cell(x):
        if isinstance(x, (bool, np.bool_)):
            return "yes" if x else "no"
        if isinstance(x, (float, np.floating)):
            return "–" if np.isnan(x) else fmt.format(x)
        if isinstance(x, (list, tuple)):
            return ", ".join(map(str, x)) if len(x) else "–"
        return str(x)
    lines = ["| " + " | ".join([index_label] + [str(c) for c in df.columns]) + " |",
             "|" + "---|" * (len(df.columns) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [cell(v) for v in row]) + " |")
    return "\n".join(lines)


def with_summary(df, regular):
    """Append mean/min/max rows computed over the regular laps only."""
    reg = df[regular.to_numpy()]
    extra = pd.DataFrame([reg.mean(), reg.min(), reg.max()],
                         index=["mean (regular)", "min (regular)", "max (regular)"])
    return pd.concat([df, extra])


def write_report(out, args, run, gates, laps, segs, rejected, gate_times):
    reg_mask = laps["status"] == "regular"
    reg = laps[reg_mask]
    L = []                                          # list of Markdown lines
    add = L.append

    add(f"# Race report: `{Path(args.csv).name}`\n")
    add(f"_Generated on {datetime.now():%Y-%m-%d %H:%M}_\n")

    # ---- 7.1 overview
    add("## 1. Overview\n")
    add(f"- Duration of the ground-truth recording: **{run['t'][-1]:.1f} s** "
        f"({run['n_gt']} ground-truth samples, about {run['rate']:.0f} Hz)")
    add(f"- Laps found (gate {gates['ids'][0]} → gate {gates['ids'][0]}): **{len(laps)}**, "
        f"of which regular: **{int(reg_mask.sum())}**, "
        f"irregular: {int((laps['status'] == 'irregular').sum())}, "
        f"partial: {int((laps['status'] == 'partial').sum())}")
    add(f"- Speed computed from: {run['speed_src']}")
    r, rr = run["speed_scale"], run["ref_speed_scale"]
    if r != 1.0:
        extra = f" (reference speed: 1/{rr:.2f})" if rr != 1.0 else ""
        add(f"\n> **Warning:** the twist speeds and the timestamps disagree by a factor {r:.2f}, "
            f"so the speeds in this report were divided by {r:.2f} to match the recorded clock{extra}.  ")
        add(f"> All times are on the recorded clock; if the simulation ran slower than real time, "
            f"true speeds are about {r:.2f}x higher and lap times about {r:.2f}x shorter.\n")
    add(f"- Parameters: margin {args.margin} m, z-shift {args.z_shift} m, "
        f"aperture {gates['aperture']} m, gate z offset {GATE_Z_OFFSET:+.2f} m\n")

    # ---- 7.2 lap times
    add("## 2. Lap times\n")
    if len(reg):
        f, s = reg["time [s]"].idxmin(), reg["time [s]"].idxmax()
        add(f"- Fastest lap: **lap {f}**, {reg.loc[f, 'time [s]']:.2f} s")
        add(f"- Slowest lap: **lap {s}**, {reg.loc[s, 'time [s]']:.2f} s")
        add(f"- Average lap: **{reg['time [s]'].mean():.2f} s** (over {len(reg)} regular laps)\n")
    else:
        add("_No regular lap: statistics not available._\n")
    add(df_to_md(laps[["status", "time [s]", "length [m]", "missed gates"]], index_label="lap"))
    add("\n_Statistics use regular laps only (closed, all gates crossed). "
        "A partial lap ends with the recording; its time is not defined._\n")

    # ---- 7.3 speed per lap
    add("## 3. Speed per lap\n")
    add("`avg` = path length / lap time. `mean`, `max`, `min` are computed on the "
        "instantaneous speed samples inside the lap (the partial lap uses the samples available).\n")
    add(df_to_md(laps[["avg speed [m/s]", "mean speed [m/s]", "max speed [m/s]",
                       "min speed [m/s]"]], index_label="lap"))
    add("")

    # ---- 7.4 segments
    add("## 4. Gate-to-gate segments\n")
    add("Column `i→j` is the flight from gate i to gate j; the last one returns to gate 1. "
        "A dash means the segment is not available (gate missed, or lap not closed). "
        "Summary rows use regular laps only.\n")
    add("### 4.1 Segment time [s]\n")
    add(df_to_md(with_summary(segs["time"], reg_mask), index_label="lap"))
    add("\n### 4.2 Segment distance [m]\n")
    add(df_to_md(with_summary(segs["dist"], reg_mask), index_label="lap"))
    add("\n### 4.3 Segment average speed [m/s]\n")
    add(df_to_md(with_summary(segs["avg_speed"], reg_mask), index_label="lap"))
    add("\n### 4.4 Segment maximum speed [m/s]\n")
    add(df_to_md(with_summary(segs["max_speed"], reg_mask), index_label="lap"))
    add("\n### 4.5 Segment minimum speed [m/s]\n")
    add(df_to_md(with_summary(segs["min_speed"], reg_mask), index_label="lap"))
    add("\n![Segment times](figures/segment_times.png)\n")

    # ---- 7.5 speed at gates
    add("## 5. Speed at the gate crossings [m/s]\n")
    add(df_to_md(with_summary(segs["gate_speed"], reg_mask), index_label="lap"))
    add("")

    # ---- 7.6 tracking
    add("## 6. Speed profile and tracking error\n")
    add("Reference (controller) position interpolated onto the ground-truth timestamps.\n")
    add(df_to_md(laps[["track err mean [m]", "track err max [m]"]], fmt="{:.3f}",
                 index_label="lap"))
    add("\n![Speed profile](figures/speed_profile.png)\n")

    # ---- 7.7 trajectory
    add("## 7. 3D trajectory\n")
    add("![3D trajectory](figures/trajectory_3d.png)\n")

    # ---- 7.8 crossing points
    add("## 8. Crossing points through the gates\n")
    add("Rows are laps, columns are gates. Red = path before the crossing, pink = after; "
        "the solid dot is the interpolated crossing point, labelled with its offset from the "
        "gate centre (x right, z up, as seen by the pilot).\n")
    add("![Gate crossings](figures/gate_crossings.png)\n")
    add("Vector version: [gate_crossings.pdf](figures/gate_crossings.pdf)\n")
    if rejected:
        add("**Crossings outside the aperture:**\n")
        for e in rejected:
            add(f"- gate {e['gate']} at t = {e['time']:.2f} s "
                f"(lateral {e['lateral'] * 100:+.1f} cm, vertical {e['vertical'] * 100:+.1f} cm)")
        add("")

    # ---- 7.9 notes
    add("## 9. Notes and assumptions\n")
    add("- Gate order is the order in the gates file (assumed to be the race order).")
    add("- A crossing counts only in the gate's flight direction and inside the aperture "
        f"(half-aperture + {args.margin} m margin).")
    add("- Path length and speeds are 3D. Lap length is measured between interpolated crossing points.")
    add("- Ground truth and reference are published at different rates: the reference is "
        "linearly interpolated onto the ground-truth timestamps.")
    add("- The drone footprint in the gate grid is a visual aid, not measured data.\n")

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out = Path(args.out) if args.out else Path(f"report_{csv_path.stem}")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    # ---- 8.1 load and analyse
    print("Loading data...")
    run = load_run(csv_path, args.ns, args.z_shift)
    gates = load_gates()
    table, gate_times, rejected = build_laps(run["t"], run["xyz"], run["speed"],
                                             gates, args.margin)
    laps, segs = analyze_laps(run, gate_times, gates)

    # ---- 8.1b consistency check: twist speed vs positions/timestamps. If they disagree,
    #           rescale the twist speeds (one factor per run) so all tables agree.
    r = laps["twist/position ratio"].median()
    rr = laps["ref twist/position ratio"].median()
    if not np.isnan(r) and not 0.9 <= r <= 1.1:
        run["speed"] = run["speed"] / r
        run["speed_scale"] = r
        run["speed_src"] += f", rescaled by 1/{r:.2f}"
    if run["ref_speed"] is not None and not np.isnan(rr) and not 0.9 <= rr <= 1.1:
        run["ref_speed"] = run["ref_speed"] / rr
        run["ref_speed_scale"] = rr
    if run["speed_scale"] != 1.0:
        laps, segs = analyze_laps(run, gate_times, gates)
    print(f"{len(laps)} laps found, {int((laps['status'] == 'regular').sum())} regular, "
          f"{len(rejected)} crossings outside the aperture")

    # ---- 8.2 figures
    print("Drawing figures...")
    fig_dir = out / "figures"
    plot_speed_profile(run, gate_times, fig_dir / "speed_profile.png", args.dpi)
    plot_trajectory_3d(run, gate_times, gates, fig_dir / "trajectory_3d.png", args.dpi)
    plot_segment_bars(segs, laps, fig_dir / "segment_times.png", args.dpi)
    plot_gate_grid(run, table, len(laps), gates, args.window, args.drone_diam,
                   str(fig_dir / "gate_crossings"), args.dpi, lim=args.grid_lim)

    # ---- 8.3 tables and configuration
    laps.to_csv(out / "tables" / "laps.csv")
    for name, df in segs.items():
        df.to_csv(out / "tables" / f"segments_{name}.csv")
    (out / "config.json").write_text(json.dumps(
        {**vars(args), "n_laps": len(laps), "speed_source": run["speed_src"],
         "speed_scale": run["speed_scale"]}, indent=2))

    # ---- 8.4 report
    write_report(out, args, run, gates, laps, segs, rejected, gate_times)
    print(f"Done. Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()