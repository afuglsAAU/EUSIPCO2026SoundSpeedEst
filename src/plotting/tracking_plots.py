# plotting/tracking_plots.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..utils.simdata import load_results


def select_runs(
    df,
    group="one_lsp_one_mic",
    grid_mode="full_grid",
    input=None,
    schedule=None,
    direction=None,
    track_vast_rank=None,
):
    sel = (df["group"] == group)

    if grid_mode is not None:
        sel &= (df["grid_mode"] == grid_mode)

    if input is not None:
        sel &= (df["input"] == input)

    if schedule is not None:
        sel &= (df["schedule"] == schedule)

    if direction is not None:
        sel &= (df["direction"] == direction)

    if track_vast_rank is not None:
        sel &= (df["track_vast_rank"] == track_vast_rank)

    return df[sel].copy()



def plot_speed_tracking(row, ax=None, title=None):
    res = load_results(row)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(res["true_speed"], label="True speed", linewidth=2)
    ax.plot(res["est_speed"], label="Estimated speed", alpha=0.8)

    if res["speed_change_frames"] is not None:
        for f in res["speed_change_frames"]:
            ax.axvline(f, color="k", linestyle=":", alpha=0.3)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed")
    ax.legend()
    ax.grid(True)

    if title:
        ax.set_title(title)

    return ax



def plot_tracking_error(row, ax=None, signed=True, normalize_by_power=False):
    res = load_results(row)

    err = res["est_speed"] - res["true_speed"]
    if not signed:
        err = np.abs(err)

    if normalize_by_power and res["input_power"] is not None:
        err = err / (res["input_power"] + 1e-12)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(err)
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed error")
    ax.grid(True)

    return ax


def extract_event_aligned_errors(
    row,
    pre_frames=50,
    post_frames=150,
    signed=True,
):
    res = load_results(row)

    err = res["est_speed"] - res["true_speed"]
    if not signed:
        err = np.abs(err)

    events = res["speed_change_frames"]
    if events is None:
        return None

    segments = []
    for f in events:
        lo = max(0, f - pre_frames)
        hi = min(len(err), f + post_frames)
        seg = err[lo:hi]

        if len(seg) < pre_frames + post_frames:
            pad = pre_frames + post_frames - len(seg)
            seg = np.pad(seg, (0, pad), constant_values=np.nan)

        segments.append(seg)

    return np.vstack(segments)



def plot_event_aligned_envelope(
    rows,
    pre_frames=5,
    post_frames=5,
    signed=False,
    label=None,
    ax=None,
):
    all_segs = []

    for _, row in rows.iterrows():
        segs = extract_event_aligned_errors(
            row,
            pre_frames=pre_frames,
            post_frames=post_frames,
            signed=signed,
        )
        if segs is not None:
            all_segs.append(segs)

    if not all_segs:
        return None

    data = np.vstack(all_segs)
    t = np.arange(-pre_frames, post_frames)

    mean = np.nanmean(data, axis=0)
    p25 = np.nanpercentile(data, 25, axis=0)
    p75 = np.nanpercentile(data, 75, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t, mean, label=label)
    ax.fill_between(t, p25, p75, alpha=0.3)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Frames relative to speed change")
    ax.set_ylabel("Tracking error")
    ax.grid(True)

    return ax


def plot_error_distribution(rows, ax=None, normalize_by_power=False):
    vals = []

    for _, row in rows.iterrows():
        res = load_results(row)
        err = np.abs(res["est_speed"] - res["true_speed"])
        if normalize_by_power and res["input_power"] is not None:
            err = err / (res["input_power"] + 1e-12)
        vals.append(err)

    vals = np.concatenate(vals)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(vals, bins=50, density=True, alpha=0.7)
    ax.set_xlabel("Absolute tracking error")
    ax.set_ylabel("Density")
    ax.grid(True)

    return ax

def plot_AC_vs_error(rows, ax=None):
    xs, ys = [], []

    for _, row in rows.iterrows():
        res = load_results(row)
        if res["AC"] is None:
            continue
        err = np.abs(res["est_speed"] - res["true_speed"])
        xs.append(np.nanmean(err))
        ys.append(np.nanmean(res["AC"]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(xs, ys)
    ax.set_xlabel("Mean |speed error|")
    ax.set_ylabel("Mean AC")
    ax.grid(True)

    return ax
