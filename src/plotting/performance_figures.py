# paper_figures/performance_figures.py

import numpy as np
import matplotlib.pyplot as plt
import string
from typing import Optional, Any
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from ..utils.simdata import load_results
from .style import INPUT_NAMES

FILTER_LABELS = {
    "szc_no_update": "No correction",
    "szc_update": "SZC (update)",
    "szc_gt_baseline": "GT baseline",
    "szc_sicer_baseline": "SICER (oracle)",
}

FILTER_COLORS = {
    "szc_no_update": "tab:red",
    "szc_update": "tab:blue",
    "szc_gt_baseline": "black",
    "szc_sicer_baseline": "tab:olive",
}

FILTER_LINESTYLES = {
    "szc_no_update": "solid",
    "szc_update": "dashed",
    "szc_gt_baseline": "solid",
    "szc_sicer_baseline": "solid",
}

def schedule_direction_to_title(schedule, direction):
    split = schedule.split("_")
    change = split[1]
    delta = split[3]
    dir_map = {
        "up": "From 333m/s to 353 m/s",
        "down": "From 353m/s to 333 m/s",
    }
    dir_title = dir_map.get(direction, direction)
    return f"{dir_title} - Speed Change: {change}m/s, Δt={delta}s"




def _summarize_curves(curves):
    data = np.vstack(curves)
    mean = np.nanmean(data, axis=0)
    p25 = np.nanpercentile(data, 25, axis=0)
    p75 = np.nanpercentile(data, 75, axis=0)
    t = np.arange(len(mean))
    return t, mean, p25, p75


def _pick_one_row(df_sel, preferred_ranks=None):
    if df_sel is None or df_sel.empty:
        return None
    if preferred_ranks is not None and "track_vast_rank" in df_sel.columns:
        for r in preferred_ranks:
            hit = df_sel[df_sel["track_vast_rank"] == r]
            if not hit.empty:
                return hit.iloc[0]
    return df_sel.iloc[0]


def _compute_limits_excluding_events(arrays, events_idx):
    vals = []
    for arr in arrays:
        a = np.asarray(arr)
        m = np.isfinite(a)
        if events_idx is not None:
            for e in events_idx:
                if 0 <= e < a.shape[0]:
                    m[e] = False
        if np.any(m):
            vals.append(a[m])
    if not vals:
        return None
    cat = np.concatenate(vals)
    if cat.size == 0:
        return None
    vmin = float(np.nanmin(cat))
    vmax = float(np.nanmax(cat))
    span = vmax - vmin
    if span <= 0:
        pad = 0.05 * (abs(vmax) + 1.0)
    else:
        pad = 0.05 * span
    return (vmin - pad, vmax + pad)


def load_performance(row, metric="AC"):
    res = load_results(row)

    if metric == "AC":
        return res["AC"]
    elif metric == "nSDP":
        return res["nSDP"]
    else:
        raise ValueError(metric)


def compute_delta_c(row):
    res = load_results(row)
    true_speed = res["true_speed"]
    c0 = true_speed[0]
    return true_speed - c0


def fig_performance_vs_time(
    df,
    schedule,
    direction,
    input,
    track_vast_rank=None,
    save_path=None,
):
    # Three stacked subplots: AC, nSDP, speed tracking
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    ax_ac, ax_nsdp, ax_track = axes

    # Legend collection for a single figure-level legend
    label_to_handle = {}

    # Arrays used to compute y-limits excluding event frames
    ac_arrays_for_limits = []
    nsdp_arrays_for_limits = []
    events_indices = None

    # Helper: compute summary bands
    summarize_curves = _summarize_curves

    # Preferred ranks order when multiple available
    preferred_ranks = [4041, 8000, 1]
    pick_one_row = lambda df_sel: _pick_one_row(df_sel, preferred_ranks=preferred_ranks)

    # Zero-error reference line for tracking subplot
    ax_track.axhline(0.0, color="gray", linewidth=1, alpha=0.8)

    # Iterate filter modes
    events_drawn = False
    for filter_mode, label in FILTER_LABELS.items():
        base_sel = (
            (df["group"] == "multi_lsp_one_mic") &
            (df["apply_filter"] == True) &
            (df["filter_mode"] == filter_mode) &
            (df["schedule"] == schedule) &
            (df["direction"] == direction) &
            (df["input"] == input)
        )
        if track_vast_rank is not None and "track_vast_rank" in df.columns:
            base_sel &= (df["track_vast_rank"] == track_vast_rank)

        rows_all = df[base_sel]
        if rows_all.empty:
            continue

        color = FILTER_COLORS.get(filter_mode, "tab:gray")
        styles = ["--", ":", "-.", (0, (3, 1, 1, 1))]

        # Base row for event markers (draw once across figure)
        base_row = pick_one_row(rows_all)
        if base_row is not None:
            res_base = load_results(base_row)
            if not events_drawn:
                events = res_base.get("speed_change_frames", None)
                if events is not None:
                    # Store once for use in y-limit computation
                    try:
                        events_indices = [int(e) for e in events]
                    except Exception:
                        events_indices = list(events)
                    for f in events:
                        ax_track.axvline(f, color="k", linestyle=":", alpha=0.3)
                        ax_ac.axvline(f, color="k", linestyle=":", alpha=0.2)
                        ax_nsdp.axvline(f, color="k", linestyle=":", alpha=0.2)
                events_drawn = True

        if filter_mode == "szc_update":
            # Full grid performance (AC & nSDP)
            rows_fg = rows_all[rows_all["grid_mode"] == "full_grid"]
            if not rows_fg.empty:
                curves_ac = [load_performance(row, "AC") for _, row in rows_fg.iterrows()]
                curves_nsdp = [load_performance(row, "nSDP") for _, row in rows_fg.iterrows()]
                t, mean_ac, p25_ac, p75_ac = summarize_curves(curves_ac)
                _, mean_nsdp, p25_nsdp, p75_nsdp = summarize_curves(curves_nsdp)
                h1 = ax_ac.plot(t, mean_ac, label=f"{label} – full_grid", color=color, linestyle="-", linewidth=2)[0]
                ax_ac.fill_between(t, p25_ac, p75_ac, color=color, alpha=0.15)
                h2 = ax_nsdp.plot(t, mean_nsdp, label=f"{label} – full_grid", color=color, linestyle="-", linewidth=2)[0]
                ax_nsdp.fill_between(t, p25_nsdp, p75_nsdp, color=color, alpha=0.15)
                # Collect for y-limit computation
                ac_arrays_for_limits.extend([mean_ac, p25_ac, p75_ac])
                nsdp_arrays_for_limits.extend([mean_nsdp, p25_nsdp, p75_nsdp])
                for h, lab in [(h1, f"{label} – full_grid"), (h2, f"{label} – full_grid")]:
                    if lab not in label_to_handle:
                        label_to_handle[lab] = h

                # Tracking subplot: representative estimated error for full_grid
                row_fg_rep = pick_one_row(rows_fg)
                if row_fg_rep is not None:
                    res_fg = load_results(row_fg_rep)
                    err_fg = res_fg["est_speed"] - res_fg["true_speed"]
                    h = ax_track.plot(err_fg, label=f"Est – {label} full_grid", color=color, alpha=0.9, linestyle="-")[0]
                    # if h.get_label() not in label_to_handle:
                    #     label_to_handle[h.get_label()] = h

            # Adaptive window performance by width
            rows_aw = rows_all[rows_all["grid_mode"] == "adaptive_window"]
            if not rows_aw.empty and "search_window_width" in rows_aw.columns:
                widths = sorted([w for w in rows_aw["search_window_width"].dropna().unique().tolist()])
                for i, w in enumerate(widths):
                    rows_w = rows_aw[rows_aw["search_window_width"] == w]
                    if rows_w.empty:
                        continue
                    curves_ac = [load_performance(row, "AC") for _, row in rows_w.iterrows()]
                    curves_nsdp = [load_performance(row, "nSDP") for _, row in rows_w.iterrows()]
                    t, mean_ac, p25_ac, p75_ac = summarize_curves(curves_ac)
                    _, mean_nsdp, p25_nsdp, p75_nsdp = summarize_curves(curves_nsdp)
                    try:
                        w_txt = int(w) if float(w).is_integer() else w
                    except Exception:
                        w_txt = w
                    style = styles[i % len(styles)]
                    lab_aw = f"{label} – aw ww{w_txt}"
                    h1 = ax_ac.plot(t, mean_ac, label=lab_aw, color=color, linestyle=style)[0]
                    ax_ac.fill_between(t, p25_ac, p75_ac, color=color, alpha=0.10)
                    h2 = ax_nsdp.plot(t, mean_nsdp, label=lab_aw, color=color, linestyle=style)[0]
                    ax_nsdp.fill_between(t, p25_nsdp, p75_nsdp, color=color, alpha=0.10)
                    # Collect for y-limit computation
                    ac_arrays_for_limits.extend([mean_ac, p25_ac, p75_ac])
                    nsdp_arrays_for_limits.extend([mean_nsdp, p25_nsdp, p75_nsdp])
                    for h, lab in [(h1, lab_aw), (h2, lab_aw)]:
                        if lab not in label_to_handle:
                            label_to_handle[lab] = h

                    # Tracking subplot: representative estimated error for this width
                    row_w_rep = pick_one_row(rows_w)
                    if row_w_rep is not None:
                        res_w = load_results(row_w_rep)
                        lab_tr = f"Est – {label} aw ww{w_txt}"
                        err_w = res_w["est_speed"] - res_w["true_speed"]
                        h = ax_track.plot(err_w, label=lab_tr, color=color, alpha=0.8, linestyle=style)[0]
                        # if lab_tr not in label_to_handle:
                        #     label_to_handle[lab_tr] = h

        else:
            # Other filters aggregated (AC & nSDP)
            curves_ac = [load_performance(row, "AC") for _, row in rows_all.iterrows()]
            curves_nsdp = [load_performance(row, "nSDP") for _, row in rows_all.iterrows()]
            t, mean_ac, p25_ac, p75_ac = summarize_curves(curves_ac)
            _, mean_nsdp, p25_nsdp, p75_nsdp = summarize_curves(curves_nsdp)
            h1 = ax_ac.plot(t, mean_ac, label=label, color=color)[0]
            ax_ac.fill_between(t, p25_ac, p75_ac, color=color, alpha=0.25)
            h2 = ax_nsdp.plot(t, mean_nsdp, label=label, color=color)[0]
            ax_nsdp.fill_between(t, p25_nsdp, p75_nsdp, color=color, alpha=0.25)
            # Collect for y-limit computation
            ac_arrays_for_limits.extend([mean_ac, p25_ac, p75_ac])
            nsdp_arrays_for_limits.extend([mean_nsdp, p25_nsdp, p75_nsdp])
            for h, lab in [(h1, label), (h2, label)]:
                if lab not in label_to_handle:
                    label_to_handle[lab] = h

            # Tracking subplot: Is only for the proposed method, separate figure compares estimation for multi loudspeaker case
            #  representative estimated speed
            if filter_mode != "szc_gt_baseline":
                row_rep = pick_one_row(rows_all)
                if row_rep is not None:
                    res = load_results(row_rep)
                    lab_tr = f"Est – {label}"
                    err_no_update = res["est_speed"] - res["true_speed"]
                    h = ax_track.plot(err_no_update, label=lab_tr, color=color, alpha=0.9)[0]
                    # if lab_tr not in label_to_handle:
                    #     label_to_handle[lab_tr] = h

    # Labels and grid
    ax_track.set_xlabel("Frame")
    ax_ac.set_ylabel("AC")
    ax_nsdp.set_ylabel("nSDP")
    ax_track.set_ylabel("Speed error")
    for ax in axes:
        ax.grid(True)

    fig.suptitle(f"Performance & tracking – {schedule}, {direction}, {input}" + (f", rank {track_vast_rank}" if track_vast_rank is not None else ""))

    # Compute y-limits excluding event frames (exact indices)
    lim_ac = _compute_limits_excluding_events(ac_arrays_for_limits, events_indices)
    lim_nsdp = _compute_limits_excluding_events(nsdp_arrays_for_limits, events_indices)
    if lim_ac is not None:
        ax_ac.set_ylim(*lim_ac)
    if lim_nsdp is not None:
        ax_nsdp.set_ylim(*lim_nsdp)

    # Single common legend for all subplots
    handles = list(label_to_handle.values())
    labels = list(label_to_handle.keys())
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=min(3, len(labels)))
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    if save_path:
        fig.savefig(save_path)


def fig_performance_inputs_vs_filters(
    df,
    schedule,
    direction,
    track_vast_rank,
    save_path,
    *,
    inputs=("white", "speech", "rock", "speech_music"),
    filter_modes=("szc_gt_baseline", "szc_sicer_baseline", "szc_no_update", "szc_update"),
    adaptive_window_widths=None,
    apply_filter_only=True,
):
    """Combined performance figure.

    Creates one figure per (schedule, direction, rank).

    Layout:
      - Columns: input type
      - Rows: AC (top), nSDP (bottom)

    Each panel overlays:
      - GT baseline performance
      - SZC (no update) performance
      - SZC (update) performance for full_grid and selected adaptive-window widths
    """

    metrics = [("AC", "AC [dB]"), ("nSDP", "nSDP [dB]")]
    nrows = len(metrics)
    ncols = len(inputs)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        figsize=(3.6 * ncols, 2.0 * nrows),
        squeeze=False,
    )

    label_to_handle = {}
    preferred_ranks = [4041, 8000, 1]
    styles = [":", "--", "-.", (0, (3, 1, 1, 1))]
    events_indices = None
    events_drawn = False

    # For per-subplot y-limits (metric, input)
    # arrays_for_limits = {}

    for col_idx, inp in enumerate(inputs):
        for row_idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            arrays_for_limits = []

            for filter_mode in filter_modes:
                base_sel = (
                    (df["schedule"] == schedule)
                    & (df["direction"] == direction)
                    & (df["input"] == inp)
                )
                if "filter_mode" in df.columns:
                    base_sel &= (df["filter_mode"] == filter_mode)
                if apply_filter_only and "apply_filter" in df.columns:
                    base_sel &= (df["apply_filter"] == True)
                if track_vast_rank is not None and "track_vast_rank" in df.columns:
                    base_sel &= (df["track_vast_rank"] == track_vast_rank)

                rows_all = df[base_sel]
                if rows_all.empty:
                    continue

                # Event markers (draw once across all axes)
                if not events_drawn:
                    base_row = _pick_one_row(rows_all, preferred_ranks=preferred_ranks)
                    if base_row is not None:
                        res_base = load_results(base_row)
                        events = res_base.get("speed_change_frames", None)
                        if events is not None:
                            try:
                                events_indices = [int(e) for e in events]
                            except Exception:
                                events_indices = list(events)
                            for r in range(nrows):
                                for c in range(ncols):
                                    for f in events:
                                        axes[r][c].axvline(f, color="k", linestyle=":", alpha=0.15)
                    events_drawn = True

                color = FILTER_COLORS.get(filter_mode, "tab:gray")
                linestyle = FILTER_LINESTYLES.get(filter_mode, "solid")
                base_label = FILTER_LABELS.get(filter_mode, filter_mode)

                # Full grid (if present)
                rows_fg = rows_all
                if "grid_mode" in rows_all.columns:
                    rows_fg = rows_all[rows_all["grid_mode"] == "full_grid"]
                if not rows_fg.empty:
                    curves = [load_performance(row, metric_key) for _, row in rows_fg.iterrows()]
                    curves = [c for c in curves if c is not None]
                    if curves:
                        t, mean, p25, p75 = _summarize_curves(curves)
                        lab = base_label if filter_mode != "szc_update" else f"Prop. – full grid"
                        h = ax.plot(t, mean, label=lab, color=color, linestyle=linestyle)[0]
                        ax.fill_between(t, p25, p75, color=color, alpha=0.12)
                        label_to_handle.setdefault(lab, h)
                        arrays_for_limits.extend([mean, p25, p75])

                # Adaptive window curves by width (if present)
                rows_aw = rows_all
                if "grid_mode" in rows_all.columns:
                    rows_aw = rows_all[rows_all["grid_mode"] == "adaptive_window"]
                if not rows_aw.empty and "search_window_width" in rows_aw.columns:
                    widths = sorted([w for w in rows_aw["search_window_width"].dropna().unique().tolist()])
                    if adaptive_window_widths is not None and widths:
                        if callable(adaptive_window_widths):
                            widths = [w for w in widths if adaptive_window_widths(w)]
                        else:
                            try:
                                allowed = set(adaptive_window_widths)
                            except TypeError:
                                allowed = {adaptive_window_widths}
                            widths = [w for w in widths if w in allowed]

                    for i, w in enumerate(widths):
                        rows_w = rows_aw[rows_aw["search_window_width"] == w]
                        if rows_w.empty:
                            continue
                        curves = [load_performance(row, metric_key) for _, row in rows_w.iterrows()]
                        curves = [c for c in curves if c is not None]
                        if not curves:
                            continue
                        t, mean, p25, p75 = _summarize_curves(curves)
                        try:
                            w_txt = int(w) if float(w).is_integer() else w
                        except Exception:
                            w_txt = w
                        style = styles[i % len(styles)]
                        lab = f"Prop. – aw ww{w_txt}"
                        h = ax.plot(t, mean, label=lab, color=color, linestyle=style)[0]
                        ax.fill_between(t, p25, p75, color=color, alpha=0.08)
                        label_to_handle.setdefault(lab, h)
                        arrays_for_limits.extend([mean, p25, p75])

            # Titles/labels per subplot
            if row_idx == 0:
                ax.set_title(INPUT_NAMES.get(inp, inp))
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            else:
                ax.set_ylabel("")
            if row_idx == nrows - 1:
                ax.set_xlabel("Frame")
            lim = _compute_limits_excluding_events(arrays_for_limits, events_indices)
            ax.set_ylim(lim if lim is not None else (0, 1))
            if track_vast_rank == 1 and metric_key == "nSDP":
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.grid(True)


    # # Apply y-limits per subplot (avoid sharey)
    # for col_idx, inp in enumerate(inputs):
    #     for row_idx, (metric_key, _) in enumerate(metrics):
    #         lim = _compute_limits_excluding_events(arrays_for_limits.get((metric_key, inp), []), events_indices)
    #         if lim is not None:
    #             axes[row_idx][col_idx].set_ylim(*lim)

    title = f"Performance – {schedule}, {direction}, rank {track_vast_rank}"
    fig.suptitle(title)

    handles = list(label_to_handle.values())
    labels = list(label_to_handle.keys())
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            bbox_to_anchor=(0.5, 0.94),
        )

    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(save_path)
    plt.close(fig)


def fig_performance_inputs_vs_filters_ranks_combined(
    df,
    schedule,
    direction,
    ranks,
    save_path,
    *,
    inputs=("white", "speech", "rock", "speech_music"),
    filter_modes=("szc_gt_baseline", "szc_sicer_baseline", "szc_no_update", "szc_update"),
    metrics_by_rank=None,
    adaptive_window_widths=None,
    apply_filter_only=True,
    legend_on_first_rank_only=True,
    grid_wspace=0.0,
    grid_hspace=0.08,
    title_row_height_ratio=0.12,
    title_text_y=0.02,
    cl_w_pad=0.01,
    cl_h_pad=0.01,
    cl_wspace=0.01,
    cl_hspace=0.01,
    suptitle_y=1.02,
    legend_y=1.02,
    col_width=3.6,
    row_height=1.2,
    row_title_height=0.2,
    draw_rank_separators=True,
    rank_separator_lw=0.6,
    rank_separator_alpha=0.35,
    rank_separator_color="k",
):
    """Combine multiple ranks into one vertically-stacked figure.

    Each rank becomes a "sub-figure" block stacked row-wise.
    Within each block, the layout matches `fig_performance_inputs_vs_filters`:
      - Columns: input types
      - Rows: selected performance metrics (can vary per rank)

    Titles:
      - One per rank block: "(a) rank <r>", "(b) rank <r>", ...

    Legend:
      - A single legend, optionally derived from the first rank block.

    Parameters
    ----------
    metrics_by_rank : dict[int, tuple[str] | list[str]] | None
        Optional mapping {rank: ("AC", "nSDP", ...)} to selectively plot different
        metrics for different ranks. If None, plots ("AC", "nSDP") for all ranks.
    """

    def _metric_label(metric_key: str) -> str:
        if metric_key == "AC":
            return "AC [dB]"
        if metric_key == "nSDP":
            return "nSDP [dB]"
        return str(metric_key)

    ranks = list(ranks)
    if not ranks:
        fig, _ = plt.subplots()
        fig.savefig(save_path)
        plt.close(fig)
        return

    default_metric_keys = ("AC", "nSDP")
    metrics_by_rank = metrics_by_rank or {}

    metric_keys_per_rank: list[list[str]] = []
    for r in ranks:
        keys = metrics_by_rank.get(int(r), default_metric_keys)
        if isinstance(keys, str):
            keys = (keys,)
        metric_keys_per_rank.append([str(k) for k in keys])

    ncols = len(inputs)

    # Build one global grid so every subplot column has identical width.
    # Rows are: [rank title] + [metric row]* for each rank.
    row_specs = []  # list[tuple[str, int, Optional[str]]]
    height_ratios = []
    for rank_idx, metric_keys in enumerate(metric_keys_per_rank):
        row_specs.append(("title", rank_idx, None))
        height_ratios.append(float(title_row_height_ratio))
        for m in metric_keys:
            row_specs.append(("metric", rank_idx, m))
            height_ratios.append(1.0)

    total_metric_rows = int(sum(len(keys) for keys in metric_keys_per_rank))
    fig_w = col_width * max(1, ncols)
    fig_h = row_height * max(1, total_metric_rows) + row_title_height * len(ranks)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    _set_pads = getattr(fig, "set_constrained_layout_pads", None)
    if callable(_set_pads):
        _set_pads(
            w_pad=float(cl_w_pad),
            h_pad=float(cl_h_pad),
            wspace=float(cl_wspace),
            hspace=float(cl_hspace),
        )

    gs = fig.add_gridspec(
        nrows=len(row_specs),
        ncols=ncols,
        height_ratios=height_ratios,
        wspace=float(grid_wspace),
        hspace=float(grid_hspace),
    )

    global_label_to_handle: dict[str, Axes] = {}
    events_indices_global = None
    styles = [":", "--", "-.", (0, (3, 1, 1, 1))]
    preferred_ranks = [4041, 8000, 1]

    master_x: Optional[Axes] = None
    axes_by_rank_metric = {}
    title_axes_by_rank: dict[int, Axes] = {}

    # Create all axes first (ensures consistent widths and global sharex)
    row_ptr = 0
    for rank_idx, (track_vast_rank, metric_keys) in enumerate(zip(ranks, metric_keys_per_rank)):
        # Title axis spanning full width
        title_ax = fig.add_subplot(gs[row_ptr, :])
        title_ax.axis("off")
        title_axes_by_rank[rank_idx] = title_ax
        letter = string.ascii_lowercase[rank_idx % 26]
        title_ax.text(
            0.0,
            float(title_text_y),
            f"({letter}) rank V={int(track_vast_rank)}",
            ha="left",
            va="bottom",
        )
        row_ptr += 1

        for metric_row_idx, metric_key in enumerate(metric_keys):
            for col_idx, inp in enumerate(inputs):
                if master_x is None:
                    ax = fig.add_subplot(gs[row_ptr, col_idx])
                    master_x = ax
                else:
                    ax = fig.add_subplot(gs[row_ptr, col_idx], sharex=master_x)
                axes_by_rank_metric[(rank_idx, metric_row_idx, col_idx)] = ax
            row_ptr += 1

    # Plot per-axis data
    for rank_idx, (track_vast_rank, metric_keys) in enumerate(zip(ranks, metric_keys_per_rank)):
        label_to_handle = {}

        for metric_row_idx, metric_key in enumerate(metric_keys):
            for col_idx, inp in enumerate(inputs):
                ax = axes_by_rank_metric[(rank_idx, metric_row_idx, col_idx)]
                arrays_for_limits = []

                for filter_mode in filter_modes:
                    base_sel = (
                        (df["schedule"] == schedule)
                        & (df["direction"] == direction)
                        & (df["input"] == inp)
                    )
                    if "filter_mode" in df.columns:
                        base_sel &= (df["filter_mode"] == filter_mode)
                    if apply_filter_only and "apply_filter" in df.columns:
                        base_sel &= (df["apply_filter"] == True)
                    if track_vast_rank is not None and "track_vast_rank" in df.columns:
                        base_sel &= (df["track_vast_rank"] == track_vast_rank)

                    rows_all = df[base_sel]
                    if rows_all.empty:
                        continue

                    # Detect event markers once (global)
                    if events_indices_global is None:
                        base_row = _pick_one_row(rows_all, preferred_ranks=preferred_ranks)
                        if base_row is not None:
                            res_base = load_results(base_row)
                            events = res_base.get("speed_change_frames", None)
                            if events is not None:
                                try:
                                    events_indices_global = [int(e) for e in events]
                                except Exception:
                                    events_indices_global = list(events)

                    color = FILTER_COLORS.get(filter_mode, "tab:gray")
                    linestyle = FILTER_LINESTYLES.get(filter_mode, "solid")
                    base_label = FILTER_LABELS.get(filter_mode, filter_mode)

                    # Full grid (if present)
                    rows_fg = rows_all
                    if "grid_mode" in rows_all.columns:
                        rows_fg = rows_all[rows_all["grid_mode"] == "full_grid"]
                    if not rows_fg.empty:
                        curves = [load_performance(row, metric_key) for _, row in rows_fg.iterrows()]
                        curves = [c for c in curves if c is not None]
                        if curves:
                            t, mean, p25, p75 = _summarize_curves(curves)
                            lab = base_label if filter_mode != "szc_update" else "Prop. – full grid"
                            h = ax.plot(t, mean, label=lab, color=color, linestyle=linestyle)[0]
                            ax.fill_between(t, p25, p75, color=color, alpha=0.12)
                            label_to_handle.setdefault(lab, h)
                            arrays_for_limits.extend([mean, p25, p75])

                    # Adaptive window curves (if present)
                    rows_aw = rows_all
                    if "grid_mode" in rows_all.columns:
                        rows_aw = rows_all[rows_all["grid_mode"] == "adaptive_window"]
                    if not rows_aw.empty and "search_window_width" in rows_aw.columns:
                        widths = sorted([w for w in rows_aw["search_window_width"].dropna().unique().tolist()])
                        if adaptive_window_widths is not None and widths:
                            if callable(adaptive_window_widths):
                                widths = [w for w in widths if adaptive_window_widths(w)]
                            else:
                                try:
                                    allowed = set(adaptive_window_widths)
                                except TypeError:
                                    allowed = {adaptive_window_widths}
                                widths = [w for w in widths if w in allowed]

                        for i, w in enumerate(widths):
                            rows_w = rows_aw[rows_aw["search_window_width"] == w]
                            if rows_w.empty:
                                continue
                            curves = [load_performance(row, metric_key) for _, row in rows_w.iterrows()]
                            curves = [c for c in curves if c is not None]
                            if not curves:
                                continue
                            t, mean, p25, p75 = _summarize_curves(curves)
                            try:
                                w_txt = int(w) if float(w).is_integer() else w
                            except Exception:
                                w_txt = w
                            style = styles[i % len(styles)]
                            lab = f"Prop. – aw ww{w_txt}"
                            h = ax.plot(t, mean, label=lab, color=color, linestyle=style)[0]
                            ax.fill_between(t, p25, p75, color=color, alpha=0.08)
                            label_to_handle.setdefault(lab, h)
                            arrays_for_limits.extend([mean, p25, p75])

                # Titles/labels per subplot
                if metric_row_idx == 0:
                    ax.set_title(INPUT_NAMES.get(inp, inp))
                if col_idx == 0:
                    ax.set_ylabel(_metric_label(metric_key))
                else:
                    ax.set_ylabel("")

                # X label only on the last metric row of each rank block
                is_last_metric_row_in_rank = (metric_row_idx == len(metric_keys) - 1)
                if is_last_metric_row_in_rank:
                    ax.set_xlabel("Frame")
                    ax.tick_params(labelbottom=True)
                else:
                    ax.set_xlabel("")
                    ax.tick_params(labelbottom=False)

                # Draw event markers if we have them
                if events_indices_global is not None:
                    for f in events_indices_global:
                        ax.axvline(f, color="k", linestyle=":", alpha=0.15)

                lim = _compute_limits_excluding_events(arrays_for_limits, events_indices_global)
                if lim is not None:
                    ax.set_ylim(*lim)
                if track_vast_rank == 1 and metric_key == "nSDP":
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.grid(True)

        # Stash legend entries (optionally only from first rank)
        if not legend_on_first_rank_only or rank_idx == 0:
            for lab, h in label_to_handle.items():
                global_label_to_handle.setdefault(lab, h)

    handles = list(global_label_to_handle.values())
    labels = list(global_label_to_handle.keys())
    legend = None
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            bbox_to_anchor=(0.5, float(legend_y)),
        )
        # Keep legend position fixed (don't let constrained_layout shift it)
        _leg_set_in_layout = getattr(legend, "set_in_layout", None)
        if callable(_leg_set_in_layout):
            _leg_set_in_layout(False)

    # With constrained_layout, `suptitle` participates in layout and can push
    # other figure-level artists (like the legend) around. Use figure-level text
    # instead so moving the title doesn't move the legend.

    title_artist = fig.text(
        0.5,
        float(suptitle_y),
        f"Performance – {schedule_direction_to_title(schedule, direction)}",
        ha="center",
        va="bottom",
        transform=fig.transFigure,
    )
    _set_in_layout = getattr(title_artist, "set_in_layout", None)
    if callable(_set_in_layout):
        _set_in_layout(False)

    # Optional: draw subtle horizontal separators between rank blocks.
    if draw_rank_separators and len(ranks) > 1:
        try:
            fig.canvas.draw()
        except Exception:
            # If a backend doesn't support drawing here, saving will still draw.
            pass

        renderer: Any = None
        _get_renderer = getattr(fig.canvas, "get_renderer", None)
        if callable(_get_renderer):
            try:
                renderer = _get_renderer()
            except Exception:
                renderer = None

        inv_fig = fig.transFigure.inverted()

        for rank_idx in range(len(ranks) - 1):
            last_metric_row_idx = len(metric_keys_per_rank[rank_idx]) - 1
            if last_metric_row_idx < 0:
                continue

            left_ax = axes_by_rank_metric.get((rank_idx, last_metric_row_idx, 0))
            right_ax = axes_by_rank_metric.get((rank_idx, last_metric_row_idx, ncols - 1))
            next_title_ax = title_axes_by_rank.get(rank_idx + 1)
            if left_ax is None or right_ax is None or next_title_ax is None:
                continue

            # Use tight bounding boxes so the separator doesn't overlap x tick
            # labels / x-axis labels of the last metric row.
            if renderer is not None:
                bb_last_disp = None
                bb_next_disp = None
                try:
                    bb_last_disp = left_ax.get_tightbbox(renderer)
                    bb_next_disp = next_title_ax.get_tightbbox(renderer)
                except Exception:
                    bb_last_disp = None
                    bb_next_disp = None

                if bb_last_disp is not None and bb_next_disp is not None:
                    bb_last_tight = inv_fig.transform_bbox(bb_last_disp)
                    bb_next_title = inv_fig.transform_bbox(bb_next_disp)
                    y_top = float(bb_last_tight.y0)  # bottom of last row's labels
                    y_bot = float(bb_next_title.y1)  # top of next title row
                else:
                    pos_last = left_ax.get_position()
                    pos_next_title = next_title_ax.get_position()
                    y_top = float(pos_last.y0)
                    y_bot = float(pos_next_title.y1)
            else:
                pos_last = left_ax.get_position()
                pos_next_title = next_title_ax.get_position()
                y_top = float(pos_last.y0)
                y_bot = float(pos_next_title.y1)

            # Place the line centered in the gap between the bottom of the last
            # row's labels and the top of the next title row.
            y = 0.5 * (y_top + y_bot)

            x0 = float(left_ax.get_position().x0)
            x1 = float(right_ax.get_position().x1)
            sep = Line2D(
                [x0, x1],
                [y, y],
                transform=fig.transFigure,
                color=rank_separator_color,
                linewidth=float(rank_separator_lw),
                alpha=float(rank_separator_alpha),
                solid_capstyle="butt",
            )
            _sep_set_in_layout = getattr(sep, "set_in_layout", None)
            if callable(_sep_set_in_layout):
                _sep_set_in_layout(False)
            fig.add_artist(sep)

    # If title/legend are placed outside the [0,1] figure box, avoid clipping.
    # Note: bbox_inches='tight' ignores artists with in_layout=False unless they
    # are provided via bbox_extra_artists.
    extra_artists: list[object] = [title_artist]
    if legend is not None:
        extra_artists.append(legend)

    max_y = max(float(suptitle_y), float(legend_y))
    min_y = min(float(suptitle_y), float(legend_y))
    if min_y < 0.0 or max_y > 1.0:
        fig.savefig(save_path, bbox_inches="tight", bbox_extra_artists=extra_artists)
    else:
        # Still pass bbox_extra_artists so small overflows can be retained when
        # backends decide to clip figure-level text.
        fig.savefig(save_path, bbox_extra_artists=extra_artists)
    plt.close(fig)




def fig_performance_vs_delta_c(
    df,
    schedule,
    direction,
    input,
    metric="AC",
    track_vast_rank=None,
    save_path=None,
):
    fig, ax = plt.subplots()

    # Collect curves per label to share global binning
    curves_by_label = {}
    style_by_label = {}
    x_all = []

    for filter_mode, base_label in FILTER_LABELS.items():
        base_sel = (
            (df["group"] == "multi_lsp_one_mic") &
            (df["apply_filter"] == True) &
            (df["filter_mode"] == filter_mode) &
            (df["schedule"] == schedule) &
            (df["direction"] == direction) &
            (df["input"] == input)
        )
        if track_vast_rank is not None and "track_vast_rank" in df.columns:
            base_sel &= (df["track_vast_rank"] == track_vast_rank)

        rows_all = df[base_sel]
        if rows_all.empty:
            continue

        color = FILTER_COLORS.get(filter_mode, "tab:gray")
        styles = ["-", "--", ":", "-."]

        if filter_mode == "szc_update":
            # Full grid curve
            rows_fg = rows_all[rows_all["grid_mode"] == "full_grid"]
            if not rows_fg.empty:
                xs, ys = [], []
                for _, row in rows_fg.iterrows():
                    perf = load_performance(row, metric)
                    delta_c = np.abs(compute_delta_c(row))
                    xs.append(delta_c)
                    ys.append(perf)
                label = f"{base_label} – full_grid"
                curves_by_label[label] = (np.concatenate(xs), np.concatenate(ys))
                style_by_label[label] = {"color": color, "linestyle": styles[0], "marker": "o"}
                x_all.append(curves_by_label[label][0])

            # Adaptive window curves by width
            rows_aw = rows_all[rows_all["grid_mode"] == "adaptive_window"]
            if not rows_aw.empty and "search_window_width" in rows_aw.columns:
                widths = sorted([w for w in rows_aw["search_window_width"].dropna().unique().tolist()])
                for i, w in enumerate(widths):
                    rows_w = rows_aw[rows_aw["search_window_width"] == w]
                    if rows_w.empty:
                        continue
                    xs, ys = [], []
                    for _, row in rows_w.iterrows():
                        perf = load_performance(row, metric)
                        delta_c = np.abs(compute_delta_c(row))
                        xs.append(delta_c)
                        ys.append(perf)
                    try:
                        w_txt = int(w) if float(w).is_integer() else w
                    except Exception:
                        w_txt = w
                    label = f"{base_label} – aw ww{w_txt}"
                    curves_by_label[label] = (np.concatenate(xs), np.concatenate(ys))
                    style_by_label[label] = {"color": color, "linestyle": styles[(i + 1) % len(styles)], "marker": "o"}
                    x_all.append(curves_by_label[label][0])
        else:
            # Aggregate across modes
            xs, ys = [], []
            for _, row in rows_all.iterrows():
                perf = load_performance(row, metric)
                delta_c = np.abs(compute_delta_c(row))
                xs.append(delta_c)
                ys.append(perf)
            label = base_label
            curves_by_label[label] = (np.concatenate(xs), np.concatenate(ys))
            style_by_label[label] = {"color": color, "linestyle": styles[0], "marker": "o"}
            x_all.append(curves_by_label[label][0])

    # Global binning across all curves for consistent x-axis
    if not x_all:
        # Nothing to plot
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        return
    x_concat = np.concatenate([np.asarray(x) for x in x_all])
    x_concat = x_concat[np.isfinite(x_concat)]
    max_x = float(np.nanmax(x_concat)) if x_concat.size > 0 else 0.0
    if max_x <= 0:
        max_x = 1e-6
    bins = np.linspace(0.0, max_x, 20)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot each curve binned by Δc
    for label, (x, y) in curves_by_label.items():
        x = np.asarray(x)
        y = np.asarray(y)
        # Guard against mismatched lengths or NaNs
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        means = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            sel = (x >= b0) & (x < b1)
            if not np.any(sel):
                means.append(np.nan)
            else:
                means.append(np.nanmean(y[sel]))
        st = style_by_label.get(label, {"color": "tab:gray", "linestyle": "-", "marker": "o"})
        ax.plot(centers, means, label=label, color=st["color"], linestyle=st["linestyle"], marker=st["marker"])

    ax.set_xlabel("|Δ sound speed|")
    ax.set_ylabel(metric)
    title_rank = f", rank {track_vast_rank}" if track_vast_rank is not None else ""
    ax.set_title(f"{metric} vs |Δc| – {schedule}, {direction}, {input}{title_rank}")
    ax.legend()
    ax.grid(True)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
