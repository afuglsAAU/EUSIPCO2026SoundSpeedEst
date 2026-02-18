# paper_figures/tracking_figures.py
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from scipy import signal
from scipy.signal import windows
from .style import INPUT_COLORS, INPUT_NAMES
from .tracking_plots import (plot_event_aligned_envelope, plot_error_distribution,
                             plot_AC_vs_error)
from ..utils.simdata import load_results

SPEED_TRACK_COLORS = {
    "true": "deepskyblue",
    "full_grid": "red",
    "aw_3": "black",
    # "szc_sicer_baseline": "tab:orange",
    "stft_color": "Greys",
}



def _format_lsp_count(lsp_count):
    if lsp_count is None:
        return None
    try:
        n = int(lsp_count)
    except Exception:
        return str(lsp_count)
    return "1 active LSP" if n == 1 else f"{n} active LSPs"


def _pick_one_row(df_sel, preferred_ranks=None):
    if df_sel is None or df_sel.empty:
        return None
    if preferred_ranks is not None and "track_vast_rank" in df_sel.columns:
        for r in preferred_ranks:
            hit = df_sel[df_sel["track_vast_rank"] == r]
            if not hit.empty:
                return hit.iloc[0]
    return df_sel.iloc[0]


def _compute_stft_from_results(res_base, base_row):
    """Compute STFT from results without normalization.

    Returns: (S_db, SFT, fs, frame_size, t0_frames, t1_frames, f0, f1) or None if data unavailable.
    """
    x = res_base.get("input_signal_array", None)
    exp_dir = base_row.get("experiment_dir") if hasattr(base_row, "get") else base_row["experiment_dir"]

    try:
        meta_path = Path(exp_dir) / "meta.json"
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        meta_params = meta.get("params", meta)
        frame_size = int(meta_params.get("frame_size"))
        nfft_perf = int(meta_params.get("nfft_performance", frame_size))
        fs = int(meta_params.get("fs", meta_params.get("sample_rate", 16000)))
    except Exception:
        frame_size = None
        nfft_perf = None
        fs = 16000

    if x is None or frame_size is None:
        return None

    nfft = nfft_perf if (nfft_perf is not None and nfft_perf >= frame_size) else frame_size
    SFT = signal.ShortTimeFFT(
        windows.hann(frame_size),
        hop=frame_size,
        fs=fs,
        mfft=nfft,
    )

    Sx = np.abs(SFT.spectrogram(x))
    Sx = np.asarray(Sx, dtype=float)
    # if np.nanmax(Sx) > 0:
        # Sx = Sx / np.nanmax(Sx)
    # S_db = 10 * np.log10(np.fmax(Sx, 10 ** (-6.5)))
    S_db = 10 * np.log10(Sx)

    t0_s, t1_s, f0, f1 = SFT.extent(len(x), center_bins=True)
    t0_frames = t0_s * fs / frame_size
    t1_frames = t1_s * fs / frame_size

    return (S_db, SFT, fs, frame_size, t0_frames, t1_frames, f0, f1)


def _overlay_stft_from_results(ax, stft_data, vmin=None, vmax=None):
    """Overlay STFT on a secondary y-axis using precomputed STFT data.

    Args:
        ax: Primary axis
        stft_data: Tuple from _compute_stft_from_results
        vmin: Minimum value for colormap (None for auto)
        vmax: Maximum value for colormap (None for auto)

    Returns: (ax2, im) or None if stft_data is None.
    """
    if stft_data is None:
        return None

    S_db, SFT, fs, frame_size, t0_frames, t1_frames, f0, f1 = stft_data

    ax2 = ax.twinx()
    im = ax2.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        cmap=SPEED_TRACK_COLORS.get("stft_color", "viridis"),
        extent=(t0_frames, t1_frames, f0 / 1000, f1 / 1000),
        alpha=0.8,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_ylabel("Freq. [kHz]")
    ax2.grid(False)
    ax.set_facecolor("none")
    ax.set_zorder(2)
    ax2.set_zorder(1)
    return ax2, im


def _plot_tracking_grid_options(
    df,
    *,
    ax,
    input,
    schedule,
    direction,
    filter_mode=None,
    track_vast_rank=None,
    plot_stft=False,
    stft_data=None,
    stft_vmin=None,
    stft_vmax=None,
    adaptive_window_widths=None,
    preferred_ranks=None,
    label_to_handle=None,
):
    """Plot true speed + estimated speed for full_grid and adaptive_window widths.

    Args:
        stft_data: Precomputed STFT data from _compute_stft_from_results
        stft_vmin: Global min for STFT colormap
        stft_vmax: Global max for STFT colormap
    """
    if label_to_handle is None:
        label_to_handle = {}
    if preferred_ranks is None:
        preferred_ranks = [4041, 8000, 1]

    base_sel = (
        (df["input"] == input)
        & (df["schedule"] == schedule)
        & (df["direction"] == direction)
    )
    if filter_mode is not None and "filter_mode" in df.columns:
        base_sel &= (df["filter_mode"] == filter_mode)
    if track_vast_rank is not None and "track_vast_rank" in df.columns:
        base_sel &= (df["track_vast_rank"] == track_vast_rank)

    df_full = df[base_sel & (df["grid_mode"] == "full_grid")]
    row_full = _pick_one_row(df_full, preferred_ranks=preferred_ranks)

    df_aw = df[base_sel & (df["grid_mode"] == "adaptive_window")]
    widths = []
    if not df_aw.empty and "search_window_width" in df_aw.columns:
        widths = sorted([w for w in df_aw["search_window_width"].dropna().unique().tolist()])

    # Optionally restrict which adaptive-window widths are plotted.
    # - None: plot all widths (default)
    # - callable: keep widths where callable(width) is truthy
    # - iterable: keep widths present in the iterable
    if adaptive_window_widths is not None and widths:
        if callable(adaptive_window_widths):
            widths = [w for w in widths if adaptive_window_widths(w)]
        else:
            try:
                allowed = set(adaptive_window_widths)
            except TypeError:
                allowed = {adaptive_window_widths}
            widths = [w for w in widths if w in allowed]

    base_row = row_full if row_full is not None else (_pick_one_row(df_aw, preferred_ranks=preferred_ranks) if not df_aw.empty else None)
    if base_row is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True)
        return label_to_handle

    res_base = load_results(base_row)
    frames = np.arange(len(res_base["true_speed"]))
    line = ax.plot(
        frames,
        res_base["true_speed"],
        label="True",
        linewidth=3.5,
        color=SPEED_TRACK_COLORS.get("true", "black"),
    )[0]
    label_to_handle.setdefault("True", line)

    if res_base.get("speed_change_frames", None) is not None:
        for f in res_base["speed_change_frames"]:
            ax.axvline(f, color="k", linestyle=":", alpha=0.4)

    ax2 = None
    stft_im = None
    if plot_stft and stft_data is not None:
        result = _overlay_stft_from_results(ax, stft_data, vmin=stft_vmin, vmax=stft_vmax)
        if result is not None:
            ax2, stft_im = result

    # Full grid
    if row_full is not None:
        res_fg = load_results(row_full)
        frames = np.arange(len(res_fg["est_speed"]))
        line = ax.plot(
            frames,
            res_fg["est_speed"],
            label="Est. full grid",
            alpha=0.9,
            linestyle="--",
            color=SPEED_TRACK_COLORS.get("full_grid", "C0")
        )[0]
        label_to_handle.setdefault("Est. full grid", line)

    # Adaptive window widths
    for w_idx, w in enumerate(widths):
        df_w = df_aw[df_aw["search_window_width"] == w]
        row_w = _pick_one_row(df_w, preferred_ranks=preferred_ranks)
        if row_w is None:
            continue
        res_w = load_results(row_w)
        try:
            w_txt = int(w) if float(w).is_integer() else w
        except Exception:
            w_txt = w
        # label = f"Est. aw ww{w_txt//2}"
        label = fr"Est. adapt. $c_{{width}}={w_txt//2}$"
        frames = np.arange(len(res_w["est_speed"]))
        line = ax.plot(
            frames,
            res_w["est_speed"],
            label=label,
            alpha=0.8,
            linestyle="--",
            color=SPEED_TRACK_COLORS.get(f"aw_{w_txt//2}", f"C{w_idx + 2}"),
        )[0]
        label_to_handle.setdefault(label, line)

    ax.grid(True)

    return (label_to_handle, ax2, stft_im)


def fig_speed_tracking_examples(
    df,
    schedule,
    direction,
    save_path,
    filter_mode=None,
    track_vast_rank=None,
    plot_stft=False,
    adaptive_window_widths=None,
    lsp_count=None,
    # infer_lsp_count=False,
):
    inputs = ["white", "speech", "rock", 'speech_music']

    fig, axes = plt.subplots(len(inputs), 1, sharex=True, figsize=(12, 9), sharey=False)

    # Preference order for VAST rank when multiple runs are available
    preferred_ranks = [4041, 8000, 1]

    # Collect unique legend entries across all axes
    label_to_handle = {}

    for ax, inp in zip(axes, inputs):
        label_to_handle, _, _ = _plot_tracking_grid_options(
            df,
            ax=ax,
            input=inp,
            schedule=schedule,
            direction=direction,
            filter_mode=filter_mode,
            track_vast_rank=track_vast_rank,
            plot_stft=plot_stft,
            adaptive_window_widths=adaptive_window_widths,
            preferred_ranks=preferred_ranks,
            label_to_handle=label_to_handle,
        )

        ax.set_title(INPUT_NAMES.get(inp, inp))
        ax.set_xlabel("")
        change = schedule.split("_")[1]
        # ax.set_yticks(np.arange(333, 353 + int(change), int(change)))
        ax.set_ylabel("Speed [m/s]")
        ax.grid(True)
        # Per-axis legends removed; we will use a single figure-level legend

    axes[-1].set_xlabel("Frame index")

    lsp_txt = _format_lsp_count(lsp_count)
    title = "Speed Tracking – " + schedule_direction_to_title(schedule, direction)
    if lsp_txt:
        title += f" ({lsp_txt})"
    fig.suptitle(title)
    # Single common legend for all axes
    handles = list(label_to_handle.values())
    labels = list(label_to_handle.keys())
    if handles:
        if direction == "up":
            bbox_x = 0.07
            bbox_y = 0.93
        else:
            bbox_x = 0.07
            bbox_y = 0.87
        fig.legend(handles, labels,
                   loc="upper left",
                    # fontsize=8,
                      ncol=2,
                   bbox_to_anchor=(bbox_x, bbox_y)
                   )
    fig.tight_layout(rect=(0, 0.03, 1, 1.01))
    fig.savefig(save_path)
    plt.close(fig)


def fig_tracking_inputs_vs_filters(
    df,
    schedule,
    direction,
    save_path,
    *,
    inputs=("white", "speech", "rock", "speech_music"),
    ranks=(1, 4041, 8000),
    plot_stft=False,
    plot_updated_filters=False,
    adaptive_window_widths=None,
    lsp_count=None,
    transpose=False,
    row_height=2.0,
    col_width=4.2,
    vmin_stft=None,
    vmax_stft=None,
    # infer_lsp_count=False,
):
    """Combined tracking figure.

    Layout:
      - Rows: input type
      - Columns: one column for no_filter, then one column per (szc_no_update, rank)

    Each panel overlays grid options (full_grid + adaptive_window widths).
    """

    # Preference order when multiple runs exist (only relevant for no_filter column)
    preferred_ranks = [4041, 8000, 1]

    columns = [("no_filter", None)]
    if plot_updated_filters:
        filter_columns = [("szc_update", int(r)) for r in ranks]
    else:
        filter_columns = [("szc_no_update", int(r)) for r in ranks]
    columns = columns + filter_columns


    if transpose:
        nrows = len(columns)
        ncols = len(inputs)
    else:
        nrows = len(inputs)
        ncols = len(columns)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        sharey=True,
        figsize=(col_width * ncols, row_height * nrows),
        squeeze=False,
        layout="constrained"
    )

    label_to_handle = {}
    stft_image = None

    # Precompute STFT data and global min/max per column (if plot_stft)
    stft_cache = {}  # (filt, rnk, inp) -> stft_data
    stft_vmin_by_col = {}  # (filt, rnk) -> vmin
    stft_vmax_by_col = {}  # (filt, rnk) -> vmax

    if plot_stft:
        preferred_ranks_list = [4041, 8000, 1]
        for filt, rnk in columns:
            S_db_list = []
            for inp in inputs:
                base_sel = (
                    (df["input"] == inp)
                    & (df["schedule"] == schedule)
                    & (df["direction"] == direction)
                )
                if filt is not None and "filter_mode" in df.columns:
                    base_sel &= (df["filter_mode"] == filt)
                if rnk is not None and "track_vast_rank" in df.columns:
                    base_sel &= (df["track_vast_rank"] == rnk)

                df_full = df[base_sel & (df["grid_mode"] == "full_grid")]
                row_full = _pick_one_row(df_full, preferred_ranks=preferred_ranks_list)

                if row_full is None:
                    df_aw = df[base_sel & (df["grid_mode"] == "adaptive_window")]
                    row_full = _pick_one_row(df_aw, preferred_ranks=preferred_ranks_list) if not df_aw.empty else None

                if row_full is not None:
                    res_base = load_results(row_full)
                    stft_data = _compute_stft_from_results(res_base, row_full)
                    stft_cache[(filt, rnk, inp)] = stft_data
                    if stft_data is not None:
                        S_db_list.append(stft_data[0])

            # Compute global min/max for this column
            if S_db_list:
                all_S_db = np.concatenate(S_db_list, axis=1)  # Concatenate along frequency axis
                stft_vmin_by_col[(filt, rnk)] = np.nanmin(all_S_db)
                stft_vmax_by_col[(filt, rnk)] = np.nanmax(all_S_db)

    if transpose:
        for i, (filt, rnk) in enumerate(columns):
            col_key = (filt, rnk)
            stft_vmin = stft_vmin_by_col.get(col_key) if vmin_stft is None else vmin_stft
            stft_vmax = stft_vmax_by_col.get(col_key) if vmax_stft is None else vmax_stft

            for j, inp in enumerate(inputs):
                ax = axes[i][j]
                cache_key = (filt, rnk, inp)
                stft_data = stft_cache.get(cache_key) if plot_stft else None

            # Plot content
                label_to_handle, ax2, stft_im = _plot_tracking_grid_options(
                    df,
                    ax=ax,
                    input=inp,
                    schedule=schedule,
                    direction=direction,
                    filter_mode=filt,
                    track_vast_rank=rnk,
                    plot_stft=plot_stft,
                    stft_data=stft_data,
                    stft_vmin=stft_vmin,
                    stft_vmax=stft_vmax,
                    adaptive_window_widths=adaptive_window_widths,
                    preferred_ranks=preferred_ranks,
                    label_to_handle=label_to_handle,
                )
                if stft_im is not None and stft_image is None:
                    stft_image = stft_im

            # Column titles
                if i == 0:
                    ax.set_title(INPUT_NAMES.get(inp, inp))

            # Row labels and axis labels
                if ax2 is not None and j != len(inputs) - 1:
                    ax2.set_ylabel("")
                    ax2.set_yticklabels([])
                if j == 0:
                    if filt == "no_filter":
                        ax.set_ylabel("No filter\nSpeed [m/s]")
                    else:
                        ax.set_ylabel(
                            f"{'update' if filt == 'szc_update' else 'NC'} V={rnk}\nSpeed [m/s]"
                        )
                else:
                    ax.set_ylabel("")

                if i == nrows - 1:
                    ax.set_xlabel("Frame index")
                else:
                    ax.set_xlabel("")
    else:
        for i, inp in enumerate(inputs):
            for j, (filt, rnk) in enumerate(columns):
                ax = axes[i][j]
                col_key = (filt, rnk)
                cache_key = (filt, rnk, inp)
                stft_data = stft_cache.get(cache_key) if plot_stft else None
                stft_vmin = stft_vmin_by_col.get(col_key)
                stft_vmax = stft_vmax_by_col.get(col_key)

                # Plot content
                label_to_handle, ax2, stft_im = _plot_tracking_grid_options(
                    df,
                    ax=ax,
                    input=inp,
                    schedule=schedule,
                    direction=direction,
                    filter_mode=filt,
                    track_vast_rank=rnk,
                    plot_stft=plot_stft,
                    stft_data=stft_data,
                    stft_vmin=stft_vmin,
                    stft_vmax=stft_vmax,
                    adaptive_window_widths=adaptive_window_widths,
                    preferred_ranks=preferred_ranks,
                    label_to_handle=label_to_handle,
                )
                if stft_im is not None and stft_image is None:
                    stft_image = stft_im

                # Column titles
                if i == 0:
                    if filt == "no_filter":
                        ax.set_title("No filter")
                    else:
                        ax.set_title(f"SZC {'update' if filt == 'szc_update' else 'no-update'}\nrank={rnk}")

                # Row labels and axis labels
                if ax2 is not None and j != len(columns) - 1:
                    ax2.set_ylabel("")
                    ax2.set_yticklabels([])
                if j == 0:
                    ax.set_ylabel(f"{INPUT_NAMES.get(inp, inp)}\nSpeed [m/s]")
                else:
                    ax.set_ylabel("")

                if i == nrows - 1:
                    ax.set_xlabel("Frame index")
                else:
                    ax.set_xlabel("")

    lsp_txt = _format_lsp_count(lsp_count)
    title = "Speed tracking – " + schedule_direction_to_title(schedule, direction)
    if lsp_txt:
        title += f" ({lsp_txt})"
    fig.suptitle(title)

    handles = list(label_to_handle.values())
    labels = list(label_to_handle.keys())
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            bbox_to_anchor=(0.19, 0.72),
        )

    # Add colorbar for STFT if present
    if plot_stft and stft_image is not None:
        cbar = fig.colorbar(stft_image, ax=axes.ravel().tolist(), pad=0.02, aspect=30)
        cbar.set_label("STFT Magnitude [dB]")

    # fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(save_path)
    plt.close(fig)


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



def fig_event_aligned_error(df, schedule, direction, save_path, filter_mode=None):
    fig, ax = plt.subplots()

    for inp, color in INPUT_COLORS.items():
        sel = (
            (df["input"] == inp) &
            (df["schedule"] == schedule) &
            (df["direction"] == direction)
        )
        if filter_mode is not None and "filter_mode" in df.columns:
            sel &= (df["filter_mode"] == filter_mode)
        rows = df[sel]

        plot_event_aligned_envelope(
            rows,
            signed=False,
            label=inp,
            ax=ax,
        )

    ax.set_ylabel("Absolute speed error")
    ax.legend()
    ax.set_title("Event-aligned error – " + schedule_direction_to_title(schedule, direction))

    fig.savefig(save_path)
    plt.close(fig)




def fig_error_distribution(df, schedule, direction, save_path, filter_mode=None):
    fig, ax = plt.subplots()

    for inp in ["white", "speech", "rock"]:
        sel = (
            (df["input"] == inp) &
            (df["schedule"] == schedule) &
            (df["direction"] == direction)
        )
        if filter_mode is not None and "filter_mode" in df.columns:
            sel &= (df["filter_mode"] == filter_mode)
        rows = df[sel]

        plot_error_distribution(rows, ax=ax)

    ax.set_xlabel("Absolute speed error")
    ax.set_title(f"Tracking error distribution – {schedule}, {direction}")

    fig.savefig(save_path)
    plt.close(fig)



def fig_AC_vs_error(df, schedule, direction, save_path, filter_mode=None):
    fig, ax = plt.subplots()

    sel = (
        (df["schedule"] == schedule) &
        (df["direction"] == direction)
    )
    if filter_mode is not None and "filter_mode" in df.columns:
        sel &= (df["filter_mode"] == filter_mode)
    rows = df[sel]

    plot_AC_vs_error(rows, ax=ax)

    ax.set_title(f"AC vs tracking error – {schedule}, {direction}")
    fig.savefig(save_path)
    plt.close(fig)

