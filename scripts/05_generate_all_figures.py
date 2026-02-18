"""
Generates all the figures for the paper.
- Tracking figures: time series for all schedules, directions (speed increase or decrease), filters, ranks. Organized in a grid with rows=inputs and cols=filters(+rank).
- Performance figures: AC and nSDP vs VAST rank for all schedules, directions, filters. Organized in a grid with rows=metrics and cols=inputs, with separate lines for each filter+rank.

Figure sizes, fonts etc. have been adjusted for paper presentation. For larger figures / better visibility, the sizes can be adjusted.

- Individual tracking examples: time series for each schedule, direction, filter, rank. One figure per combination, with all inputs overlaid. Not included in the paper, but useful for analysis and future reference.


"""
# Local imports from project
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import find_dotenv, dotenv_values
Loc_env = find_dotenv('.env')
CONFIG_ENV = dotenv_values(Loc_env)
_main_path = CONFIG_ENV.get('MainCodePath')
if isinstance(_main_path, str) and _main_path:
    sys.path.append(_main_path)

from src.utils.config_loader import get_shared_paths
from src.plotting.style import apply_paper_style
from src.plotting.tracking_figures import fig_tracking_inputs_vs_filters, fig_speed_tracking_examples
from src.plotting.performance_figures import fig_performance_inputs_vs_filters_ranks_combined, fig_performance_inputs_vs_filters
from src.utils.simdata import load_index

paths = get_shared_paths(config_env=CONFIG_ENV)
apply_paper_style()


df = load_index(csv_path=paths['results_path'] / "speed_track_results_index.csv")
df_one_lsp_one_mic = df[(df["group"] == "one_lsp_one_mic")].copy()
df_multi_lsp_one_mic = df[(df["group"] == "multi_lsp_one_mic")].copy()
out = paths['plots_path'] / "paper_figures"
out.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Combined tracking figure: rows=input, cols=filter+rank
# ------------------------------------------------------------
SCHEDULES = [
    "speed_1_dt_2",
    "speed_2_dt_2",
    "speed_4_dt_2",
]
DIRECTIONS = ["up",
              "down"
              ]
RANKS = [1,
        #  4041,
         8000
         ]

INPUTS = ["white", "speech", "rock",
        #   "speech_music"
          ]

# Adjust fonts for tracking figures
plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2,
    })

combined_out = out / "tracking_combined"
combined_out.mkdir(exist_ok=True)
for sched in SCHEDULES:
    if sched not in df["schedule"].unique():
        print(f"Skipping schedule {sched} as it is not in the data.")
        continue
    combined_out_sched = combined_out / sched
    combined_out_sched.mkdir(exist_ok=True)
    if sched == "speed_4_dt_2":
        adaptive_window_widths = [8]
    else:
        adaptive_window_widths = [6]

    for direction in DIRECTIONS:
        fig_tracking_inputs_vs_filters(
            df_multi_lsp_one_mic,
            sched,
            direction,
            combined_out_sched / f"fig_tracking_combined_multi-lsp_{sched}_{direction}.pdf",
            ranks=RANKS,
            plot_stft=True,
            adaptive_window_widths=adaptive_window_widths,
            lsp_count=1,
            transpose=True,
            inputs=INPUTS,
            row_height=1.5,
            col_width=4.9,
            vmin_stft=-50,
        )
        print(f"Generated fig_tracking_combined_multi-lsp_{sched}_{direction}.pdf")
        fig_tracking_inputs_vs_filters(
            df_one_lsp_one_mic,
            sched,
            direction,
            combined_out_sched / f"fig_tracking_combined_one-lsp_{sched}_{direction}.pdf",
            ranks=RANKS,
            plot_stft=True,
            adaptive_window_widths=adaptive_window_widths,
            lsp_count=1,
            transpose=True,
            inputs=INPUTS,
            row_height=1.5,
            col_width=4.9,
            vmin_stft=-50,
        )
        print(f"Generated fig_tracking_combined_one-lsp_{sched}_{direction}.pdf")

# ------------------------------------------------------------
# Combined performance figure: rows=metric (AC,nSDP), cols=input
# One figure per schedule+direction+rank
# ------------------------------------------------------------


# Adjust fonts for performance figures
plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
    })
perf_combined_out = out / "performance_combined"
perf_combined_out.mkdir(exist_ok=True)

# Optional: choose different metrics per rank to save space.
# Example: show both metrics for rank 1, only AC for 4041, only nSDP for 8000.
RANKS = [1, 4041, 8000]
METRICS_BY_RANK = {
    1: ("AC",),
    4041: ("AC", "nSDP"),
    8000: ("AC", "nSDP"),
}
for sched in SCHEDULES:
    if sched not in df["schedule"].unique():
        print(f"Skipping schedule {sched} as it is not in the data.")
        continue
    perf_combined_out_sched = perf_combined_out / sched
    perf_combined_out_sched.mkdir(exist_ok=True)
    for direction in DIRECTIONS:
        # for r in RANKS:
            # fig_performance_inputs_vs_filters(
        #         df_multi_lsp_one_mic,
        #         schedule=sched,
        #         direction=direction,
        #         track_vast_rank=r,
        #         save_path=perf_combined_out_sched / f"fig_perf_combined_multi-lsp_{sched}_{direction}_rank_{r}.pdf",
        #         inputs=INPUTS,
        #         adaptive_window_widths=[0],
        #     )
        #     print(f"Generated fig_perf_combined_multi-lsp_{sched}_{direction}_rank_{r}.pdf")''
        fig_performance_inputs_vs_filters_ranks_combined(
            df_multi_lsp_one_mic,
            schedule=sched,
            direction=direction,
            ranks=RANKS,
            save_path=perf_combined_out_sched / f"fig_perf_combined_multi-lsp_{sched}_{direction}_ranks_combined.pdf",
            inputs=INPUTS,
            filter_modes=("szc_gt_baseline", "szc_sicer_baseline", "szc_no_update", "szc_update"),
            metrics_by_rank=METRICS_BY_RANK,
            adaptive_window_widths=[0],
            apply_filter_only=True,
            col_width=5.0,
            legend_on_first_rank_only=True)
        print(f"Generated fig_perf_combined_multi-lsp_{sched}_{direction}_ranks_combined.pdf")


# ------------------------------------------------------------
# Individual tracking examples: one figure per schedule+direction+filter(+rank)
# ------------------------------------------------------------

# SCHEDULES = [
#             "speed_1_dt_2",
#              "speed_2_dt_2",
#               "speed_4_dt_2"
#              ]
# DIRECTIONS = ["up",
#               "down"
#               ]
# FILTER_MODES = [
#                 # "no_filter",
#                 # "szc_no_update",
#                 "szc_update",
#                ]
# RANKS = [1,
#           4041, 8000
#          ]
# track_out = out / "tracking_examples"
# track_out = out / "test"
# track_out.mkdir(exist_ok=True)
# for sched in SCHEDULES:
#     track_out_sched = track_out / sched
#     track_out_sched.mkdir(exist_ok=True)
#     for direction in DIRECTIONS:
#         for filt in FILTER_MODES:
#             if filt == "no_filter":
#                 # continue
#                 # Full figure with all ranks overlaid
#                 fig_speed_tracking_examples(
#                     df_one_lsp_one_mic, sched, direction,
#                     track_out_sched / f"fig_tracking_one-lsp_{sched}_{direction}_{filt}.pdf",
#                     filter_mode=filt,
#                     plot_stft=True,
#                 )
#                 print(f"Generated fig_tracking_one-lsp_{sched}_{direction}_{filt}.pdf")
#                 fig_speed_tracking_examples(
#                     df_multi_lsp_one_mic, sched, direction,
#                     track_out_sched / f"fig_tracking_multi-lsp_{sched}_{direction}_{filt}.pdf",
#                     filter_mode=filt,
#                     plot_stft=True,
#                 )
#                 print(f"Generated fig_tracking_multi-lsp_{sched}_{direction}_{filt}.pdf")
#             else:
#                 # Separate figures for each rank
#                 for r in RANKS:
#                     if filt == "szc_no_update":
#                         # Only onc-mic case for no-update filter, since one-mic with no update is not meaningful
#                         fig_speed_tracking_examples(
#                             df_one_lsp_one_mic, sched, direction,
#                             track_out_sched / f"fig_tracking_one-lsp_{sched}_{direction}_{filt}_rank_{r}.pdf",
#                             filter_mode=filt,
#                             track_vast_rank=r,
#                             plot_stft=True,
#                         )
#                         print(f"Generated fig_tracking_one-lsp_{sched}_{direction}_{filt}_rank_{r}.pdf")
#                     fig_speed_tracking_examples(
#                         df_multi_lsp_one_mic, sched, direction,
#                         track_out_sched / f"fig_tracking_multi-lsp_{sched}_{direction}_{filt}_rank_{r}.pdf",
#                         filter_mode=filt,
#                         track_vast_rank=r,
#                         plot_stft=True,
#                     )
#                     print(f"Generated fig_tracking_multi-lsp_{sched}_{direction}_{filt}_rank_{r}.pdf")

# Unused plots, use with care, check for errors
            # fig_event_aligned_error(
            #     df_one_lsp_one_mic, sched, direction,
            #     out / f"fig_event_error_{sched}_{direction}_{filt}.pdf",
            #     filter_mode=filt,
            # )

            # fig_error_distribution(
            #     df_one_lsp_one_mic, sched, direction,
            #     out / f"fig_error_dist_{sched}_{direction}_{filt}.pdf",
            #     filter_mode=filt,
            # )

            # fig_AC_vs_error(
            #     df_one_lsp_one_mic, sched, direction,
            #     out / f"fig_AC_vs_error_{sched}_{direction}_{filt}.pdf",
            #     filter_mode=filt,
            # )