
"""
Module for comparing speed tracking results from multiple experiments.

This script provides functionality to load and visualize speed tracking performance
metrics (AC and nSDP) from different experimental runs, allowing for side-by-side
comparison of results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from pathlib import Path
import argparse


def plot_compare_results(results_list: list[Dict], labels: list, save_path: Optional[str]=None) -> None:
    """Plot comparison of speed tracking results from multiple experiments.

    Args:
        results_list (List of Dicts): List of results dictionaries from different experiments.
        labels (list): List of labels corresponding to each results dictionary.
        save_path (Optional[str], optional): Path to save the plot. Defaults to None.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for results, label in zip(results_list, labels):
        ac_perf = results.get('AC_performance_track', None)
        nsdp_perf = results.get('nSDP_performance_track', None)
        n_frames = ac_perf.shape[0]
        t = np.arange(n_frames)
        axes[0].plot(t, ac_perf, lw=2, label=f'{label} AC', marker='o', linestyle='--', markerfacecolor='none', alpha=0.5)
        axes[1].plot(t, nsdp_perf, lw=2, label=f'{label} nSDP', marker='s', linestyle='--', markerfacecolor='none', alpha=0.5)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('AC [dB]')
    axes[0].set_title('Comparison of Tracking Results')
    axes[0].grid(True, ls='--', alpha=0.4)
    axes[0].legend()
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('nSDP [dB]')
    axes[1].set_title('Comparison of Tracking Results')
    axes[1].grid(True, ls='--', alpha=0.4)
    axes[1].legend()
    plt.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.mkdir(parents=True, exist_ok=True)
        out_name = out_path.joinpath(f'comparison_speed_tracking_plot.png')
        plt.savefig(out_name, dpi=300)
        print(f"Comparison plot saved as '{out_name}'")
    else:
        plt.savefig('comparison_performance_plot.png', dpi=300)
        print("Comparison plot saved as 'comparison_performance_plot.png'")

    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare two speed tracking results.")
    parser.add_argument('--r1', required=True, help="Path to the first results directory.")
    parser.add_argument('--r2', required=True, help="Path to the second results directory.")
    parser.add_argument('--label1', type=str, default="r1", help="Label for first results.")
    parser.add_argument('--label2', type=str, default="r2", help="Label for second results.")
    parser.add_argument('--save-path', type=str, default=None,
                        help="Directory to save the plots. If not provided, saves in same directory as results_path.")
    args = parser.parse_args()

    # res1 = np.load("exp1_SICER/results/speed_tracking/Linear_array_circ_zones/RT60_0.1/input_audio_combined_speech_16k_11_sec_2.wav/start_335_step_2/filters_applied_r_8000/filters_updated_false/adaptive_base_False/adaptive_window_True_ww6_ml3/estimator_grid/20260127_150649/results.npz")
    # res2 = np.load("exp1_SICER/results/speed_tracking/Linear_array_circ_zones/RT60_0.1/input_audio_combined_speech_16k_11_sec_2.wav/start_335_step_2/filters_applied_r_8000/filters_updated_true_sd1.0/adaptive_base_False/adaptive_window_True_ww6_ml3/estimator_grid/20260127_164429/results.npz")

    res1 = np.load(args.r1)
    res2 = np.load(args.r2)
    plot_compare_results([dict(res1), dict(res2)], labels=[args.label1, args.label2], save_path=args.save_path)