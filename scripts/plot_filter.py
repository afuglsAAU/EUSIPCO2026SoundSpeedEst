
"""
Plot VAST filters in time and frequency domains for a specific speed, rank, and loudspeaker.

This script loads Ground Truth (GT) VAST filters from configuration and generates visualization
plots showing filter coefficients in both time domain and frequency domain (magnitude response
in dB). Plots can be saved to disk for inspection and analysis.

Usage:
    python plot_filter.py --config <config_path> [--speed <speed>] [--rank <rank>] [--lsp <lsp>]

Args:
    --config: Path to config module (e.g., configs/config_RT60_0.1.py) [required]
    --speed: Sound speed to load filters for in m/s (default: 333)
    --rank: VAST rank to plot (default: 8000)
    --lsp: Loudspeaker index to plot (default: 7)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from dotenv import find_dotenv, dotenv_values
from pathlib import Path

# Local imports from project
Loc_env = find_dotenv('.env')
CONFIG_ENV = dotenv_values(Loc_env)
_main_path = CONFIG_ENV.get('MainCodePath')
if isinstance(_main_path, str) and _main_path:
    sys.path.append(_main_path)

from src.utils.config_loader import load_config_module, get_shared_paths
from src.utils.simdata import load_vast_filters_for_speed

def plot_rank(filter, rank, lsp, nfft=4096, savepath=None):
    fig, axes = plt.subplots(2, 1, figsize=(8, 4))
    axes[0].plot(filter)
    axes[0].set_title(f'Filter Coefficients')
    axes[0].set_xlabel('Coefficient Index')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid()

    freq_axis = np.fft.rfftfreq(nfft, d=1/16000)
    filter_fft = np.fft.rfft(filter, n=nfft)
    axes[1].plot(freq_axis, 20 * np.log10(np.abs(filter_fft) + 1e-12))
    axes[1].set_title(f'Frequency Response')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_xlim(0, 8000)
    axes[1].grid()

    fig.suptitle(f'VAST Filter Rank {rank}, LSP {lsp}')
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath)
        print(f'Plot saved as {savepath}')
    return fig


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot GT VAST filters for a specific speed, rank, and loudspeaker in time and frequency domains.')
    parser.add_argument('--config', type=Path, required=True,
                        help='Path to config module (e.g., exp1_SICER/config_RT60_0.1.py)')
    parser.add_argument('--speed', type=int, default=333,
                        help='Sound speed to load filters for (default: 333 m/s)')
    parser.add_argument('--rank', type=int, default=8000,
                        help='VAST rank to plot (default: 8000)')
    parser.add_argument('--lsp', type=int, default=7,
                        help='Loudspeaker (default: 7)')


    args = parser.parse_args()

    cfg = load_config_module(args.config)
    paths = get_shared_paths(config_env=CONFIG_ENV)
    params = cfg.shared_params.copy()

    filters, loaded_ranks = load_vast_filters_for_speed(args.speed, paths, params)
    rank_idx = loaded_ranks.index(args.rank)
    savepath = f'figures/GT_VAST_filters/vast_filter_speed_{args.speed}_rank_{args.rank}_lsp_{args.lsp}_plot.png'
    fig = plot_rank(filters[rank_idx, args.lsp], args.rank, args.lsp, savepath=savepath)
