# paper_figures/style.py
import matplotlib.pyplot as plt

def apply_paper_style():
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 3,
        "figure.figsize": (3.5, 4.0),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],

        # "font.family": "serif",
        # "font.serif": ["Times New Roman"],          # Match IEEEtran

    })


INPUT_COLORS = {
    "white": "black",
    "speech": "tab:blue",
    "rock": "tab:orange",
    "speech_music": "tab:green",
}

INPUT_NAMES = {
    "white": "White noise",
    "speech": "Speech",
    "rock": "Rock music",
    "speech_music": "Speech + music",
}
