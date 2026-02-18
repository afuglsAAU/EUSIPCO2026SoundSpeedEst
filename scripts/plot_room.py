"""
Plot room setup visualization with loudspeaker and microphone positions.

This script visualizes the spatial arrangement of loudspeakers and microphones in a room
for acoustic control experiments. It generates multiple plots showing:
- Detailed zoomed-in view of loudspeaker and microphone positions
- Full room layout with labeled reference speaker and observation microphone
- Layout with customized aspect ratio and legend positioning

The script loads loudspeaker and microphone positions from MATLAB files and creates
publication-quality plots using matplotlib with applied paper styling.
"""

#%%

import numpy as np
import scipy.io as sio
import sys
from dotenv import find_dotenv, dotenv_values
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Local imports
Loc_env = find_dotenv('.env')
CONFIG_ENV = dotenv_values(Loc_env)
sys.path.append(CONFIG_ENV['MainCodePath'])

from src.utils.config_loader import get_shared_paths

paths = get_shared_paths(config_env=CONFIG_ENV)

def apply_paper_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 3,
        "figure.figsize": (3.5, 4.0),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
    })
apply_paper_style()


# # The desired reverberation time and dimensions of the room
rt60_tgt = 0.3  # Reverberation time (s)
room_dim = [4.5, 4.5, 2.2]  # Room dimensions [x, y, z] (m)
rir_length = None#700  # Number of RIR samples, Each RIR Length
fs = 16000 #  Sample frequency (samples/s)
speeds = np.arange(333, 353+1, 1)

# Load loudspeaker and microphone positions used in Sankha's paper
loudspeaker_positions = sio.loadmat(paths['data_path'].joinpath("f_array.mat"))['array_geometry']
mic_pos = sio.loadmat(paths['data_path'].joinpath("f_zone.mat"))
control_mic_bz = mic_pos['br_zone']
control_mic_dz = mic_pos['dk_zone']

fig, ax = plt.subplots(1, 1, figsize=(8, 6), num=1, clear=True, tight_layout=True)
ax.set_title('Loudspeaker and Microphone Positions (Zoomed In)')
ax.scatter(loudspeaker_positions[:, 0], loudspeaker_positions[:, 1], edgecolors='blue', label='Loudspeakers', marker='^', facecolor="none")
for i in range(loudspeaker_positions.shape[0]):
    ax.annotate(f'L{i}', (loudspeaker_positions[i, 0], loudspeaker_positions[i, 1]), fontsize=10)

ax.scatter(control_mic_bz[:, 0], control_mic_bz[:, 1], edgecolors='green', label='Bright Zone Mics', marker='o', facecolor="none")
for i in range(control_mic_bz.shape[0]):
    ax.annotate(f'M{i}', (control_mic_bz[i, 0], control_mic_bz[i, 1]), fontsize=10)

ax.scatter(control_mic_dz[:, 0], control_mic_dz[:, 1], edgecolors='red', label='Dark Zone Mics', marker='o', facecolor="none")
for i in range(control_mic_dz.shape[0]):
    ax.annotate(f'M{i}', (control_mic_dz[i, 0], control_mic_dz[i, 1]), fontsize=10)

ax.legend()
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
fig.savefig(paths['plots_path'].joinpath('room_setup_zoomed.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# Full plot
tracking_mic = control_mic_bz[0]
ref_speaker = loudspeaker_positions[7]
fig, ax = plt.subplots(1, 1, figsize=(8, 4), num=2, clear=True, tight_layout=True)
# ax.set_title('Loudspeaker and Microphone Positions')
# Speakers
ax.scatter(loudspeaker_positions[:, 0], loudspeaker_positions[:, 1], edgecolors='C0', label='Loudspeakers', marker='s', facecolor="C0", s=4)

ax.scatter(ref_speaker[0], ref_speaker[1], edgecolors='C3', label='Reference Speaker, L8', marker='s', facecolor="C3", s=4)

# DZ mics
ax.scatter(control_mic_dz[:, 0], control_mic_dz[:, 1], edgecolors='black', label='Dark Zone Mics', marker='o', facecolor="black", s=8)

# BZ mics
ax.scatter(control_mic_bz[:, 0], control_mic_bz[:, 1], edgecolors='C2', label='Bright Zone Mics', marker='o', facecolor="C2", s=8)
ax.scatter(tracking_mic[0], tracking_mic[1], edgecolors='C3', label='Observation Mic, M1', marker='o',
            facecolor="C3", s=8
           )

ax.legend()
ax.set_xlabel('X Position (m)', fontsize=8)
ax.set_ylabel('Y Position (m)', fontsize=8)
ax.set_aspect('equal', 'box')
ax.set_xlim(0, room_dim[0])
ax.set_ylim(0, room_dim[1])
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
fig.savefig(paths['plots_path'].joinpath('room_setup_full.pdf'), dpi=300, bbox_inches='tight')


## Aspect off
fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), num=3, clear=True)
ax.scatter(loudspeaker_positions[:, 0], loudspeaker_positions[:, 1], edgecolors='C0', label='Loudspeakers', marker='s', facecolor="C0", s=5)

ax.scatter(ref_speaker[0], ref_speaker[1], edgecolors='C3', label='Ref. LSP, L8', marker='s', facecolor="C3", s=5)

# DZ mics
ax.scatter(control_mic_dz[:, 0], control_mic_dz[:, 1], edgecolors='black', label='Dark Zone Mics', marker='o', facecolor="black", s=8)

# BZ mics
ax.scatter(control_mic_bz[:, 0], control_mic_bz[:, 1], edgecolors='C2', label='Bright Zone Mics', marker='o', facecolor="C2", s=8)
ax.scatter(tracking_mic[0], tracking_mic[1], edgecolors='C3', label='Obs. Mic', marker='o',
            facecolor="C3", s=8
           )

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_xlim(0, room_dim[0])
ax.set_ylim(0, room_dim[1])
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(loc='upper left')


# axins = inset_axes(ax, width="45%", height="45%",
#                    bbox_to_anchor=(0.1, 0.1, 1, 1),
#                    bbox_transform=ax.transAxes, loc="lower left")


# # axins.set_title('Loudspeaker and Microphone Positions (Zoomed In)')
# axins.scatter(loudspeaker_positions[:, 0], loudspeaker_positions[:, 1], edgecolors='C0', label='Loudspeakers', marker='s', facecolor="C0", s=5)
# # for i in range(loudspeaker_positions.shape[0]):
#     # axins.annotate(f'L{i}', (loudspeaker_positions[i, 0], loudspeaker_positions[i, 1]), fontsize=7, horizontalalignment='right')

# axins.scatter(control_mic_bz[:, 0], control_mic_bz[:, 1], edgecolors='C2', label='Bright Zone Mics', marker='.', facecolor="C2")
# # for i in range(control_mic_bz.shape[0]):
# #     axins.annotate(f'M{i}', (control_mic_bz[i, 0], control_mic_bz[i, 1]), fontsize=7)

# axins.scatter(control_mic_dz[:, 0], control_mic_dz[:, 1], edgecolors='black', label='Dark Zone Mics', marker='.', facecolor="black")
# # for i in range(control_mic_dz.shape[0]):
# #     axins.annotate(f'M{i}', (control_mic_dz[i, 0], control_mic_dz[i, 1]), fontsize=7)


# axins.tick_params(axis='both', which='major', labelsize=8)
# axins.grid(True, which='both', linestyle='--', linewidth=0.5)
fig.savefig(paths['plots_path'].joinpath('room_setup_full_aspect_off.pdf'), dpi=300, bbox_inches='tight')

plt.show()

#%%
