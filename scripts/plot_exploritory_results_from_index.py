"""Plot exploratory results from the results index.

This script loads results from a CSV index and generates visualization plots:
- Time series plot of speed tracking for a specific run
- Event-aligned envelope comparison across different input types
"""

import matplotlib.pyplot as plt
from src.utils.simdata import load_index
from src.plotting.tracking_plots import plot_speed_tracking, plot_event_aligned_envelope, select_runs

df = load_index(csv_path="/home/ubuntu/SZC_Speed_Change_Extend/outputs/results/results_index.csv")

rows = select_runs(
    df,
    input="speech",
    schedule="speed_2_dt_2",
    direction="up",
)

# Time series
fig, ax = plt.subplots()
plot_speed_tracking(rows.iloc[0], title="Speech – speed_2_dt_2 – up", ax=ax)
ax.legend()
fig.savefig("speech_speed_2_dt_2_up.png")
plt.show()

# Event-aligned comparison
fig, ax = plt.subplots()
# For dt = 2 seconds, there is a change for every 8 frames (frame duration = 0.25s)
for inp in ["white", "speech", "rock"]:
    r = select_runs(df, input=inp, schedule="speed_2_dt_2", direction="up")
    plot_event_aligned_envelope(r, label=inp, ax=ax, pre_frames=5, post_frames=5)
ax.legend()
fig.savefig("event_aligned_comparison_up.png")
plt.show()
