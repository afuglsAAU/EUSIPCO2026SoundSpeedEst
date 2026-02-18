# Online Single-Channel Audio-Based Sound Speed Estimation for Robust Multi-Channel Audio Control
This is the official implementation of the paper titled [Online Single-Channel Audio-Based Sound Speed Estimation for Robust Multi-Channel Audio Control](https://vbn.aau.dk/en/publications/online-single-channel-audio-based-sound-speed-estimation-for-robu/) submitted to EUSIPCO 2026.

## Contents of repository
- Additional tracking- and Sound Zone Control (SZC)-performance figures not included in the paper: `outputs/plots/paper_figures/`


- Python codebase for our experiments around **sound-speed estimation** in SZC including:
  - Generating / loading multi-speed room impulse responses (RIRs)
  - Computing VAST control filters (Ground Truth and SICER-corrected)
  - Running sound-speed tracking experiments (grid / adaptive-window)
  - Indexing results and generating paper-style figures

Most scripts assume you run them **from the repository root**.

---

## Requirement & Installation
```bash
Python >= 3.11
```

Create an environment (venv/conda) and install dependencies:

```bash
pip install -r requirements.txt
```

If you see heavy CPU usage, some scripts explicitly set `MKL_NUM_THREADS` / `OMP_NUM_THREADS`.

## Repository layout

### Top-level files

- `requirements.txt`: pinned Python dependencies.
- `.env`: paths used by most scripts (data, outputs, audio). See **Environment / paths**.


### Key folders

- `configs/`
  - `config_RT60_0.1.py`: base configuration (defines `shared_params`).
  - `experiments.yaml`: experiment matrix used by `scripts/03_run_tracking_experiments.py`.

- `scripts/`: main entrypoints / experiment drivers (documented below).

- `src/`: reusable library code used by `scripts/`
  - `src/algorithms/`: filter generation and helpers.
  - `src/utils/`: config loading, IO helpers, speed-change interpolation, data loading.
  - `src/plotting/`: paper-style plots and figure builders.

- `SimData/`: MATLAB `.mat` files used by simulations/experiments.
  - Includes per-speed RIR `.mat` files and geometry files like `f_array.mat` / `f_zone.mat`.

- `InputAudio/`: example WAV files used when `input_signal='audio'`.

- `outputs/`: generated artifacts (created automatically if missing)
  - `outputs/filter_mat_files/`: stored VAST filter `.npz` files.
  - `outputs/results/`: tracking results and speed-sweep performance results.
  - `outputs/plots/`: generated plots and paper figures.

- `.run_params/`: auto-generated per-experiment JSON parameter files written by `scripts/03_run_tracking_experiments.py`.

## Environment / paths

Most scripts load paths from `.env` via `python-dotenv`. These variables are used by `src/utils/config_loader.py:get_shared_paths()`.

Required (typical):

```dotenv
MainCodePath=/absolute/path/to/SZC_Speed_Change_Extend/
MainOutputPath=/absolute/path/to/SZC_Speed_Change_Extend/outputs/
SimDataPath=/absolute/path/to/SZC_Speed_Change_Extend/SimData/
SimDataCovarPath=/absolute/path/to/somewhere/to/cache/covariances/
InputAudioPath=/absolute/path/to/SZC_Speed_Change_Extend/InputAudio/
```

Notes:
- `MainCodePath` is set for scripts to automatically repo is on the `PYTHONPATH`.
- `SimDataCovarPath` is where large covariance matrices are cached. 
- If you keep everything inside this repo, pointing `SimDataPath` to `./SimData` and `InputAudioPath` to `./InputAudio` works.

## Typical end-to-end workflow

The pipeline (at a high level):

1. Compute **GT** VAST filters for all speeds
2. Generate **SICER-corrected** RIRs (base → target)
3. Compute **SICER** VAST filters for those corrected RIRs
4. Run speed-tracking experiments from `configs/experiments.yaml`
5. Index results
6. Generate figures

Example commands to generate results in paper:

```bash
# 0) Compute GT VAST filters for all simulated speeds
python scripts/00_compute_GT_VAST_filters.py --config configs/config_RT60_0.1.py --parallel --max-workers 8

# 1) Create SICER-corrected IRs (base speed -> all target speeds)
python scripts/01_run_SICER_interp.py --config configs/config_RT60_0.1.py --base-speed 333 --speeds config

# 2) Compute SICER VAST filters using those corrected IRs
python scripts/02_compute_SICER_VAST_filters.py --config configs/config_RT60_0.1.py --parallel --max-workers 2 --base-speeds 333

# 3) Run the tracking experiment matrix (dry-run first; add --no-dry-run to execute)
python scripts/03_run_tracking_experiments.py --schedule speed_2_dt_2 --direction up
python scripts/03_run_tracking_experiments.py --schedule speed_2_dt_2 --direction up --no-dry-run --max-parallel 4 --skip-existing

# 4) Build a CSV index of tracking runs
python scripts/04_index_results.py

# 5) Generate paper-style figures into outputs/plots/paper_figures/
python scripts/05_generate_all_figures.py
```

To generate results for other speed change patterns remove the `schedule` and `direction` flags.

## Scripts (in `scripts/`)

This section documents what each entrypoint does and the key CLI arguments.

### `00_compute_GT_VAST_filters.py`

Computes **ground truth VAST filters** for each speed in `shared_params['sound_speeds']`.

- Loads GT RIRs (from `SimDataPath` conventions)
- Computes covariance matrices (`R_B`, `R_D`, `r_B`) per speed (cached)
- Joint-diagonalizes and generates filters across all ranks in `shared_params['vast_ranks']`

Usage:

```bash
python scripts/00_compute_GT_VAST_filters.py \
  --config configs/config_RT60_0.1.py \
  --parallel --max-workers 8
```

Outputs:

- Covariances: `SimDataCovarPath/GT_filters/`
- Filters: `outputs/filter_mat_files/RT60_<rt60>/GT_VAST_filters/`

### `01_run_SICER_interp.py`

Generates **SICER speed-corrected RIRs** by loading per-speed split `.mat` files and interpolating from a base speed to target speeds according to method of [1].

Usage:

```bash
python scripts/01_run_SICER_interp.py \
  --config configs/config_RT60_0.1.py \
  --base-speed 343 \
  --speeds config

# Or run multiple base speeds
python scripts/01_run_SICER_interp.py --config configs/config_RT60_0.1.py --base-speeds config --speeds config
```

Outputs:

- Corrected RIR `.mat` files under `SimDataPath/simulated_RIRs/SICER_corrected/<array_setup>_RT60_<rt60>/`.

### `02_compute_SICER_VAST_filters.py`

Computes VAST filters from **SICER-corrected** RIRs produced by `01_run_SICER_interp.py` to get the SICER baseline performance with tracking script.

Usage:

```bash
python scripts/02_compute_SICER_VAST_filters.py \
  --config configs/config_RT60_0.1.py \
  --base-speeds 333 \
  --target-speeds config \
  --parallel --max-workers 2
```

Outputs:

- Covariances: `SimDataCovarPath/SICER_filters/`
- Filters: `outputs/filter_mat_files/RT60_<rt60>/SICER_VAST_filters/`

### `03_run_tracking_experiments.py`

Orchestrates speed-tracking experiments defined in `configs/experiments.yaml`.

It expands the YAML into a job list and launches `scripts/tracking_algorithm.py` via subprocess, writing per-run overrides into `.run_params/<experiment_id>.json`.

Important behavior:

- Defaults to **dry-run** unless you pass `--no-dry-run`.
- Can skip already-computed runs by scanning existing `meta.json` files (`--skip-existing`).
- Supports filtering down the experiment matrix by group/schedule/direction/input/etc.

Filtered experiment usage:

```bash
# Dry-run one slice
python scripts/03_run_tracking_experiments.py \
  --group multi_lsp_one_mic \
  --experiment szc_update \
  --schedule speed_2_dt_2 \
  --direction up \
  --input speech \
  --mode adaptive_window \
  --window-width 6

# Execute the full matrix (or your filtered slice)
python scripts/03_run_tracking_experiments.py --no-dry-run --max-parallel 4 --skip-existing
```

### `tracking_algorithm.py`

Core per-experiment runner. This is what `03_run_tracking_experiments.py` ultimately executes.

Key flags:

- `--apply-filter`: apply control filters while tracking
- `--update-filter`: update filters during tracking
- `--obs-only`: compute only observation-mic signals (faster)
- `--gt-baseline` / `--sicer-baseline`: baseline runs using precomputed filters (no tracking)
- `--params-file .run_params/<id>.json`: JSON overrides (used by the orchestrator)

You can run it directly for a single experiment:

```bash
python scripts/tracking_algorithm.py --config configs/config_RT60_0.1.py --estimator grid --obs-only
```

### `04_index_results.py`

Scans the tracking results directory for `meta.json` files and consolidates them into a CSV index.

Usage:

```bash
python scripts/04_index_results.py
```

Output:

- `outputs/results/speed_track_results_index.csv`

### `05_generate_all_figures.py`

Generates the paper-style figures using the results index and `src/plotting/*`.

Assumes `outputs/results/speed_track_results_index.csv` exists.

Usage:

```bash
python scripts/05_generate_all_figures.py
```

Output:

- `outputs/plots/paper_figures/` (PDFs)

### `compare_speed_tracks.py`

Side-by-side comparison of two speed-tracking runs (plots AC/nSDP time series) for quick exploritory comparison.

Usage:

```bash
python scripts/compare_speed_tracks.py \
  --r1 /path/to/results1.npz \
  --r2 /path/to/results2.npz \
  --label1 "baseline" --label2 "update" \
  --save-path outputs/plots/
```

### `plot_filter.py`

Plots a single GT VAST filter (time + frequency response) for a chosen speed/rank/loudspeaker.

Usage:

```bash
python scripts/plot_filter.py --config configs/config_RT60_0.1.py --speed 333 --rank 8000 --lsp 7
```

### `plot_room.py`

Produces room-layout plots (loudspeakers + mic zones) from the geometry `.mat` files.

This script is not parameterized via CLI; edit variables at the top if needed and run:

```bash
python scripts/plot_room.py
```

## Troubleshooting

- **Imports fail / `src` not found**: ensure `.env` has `MainCodePath` pointing to the repo root.
- **Missing data files**: check `.env` paths for `SimDataPath` and `InputAudioPath`.
- **Results/plots not appearing under `outputs/`**: verify `.env` `MainOutputPath` points at `./outputs` (or your intended output folder).


## Citation:
If you find the paper useful in your research, please cite:  
```
@inproceedings{fuglsig_2026_online,
title = "Online Single-Channel Audio-Based Sound Speed Estimation for Robust Multi-Channel Audio Control",
author = "Fuglsig, \{Andreas Jonas\} and Christensen, \{Mads Gr{\ae}sb{\o}ll\} and Jensen, \{Jesper Rindom\}",
year = "2026",
month = feb,
language = "English",
booktitle = "2026 34th European Signal Processing Conference (EUSIPCO)",
note = "Submitted preprint"
}
```

## References & Acknowledgement

We would like to thank the authors of [1] for sharing their original Matlab code for the SICER correction method.

[1] S. S. Bhattacharjee, J. R. Jensen, and M. G. Christensen, “Sound Speed Perturbation Robust Audio: Impulse Response Correction and Sound Zone Control,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 2008–2020, 2025, doi: 10.1109/TASLPRO.2025.3570949.
