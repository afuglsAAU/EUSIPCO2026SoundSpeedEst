
"""
Index experimental results from directory structure into a CSV file.

This script scans a results directory for metadata files (meta.json) from completed
experiments, extracts relevant parameters and artifact paths, and consolidates them
into a single indexed CSV file for easy access and analysis.

Key features:
- Recursively discovers all experiments via meta.json files
- Extracts experiment metadata, runtime parameters, and artifact locations
- Handles type conversion for numeric columns
- Manages timestamps with fallback to directory modification time
- Deduplicates experiments, keeping only the most recent realization per identity
- Supports backward compatibility for legacy 'signals' field

Output:
    A CSV file (speed_track_results_index.csv) containing one row per experiment
    with consolidated metadata and paths.
"""
from pathlib import Path
import json
import pandas as pd
import sys

from dotenv import find_dotenv, dotenv_values
# Local imports from project
Loc_env = find_dotenv('.env')
CONFIG_ENV = dotenv_values(Loc_env)
_main_path = CONFIG_ENV.get('MainCodePath')
if isinstance(_main_path, str) and _main_path:
    sys.path.append(_main_path)

from src.utils.config_loader import get_shared_paths


OUTPUT_CSV = Path("results_index.csv")


def safe_get(d, key, default=None):
    return d[key] if key in d else default


def extract_row(meta_path: Path):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    params_meta = meta.get("params_meta", {})
    params_runtime = meta.get("params", {})

    row = {}

    # ----------------------------
    # Core experiment identity
    # ----------------------------
    for k, v in params_meta.items():
        row[k] = v
    # Persist experiment identity basics
    row["experiment_id"] = meta.get("experiment_id")
    row["timestamp"] = meta.get("timestamp")

    # ----------------------------
    # Useful runtime fields
    # ----------------------------
    row["estimator"] = safe_get(params_runtime, "estimator")
    row["rt60"] = safe_get(params_runtime, "rt60")
    row["track_vast_rank"] = safe_get(params_runtime, "track_vast_rank")
    row["apply_filter"] = safe_get(params_runtime, "apply_filter")
    row["update_filter"] = safe_get(params_runtime, "update_filter")
    row["adaptive_window"] = safe_get(params_runtime, "adaptive_window")

    # ----------------------------
    # Paths
    # ----------------------------
    row["experiment_dir"] = str(meta_path.parent.resolve())
    row["results_npz"] = str((meta_path.parent / "results.npz").resolve())

    # Optional artifacts
    artifacts = meta.get("artifacts", None)
    if artifacts is None:
        # Backward-compat: tracker saves under 'signals'
        artifacts = meta.get("signals", {})
    for k, v in artifacts.items():
        row[k] = v

    return row


def build_index(results_root: Path) -> pd.DataFrame:
    rows = []

    for meta_path in results_root.rglob("meta.json"):
        try:
            row = extract_row(meta_path)
            rows.append(row)
        except Exception as e:
            print(f"Skipping {meta_path}: {e}")

    df = pd.DataFrame(rows)
    return df


def main():
    # experiment_path = Path(__file__).absolute().parents[0]

    paths = get_shared_paths(config_env=CONFIG_ENV)
    # params = cfg.shared_params.copy()

    df = build_index(paths['results_path'])

    # Basic cleanup / typing
    numeric_cols = [
        "track_vast_rank",
        "speed_change",
        "speed_change_time",
        "search_window_width",
        "start_speed",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill timestamp if missing based on directory mtime
    if "timestamp" not in df.columns or df["timestamp"].isna().any():
        def mtime_from_dir(row):
            try:
                p = Path(row["experiment_dir"]).stat().st_mtime
                return pd.to_datetime(p, unit="s")
            except Exception:
                return pd.NaT
        df["timestamp_dt"] = df.apply(mtime_from_dir, axis=1)
    else:
        # Parse saved timestamp string if present
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H%M%S", errors="coerce")

    # Keep only the most recent realization per experiment identity
    if not df.empty:
        # Sort by available timestamp columns (newest first)
        sort_cols = []
        if "timestamp_dt" in df.columns:
            sort_cols.append("timestamp_dt")
        if "timestamp" in df.columns:
            sort_cols.append("timestamp")
        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")

        # Determine identity columns: prefer explicit experiment_id, otherwise fall back to key runtime params
        if "experiment_id" in df.columns and df["experiment_id"].notna().any():
            identity_cols = ["experiment_id"]
        else:
            # Fallback: use a subset of stable runtime parameters likely defining the experiment
            candidate_identity = [
                "estimator",
                "rt60",
                "track_vast_rank",
                "adaptive_window",
                "apply_filter",
                "update_filter",
            ]
            identity_cols = [c for c in candidate_identity if c in df.columns]

        if identity_cols:
            df = df.drop_duplicates(subset=identity_cols, keep="first")

    OUTPUT_CSV = paths['results_path'] / "speed_track_results_index.csv"
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved index with {len(df)} experiments → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()