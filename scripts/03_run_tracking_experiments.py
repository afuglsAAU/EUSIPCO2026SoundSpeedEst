
"""
Script to run speed-tracking experiments with configurable parameters, filtering, and parallel execution.

This script orchestrates the execution of multiple tracking experiments based on YAML configuration.
It supports:
  - Dynamic experiment generation from configuration (groups, schedules, directions, inputs)
  - Adaptive window limiting based on speed-change schedules
  - Parallel execution with configurable worker and threading policies
  - Result deduplication by experiment ID
  - CLI filtering by group, experiment, schedule, direction, input, rank, and mode
  - Threading optimization (MKL, OMP, estimator workers)
  - Dry-run mode for validation without execution

Key features:
  - Generates experiment ID hashes for result tracking
  - Optionally skips duplicate experiments if results already exist
  - Dynamically adjusts search windows based on speed-change timing
  - Manages per-experiment threading configuration
  - Supports both sequential and parallel execution modes
"""

import yaml
import subprocess
import time
from pathlib import Path
import json
import hashlib
import argparse
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import find_dotenv, dotenv_values
# Local imports from project
Loc_env = find_dotenv('.env')
CONFIG_ENV = dotenv_values(Loc_env)
_main_path = CONFIG_ENV.get('MainCodePath')
if isinstance(_main_path, str) and _main_path:
    sys.path.append(_main_path)

from src.utils.config_loader import get_shared_paths


TRACKING_SCRIPT = Path("scripts/tracking_algorithm.py")
BASE_CONFIG = Path("configs/config_RT60_0.1.py")


INPUT_AUDIO_MAP = {
    "speech": "EARS_combined_sentences_1m_35s_23LUFS_16kHz.wav",
    "rock": "music-rfm-0146_23LUFS_16kHz.wav",
    "speech_music": "EARS_speech_MUSAN_rock_rfm097_23LUFS_16kHz.wav",
}


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def hash_experiment(meta_dict):
    blob = json.dumps(meta_dict, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def build_input_params(input_name):
    if input_name == "white":
        return {"input_signal": "white"}
    return {
        "input_signal": "audio",
        "input_audio": INPUT_AUDIO_MAP[input_name],
    }


def sanitize_params(params):
    """
    Remove keys that must NEVER reach tracking_algorithm.py
    """
    forbidden_keys = {
        "vast_ranks",
        "inputs",
        "speed_sign",
        "modes",
        "_group",
        "_schedule",
        "_direction",
        "_input",
        "_mode",
    }

    cleaned = {k: v for k, v in params.items() if k not in forbidden_keys}
    # Normalize special cases
    if cleaned.get("active_lsp_indices") == "all":
        # Let tracking script fall back to full use_lsp
        cleaned.pop("active_lsp_indices", None)
    return cleaned




def determine_window_sets_for_schedule(speed_change: int, speed_change_time: int, fallback_sets):
    """
    Limit adaptive search windows by schedule as per experiment_list.md.
    Returns a list of integer widths.

    Mapping:
      - (1, 1) → [6]
      - (1, 2) → [6]
      - (2, 2) → [6, 8]
      - (4, 2) → [8, 10, 12]
      - (4, 4) → [8, 10, 12]
    Otherwise, fall back to provided default sets.
    """
    sc = abs(int(speed_change)) if speed_change is not None else None
    dt = int(speed_change_time) if speed_change_time is not None else None
    if sc == 1 and dt in (1, 2):
        return [6]
    if sc == 2 and dt == 2:
        return [6, 8]
    if sc == 4 and dt in (2, 4):
        return [8, 10, 12]
    return fallback_sets


def has_existing_results(exp_id: str) -> bool:
    """Check if any results already exist for a given experiment_id.

    Scans the local results tree for meta.json files containing this id.
    """
    paths = get_shared_paths(config_env=CONFIG_ENV)
    root = Path(paths['results_path'])
    if not root.exists():
        return False
    try:
        for mp in root.rglob("meta.json"):
            try:
                with open(mp, "r") as f:
                    meta = json.load(f)
                if str(meta.get("experiment_id")) == str(exp_id):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def select_threading(params_runtime, mkl_override=None, estimator_workers_override=None):
    """Decide per-experiment threading policy (MKL threads and estimator workers).

    Defaults:
      - If updating filters: heavy MKL, few estimator workers
      - Else: minimal MKL, high estimator parallelism
    CLI overrides take precedence.
    Note: Default numbers based on 128-core CPU and for CLI max-parallel=4 experiments; adjust as needed for different hardware.
    """
    cfg = {
        "mkl": 32 if params_runtime.get("update_filter", False) else 2,
        "estimator_workers": 2 if params_runtime.get("update_filter", False) else 16,
        "estimator_backend": params_runtime.get("grid_backend", "thread"),
    }
    if mkl_override is not None:
        cfg["mkl"] = int(mkl_override)
    if estimator_workers_override is not None:
        cfg["estimator_workers"] = int(estimator_workers_override)
    return cfg


def build_command(params_runtime, python_exec, params_path, quiet: bool = False):
    cmd = [
        python_exec,
        str(TRACKING_SCRIPT),
        "--config",
        str(BASE_CONFIG),
        "--estimator",
        params_runtime["estimator"],
        "--params-file",
        str(params_path),
    ]

    if params_runtime.get("apply_filter"):
        cmd.append("--apply-filter")
    if params_runtime.get("update_filter"):
        cmd.append("--update-filter")
    if params_runtime.get("compute_obs_only"):
        cmd.append("--obs-only")
    if params_runtime.get("gt_baseline"):
        cmd.append("--gt-baseline")
    if params_runtime.get("sicer_baseline"):
        cmd.append("--sicer-baseline")
    if (
        "active_lsp_indices" in params_runtime
        and params_runtime["active_lsp_indices"] != "all"
    ):
        cmd += [
            "--active-lsp",
            ",".join(map(str, params_runtime["active_lsp_indices"])),
        ]
    if quiet:
        cmd.append("--quiet")
    return cmd


def launch_experiment(args):
    params_runtime, params_meta, threading_cfg, dry_run, skip_existing, is_parallel = args

    exp_id = hash_experiment(params_meta)
    # Skip if results already exist
    if skip_existing and has_existing_results(exp_id):
        print(f"Skipping Experiment {exp_id}: results already exist.")
        return {"elapsed": 0.0, "skipped": True, "exp_id": exp_id}

    # Prepare runtime payload (including identity and overrides)
    payload = dict(params_runtime)
    # Ensure grid workers/backend reflect threading config
    if threading_cfg.get("estimator_workers") is not None:
        payload["grid_workers"] = int(threading_cfg["estimator_workers"])
    if threading_cfg.get("estimator_backend"):
        payload["grid_backend"] = str(threading_cfg["estimator_backend"])  # keep as-is if set
    # Quiet mode for parallel runs: disable progressbar and final print inside tracker
    if is_parallel:
        payload["quiet"] = True
        payload["disable_progressbar"] = True

    payload["experiment_id"] = exp_id
    payload["params_meta"] = params_meta
    payload['threading_config'] = threading_cfg

    # Write params JSON
    params_dir = Path(".run_params")
    params_dir.mkdir(parents=True, exist_ok=True)
    params_path = params_dir / f"{exp_id}.json"
    with open(params_path, "w") as f:
        json.dump(payload, f)


    # Build command
    python_exec = os.environ.get("PYTHON_EXECUTABLE", sys.executable)
    cmd = build_command(params_runtime, python_exec, params_path, quiet=bool(is_parallel))

    # Threading environment
    env = os.environ.copy()
    mkl_threads = int(threading_cfg.get("mkl", 1))
    env.update({
        "MKL_NUM_THREADS": str(mkl_threads),
        "OMP_NUM_THREADS": str(mkl_threads),
        # "OPENBLAS_NUM_THREADS": str(mkl_threads),
        # "NUMEXPR_NUM_THREADS": str(mkl_threads),
    })

    # For non-parallel runs, dump params like before
    if not is_parallel:
        print("\n===================================================")
        print(f"Experiment ID: {exp_id}")
        print("Runtime params:")
        print(json.dumps(payload, indent=2))
        print("Command:")
        print(" ".join(cmd))
        print("===================================================")

    if dry_run:
        print(" ".join(cmd))
        return {"elapsed": 0.0, "skipped": False, "exp_id": exp_id}

    start = time.perf_counter()
    subprocess.run(cmd, env=env, check=True)
    end = time.perf_counter()
    return {"elapsed": float(end - start), "skipped": False, "exp_id": exp_id}


# ------------------------------------------------------------
# Experiment execution
# ------------------------------------------------------------
# ------------------------------------------------------------
# Main expansion loop
# ------------------------------------------------------------
def main():
    cfg = load_yaml("configs/experiments.yaml")

    globals_cfg = cfg["globals"]
    schedules = cfg["speed_schedules"]
    directions = cfg["directions"]
    windows = cfg["adaptive_windows"]

    groups = {
        "one_lsp_one_mic": cfg["one_lsp_one_mic"],
        "multi_lsp_one_mic": cfg["multi_lsp_one_mic"],
    }

    # CLI filters for targeted runs
    parser = argparse.ArgumentParser(description="Run speed-tracking experiments with optional filters")
    parser.add_argument("--group", choices=list(groups.keys()), help="Experiment group to run", default=None)
    parser.add_argument("--experiment", help="Experiment name within group (e.g., no_filter, szc_update)", default=None)
    parser.add_argument("--schedule", choices=list(schedules.keys()), help="Speed schedule to run", default=None)
    parser.add_argument("--direction", choices=list(directions.keys()), help="Speed change direction", default=None)
    parser.add_argument("--input", choices=list(globals_cfg["inputs"]), help="Input selection", default=None)
    parser.add_argument("--rank", type=int, help="Track VAST rank override", default=None)
    parser.add_argument("--mode", choices=list(windows.keys()), help="Grid/adaptive mode", default=None)
    parser.add_argument("--window-width", type=int, help="Search window width for adaptive mode", default=None)
    parser.add_argument("--limit-input-duration", type=float, help="Override input duration (seconds) for faster validation", default=None)
    parser.add_argument("--no-dry-run", dest="no_dry_run", action="store_true", help="Execute experiments instead of dry-run")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", help="Skip experiments that already have results (by experiment_id)")
    parser.add_argument("--max-parallel", type=int, default=4, help="Max concurrent experiments to run")
    parser.add_argument("--mkl", type=int, default=None, help="Override MKL/OMP thread count for all experiments")
    parser.add_argument("--estimator-workers", type=int, default=None, help="Override estimator worker count inside tracking (grid_search.workers)")
    parser.add_argument("--estimator-backend", choices=["thread", "process"], default=None, help="Override estimator backend (thread/process)")
    parser.add_argument("--filter-updated", choices=["yes", "no"], default=None, help="Sub-select experiments by whether they update filters")
    args, _ = parser.parse_known_args()

    DRY_RUN = not bool(args.no_dry_run)

    N_experiments = 0
    default_rank = globals_cfg["vast_ranks"][-1] if globals_cfg.get("vast_ranks") else None
    # Pre-count applying filters
    groups_iter = groups.items()
    if args.group:
        groups_iter = [(args.group, groups[args.group])]
    for _grp_name, group in groups_iter:
        base_group = group["base"]
        experiments_iter = group["experiments"]
        if args.experiment:
            experiments_iter = [e for e in experiments_iter if e.get("name") == args.experiment]
        if args.filter_updated is not None:
            want_updated = (args.filter_updated == "yes")
            experiments_iter = [e for e in experiments_iter if bool(e.get("update_filter", False)) == want_updated]
        for exp_def in experiments_iter:
            need_rank = bool(exp_def.get("apply_filter", False) or exp_def.get("gt_baseline", False) or exp_def.get("sicer_baseline", False))
            rank_space = globals_cfg["vast_ranks"] if need_rank else [default_rank]
            schedules_iter = schedules.items()
            if args.schedule:
                schedules_iter = [(args.schedule, schedules[args.schedule])]
            directions_iter = directions.items()
            if args.direction:
                directions_iter = [(args.direction, directions[args.direction])]
            inputs_iter = globals_cfg["inputs"] if not args.input else [args.input]
            for _sched_name, sched in schedules_iter:
                for _dir_name, direction in directions_iter:
                    for input_name in inputs_iter:
                        for rank in rank_space:
                            modes_iter = exp_def["modes"] if not args.mode else [args.mode]
                            for mode in modes_iter:
                                if mode == "adaptive_window":
                                    fallback_sets = [ws["search_window_width"] for ws in windows[mode]["window_sets"]]
                                    allowed_sets = determine_window_sets_for_schedule(
                                        speed_change=sched.get("speed_change"),
                                        speed_change_time=sched.get("speed_change_time"),
                                        fallback_sets=fallback_sets,
                                    )
                                    if args.window_width:
                                        allowed_sets = [int(args.window_width)]
                                    win_cfgs = [{"search_window_width": int(w)} for w in allowed_sets]
                                else:
                                    win_cfgs = [{}]
                                for win in win_cfgs:
                                    N_experiments += 1
    print(f"Total experiments to run: {N_experiments}")

    # Build job list
    jobs = []
    total = 0
    groups_iter = groups.items() if not args.group else [(args.group, groups[args.group])]
    for group_name, group in groups_iter:
        base_group = group["base"]
        experiments_iter = group["experiments"] if not args.experiment else [e for e in group["experiments"] if e.get("name") == args.experiment]
        if args.filter_updated is not None:
            want_updated = (args.filter_updated == "yes")
            experiments_iter = [e for e in experiments_iter if bool(e.get("update_filter", False)) == want_updated]
        for exp_def in experiments_iter:
            schedules_iter = schedules.items() if not args.schedule else [(args.schedule, schedules[args.schedule])]
            directions_iter = directions.items() if not args.direction else [(args.direction, directions[args.direction])]
            inputs_iter = globals_cfg["inputs"] if not args.input else [args.input]
            for sched_name, sched in schedules_iter:
                for dir_name, direction in directions_iter:
                    for input_name in inputs_iter:
                        need_rank = bool(exp_def.get("apply_filter", False) or exp_def.get("gt_baseline", False) or exp_def.get("sicer_baseline", False))
                        rank_space = globals_cfg["vast_ranks"] if need_rank else [default_rank]
                        rank_space = [args.rank] if args.rank is not None else rank_space
                        modes_iter = exp_def["modes"] if not args.mode else [args.mode]
                        for rank in rank_space:
                            for mode in modes_iter:
                                if mode == "adaptive_window":
                                    fallback_sets = [ws["search_window_width"] for ws in windows[mode]["window_sets"]]
                                    allowed_sets = determine_window_sets_for_schedule(
                                        speed_change=sched.get("speed_change"),
                                        speed_change_time=sched.get("speed_change_time"),
                                        fallback_sets=fallback_sets,
                                    )
                                    if args.window_width:
                                        allowed_sets = [int(args.window_width)]
                                    win_cfgs = [{"search_window_width": int(w)} for w in allowed_sets]
                                else:
                                    win_cfgs = [{}]

                                for win in win_cfgs:
                                    # Build full param dict
                                    params = {}
                                    params.update(globals_cfg)
                                    params.update(base_group)
                                    params.update(exp_def)
                                    params.update(sched)
                                    params.update(direction)
                                    params.update(win)

                                    # Adaptive flag + lookback
                                    if mode == "adaptive_window":
                                        params["adaptive_window"] = True
                                        if "search_middle_lookback" not in params and "search_middle_lookback" in windows.get("adaptive_window", {}):
                                            params["search_middle_lookback"] = windows["adaptive_window"]["search_middle_lookback"]
                                    else:
                                        params["adaptive_window"] = False
                                    params["mode"] = mode

                                    # Apply speed sign
                                    params["speed_change"] *= params["speed_sign"]

                                    # Rank
                                    if rank is not None:
                                        params["track_vast_rank"] = rank

                                    # Input handling
                                    params.update(build_input_params(input_name))

                                    # Optional duration override
                                    if args.limit_input_duration is not None:
                                        params["input_duration"] = float(args.limit_input_duration)

                                    # Meta identity
                                    params_meta = {
                                        "group": group_name,
                                        "schedule": sched_name,
                                        "direction": dir_name,
                                        "input": input_name,
                                        "filter_mode": exp_def.get("name", ""),
                                        "grid_mode": mode,
                                        "track_vast_rank": rank,
                                        "search_window_width": params.get("search_window_width"),
                                        "speed_change": params["speed_change"],
                                        "speed_change_time": params["speed_change_time"],
                                        "start_speed": params["speed_change_start_speed"],
                                    }

                                    params_runtime = sanitize_params(params)

                                    # Threading config for this job
                                    threading_cfg = select_threading(
                                        params_runtime,
                                        mkl_override=args.mkl,
                                        estimator_workers_override=args.estimator_workers,
                                    )
                                    if args.estimator_backend:
                                        threading_cfg["estimator_backend"] = args.estimator_backend

                                    jobs.append((params_runtime, params_meta, threading_cfg, DRY_RUN, bool(args.skip_existing)))
                                    total += 1

    print(f"Prepared {total} jobs. Running with max_parallel={int(args.max_parallel)}")

    def fmt_time(seconds: float) -> str:
        seconds = int(max(0, seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        else:
            return f"{m:02d}:{s:02d}"

    def print_progress(done: int, total_jobs: int, avg_sec: float, start_time: float):
        left = max(0, total_jobs - done)
        eta = avg_sec * left if avg_sec > 0 else 0.0
        elapsed = time.perf_counter() - start_time
        bar_width = 30
        filled = int(bar_width * done / max(1, total_jobs))
        bar = "#" * filled + "." * (bar_width - filled)
        msg = (
            f"[{bar}] {done}/{total_jobs} | avg/job: {fmt_time(avg_sec)} "
            f"| elapsed: {fmt_time(elapsed)} | left: {left} | ETA: {fmt_time(eta)}"
        )
        print(msg, end='\r', flush=True)

    start_time = time.perf_counter()

    if int(args.max_parallel) <= 1:
        done = 0
        durations = []
        for job in jobs:
            jr = list(job)
            if len(jr) == 5:
                jr.append(False)
            res = launch_experiment(tuple(jr))
            done += 1
            if isinstance(res, dict) and not res.get("skipped", False):
                durations.append(float(res.get("elapsed", 0.0)))
            avg = (sum(durations) / len(durations)) if durations else 0.0
            print_progress(done, total, avg, start_time)
        print("\n", end="")
    else:
        done = 0
        durations = []
        with ProcessPoolExecutor(max_workers=int(args.max_parallel)) as pool:
            jobs_parallel = []
            for job in jobs:
                jr = list(job)
                if len(jr) == 5:
                    jr.append(True)
                jobs_parallel.append(tuple(jr))
            futures = [pool.submit(launch_experiment, job) for job in jobs_parallel]
            for f in as_completed(futures):
                res = f.result()
                done += 1
                if isinstance(res, dict) and not res.get("skipped", False):
                    durations.append(float(res.get("elapsed", 0.0)))
                # For parallel, average is per-worker as processes run in parallel
                avg = (sum(durations) / len(durations) / float(args.max_parallel)) if durations else 0.0
                print_progress(done, total, avg, start_time)
        print("\n", end="")


if __name__ == "__main__":
    main()
