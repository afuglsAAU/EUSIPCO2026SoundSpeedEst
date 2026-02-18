from pathlib import Path
import importlib.util


def load_config_module(config_path: Path):
    """Dynamically load an experiment config module.

    The module must define `shared_params`.
    """
    spec = importlib.util.spec_from_file_location("exp_config", str(config_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


#### Shared Paths ###########################################################
def get_shared_paths(experiment_path=None, config_env=None):
    paths = {}

    if config_env is not None:
        if 'SimDataPath' in config_env:
            paths['data_path'] = Path(config_env['SimDataPath'])
        if 'InputAudioPath' in config_env:
            paths['input_audio_path'] = Path(config_env['InputAudioPath'])
        if 'NoisePath' in config_env:
            paths['noise_path'] = Path(config_env['NoisePath'])
        if 'SimDataCovarPath' in config_env:
            paths['covariance_path'] = Path(config_env['SimDataCovarPath'])

    path_key_name_pairs = {
        'results_path' : 'results',
        'plots_path' : 'plots',
        'filter_files_path' : 'filter_mat_files',
    }

    # Construct paths and ensure directories exist
    for key, name in path_key_name_pairs.items():
        if key not in paths:
            if experiment_path is not None:
                base_path = experiment_path
            elif config_env is not None and 'MainOutputPath' in config_env:
                base_path = Path(config_env['MainOutputPath'])
            else:
                base_path = Path.cwd()
            paths[key] = base_path / name
        # Ensure directories exist
        if not name.endswith('.db'):
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

    return paths

