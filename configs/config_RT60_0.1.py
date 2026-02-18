import numpy as np
from pathlib import Path

#### Shared Parameters ###########################################################
# This is a base configuration file, that sets parameters, that are rarely adjusted
# They can all be changed in the experiment configuration files at will
shared_params = {

    ## Simulation specific parameters
    'fs': 16000,  # Sampling frequency of IRs to be used
    'K': 800,  # Trimmed length of impulse responses
    'L': 16,  # Number of loudspeakers used for control
    'use_lsp': np.arange(16), # Which loudspeakers to use for control, counting from 0
    'M_b': 37,  # Number of BZ microphones
    'M_d': 37, # Number of DZ microphones

    # Parameters for SICER vs VAST   fixed filters
    'sound_speeds': np.arange(333, 353+1, 1),  # Sound speeds to simulate [m/s]
    'base_speed': 343,  # No correction base sound speed
    'compare_speeds': [333, 353],  # Speeds to compare to the NC case [m/s]
    'compare_ranks': [1, 4041, 16*500],  # Ranks to compare
    'rt60': 0.1,  # Target reverberation time for simulated data [s]

    'nfft_performance': 4096,                       # nfft for fft method for performance metrics

    'array_setup': 'Linear_array_circ_zones',  # Simulation setup to use: 'Linear_array_circ_zones' for the setup in Sankha's paper

    ## Control Filter Parameters
    'vast_ranks': np.ceil(np.linspace(1, 16*500, 100)),  # number of uniformly spacecd VAST rank(s) to be evaluated between 1 and L*J (inclusive) rounded up
    'J': 500,           # Control filter length
    'ref_source': 7,    # Source for reference pressure, counting from 0
    'mu': 1,          # VAST mu weighting parameter
    'reg_param': 0,  # Regularization parameter
    'model_delay': 250,  # Modeling delay [Samples] # J/2

    'control_strat': 'GT',  # Control strategy to use: 'GT' (Ground Truth), 'SICER' (using SICER speed correction), 'NC' (no speed correction)

    'input_signal': 'audio',  # Input signal type: 'audio' for audio signal; 'white' 'white_noise' or 'noise' for white noise; 'dirac', 'impulse' or 'delta' for impulse signal

    # 'input_audio': 'EARS_combined_sentences_1m_35s_23LUFS_16kHz.wav',  # Speech
    # 'input_audio': 'music-rfm-0146_23LUFS_16kHz.wav',  # Rock
    'input_audio': 'EARS_speech_MUSAN_rock_rfm097_23LUFS_16kHz.wav',  # Mixed speech and music
    # 'input_audio': 'combined_speech_16k_11_sec_2.wav',  # Mixed speech and music
    'input_duration': 10,  # Duration of input signal in seconds (only for 'white' or 'impulse' input signals)


    #### Parameters for speed estimation experiment ##############################

    # Initial speed
    'speed_change_start_speed': 333,
    'speed_change': 2, # Speed change to apply (m/s)
    'speed_change_time': 2,  # Time between speed change (seconds)

    # frame duration in seconds
    'frame_duration': 0.25,

    # Observation microphones selection for tracking (indices are 0-based)
    # Choose a small subset from available BZ and DZ microphones.
    'obs_mic_indices': {
        'BZ': np.array([0
                        # , 6, 12, 18
                        ]),
        # 'DZ': np.array([1,
        #                 7, 13, 19
        #                 ])
    },

    'update_filter': False,  # Update control filter every n frames
    'update_filter_speed_diff': 1.0,  # Minimum speed difference between current and last estimate to trigger filter update [m/s]

    'apply_filter': False,  # Whether to apply the control filter during tracking

    # Loudspeakers used for estimating speed
    'obs_lsp_indices': [7],#np.arange(16).astype(int),

    # VAST rank to select from SICER filters when building the dictionary
    # Defaults to the highest available rank in `vast_ranks`
    'track_vast_rank': int(16*500),#int(4041),

    'estimator': 'grid',  # Estimator to use for speed tracking: 'newton', 'bisect' or 'grid'

    'estimator_base_speed': 333.0,  # Base speed for SICER IR correction inside estimator, can differ from actual simulation speed_change_start_speed [m/s]
    'search_min_speed': 326.0,  # Minimum speed to search in m/s
    'search_max_speed': 360.0,  # Maximum speed to search in m

    'adaptive_window': False,  # Whether to use adaptive search window based on previous estimates

    'search_window_width': 6,  # Size of search window around previous estimate in m/s
                                 # Plus-minus half of this value around previous / middle estimate
    'search_middle_lookback': 1,  # Number of previous frames to consider for centering search window


    'temporal_prior_weight': 0.05,#1/(2*(2**2)),
    'adaptive_base': False,  # Whether to adaptively update the base speed / IRs used for SICER during tracking

    # Parameters for grid search method
    'grid_workers': 16,  # Number of parallel workers for grid search
    'grid_backend': 'thread',  # Backend for parallel grid search: 'thread' or 'process'
    'grid_search' : {
        'tol_init': 0.25,  # Initial grid search tolerance in m/s used for full-grid (non-adaptive window) searches
        'tol': 0.1,  # Tolerance for grid search in m/s after first iteration, if adaptive_window is True
    },

    # Parameters for Newton method
    'newton': {
        'deriv_step': 0.5,  # Step size for numerical derivative
        'tol': 0.1,  # Tolerance for convergence
        'max_iters': 30,  # Maximum number of iterations
    },

    # Parameters for bisection algorithm
    'bisect': {
        'tol': 0.1,
        'max_iters': 30,
        'neighbor_frac': 0.1,
        'refine_samples': 7,
        'early_stop_patience': 5,
    },


}
