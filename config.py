import numpy as np

# Game mode: "matrix" for random matrix LPs, "leduc" for Leduc poker sequence-form LPs
GAME_MODE = "leduc"# "matrix" or "leduc"

# Leduc poker settings (only used when GAME_MODE = "leduc")
LEDUC_GAME = "leduc_poker(suit_isomorphism=true)"  # OpenSpiel game string
LEDUC_ALPHA = 100.0      # Dirichlet concentration: high alpha -> small perturbation around uniform
LEDUC_NUM_RANKS = 3     # Number of card ranks (J, Q, K)

# Training parameters
M = 40
N = 40

TIMESTEPS = 1_000_000

# Checkpoint settings: save model every CHECKPOINT_FREQ steps after CHECKPOINT_START
CHECKPOINT_START = 500_000
CHECKPOINT_FREQ = 500_000
LOAD_MODEL = False
MATRIX_MODE = "uniform" # toeplitz, uniform

# UNIFORM MODE SETTINGS
MIN_VAL = -1
MAX_VAL = 1


# TOEPLITZ Matrices
TOEPLITZ_RHO = 0.80
TOEPLITZ_SIGNED = False
TOEPLITZ_ANTISYMMETRIC = False
TOEPLITZ_BAND = None

# EPSILON used in both
EPSILON = 0.001

PREFERRED_ACTION_ID = 2
INITIAL_BIAS = 3.0

# Training settings
N_ENVS = 4
# Model save path template
MODEL_NAME_TEMPLATE = "models/ppo_simplex_random_{steps}_matrix{m}x{n}_min{min}_max{max}_epsilon{eps}.zip"

# Max BFS depth
BFS_DEPTH = 10

# Action to pivot name map
PIVOT_MAP = {
    0: 'largest_coefficient',
    1: 'largest_increase',
    2: 'steepest_edge',
    # 3: 'random_edge'
    # 3: 'blands_rule'
}

PIVOT_MAP_TEST = {
    0: 'largest_coefficient',
    1: 'largest_increase',
    2: 'steepest_edge',
    3: 'random_edge',
    4: 'blands_rule'
}

NUM_PIVOT_STRATEGIES = len(PIVOT_MAP)
NUM_PIVOT_STRATEGIES_TEST = len(PIVOT_MAP_TEST)
PIVOT_STRATEGY_NAMES = list(PIVOT_MAP_TEST.values())

# Advanced training features
USE_MACRO_STRATEGY = False  # Set to True to use MacroStrategyWrapper
USE_BIAS_ANNEALING = False # Set to True to use LogitBiasAnnealCallback
USE_INITIAL_ACTION_BIAS = False

# History tracking feature
USE_HISTORY_TRACKING = False  # Enable history tracking and printing when objective doesn't improve
HISTORY_SIZE = 20  # Number of last steps to keep in history
NO_IMPROVE_STEPS = 100  # Number of steps without improvement before printing history

# Single coordinate noise matrix feature
USE_SINGLE_COORDINATE_NOISE = False  # Set to True to use SingleCoordinateNoiseMatrix instead of Matrix
SINGLE_COORDINATE_NOISE_FLAG = False  # When USE_SINGLE_COORDINATE_NOISE=True, this controls the single_coordinate_noise flag

# Two-phase simplex: when True, build Phase 1 tableau and solve it (Bland's rule),
# then transition to Phase 2 for the RL agent. When False, construct
# the Phase 2 tableau directly (skipping Phase 1).
# NOTE: the two modes produce different tableau shapes, so models are NOT interchangeable.
USE_TWO_PHASE = True

