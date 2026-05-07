import numpy as np

# Game mode: "matrix", "leduc", or "cubes"
GAME_MODE = "leduc"

# Leduc poker settings (only used when GAME_MODE = "leduc")
LEDUC_GAME = "leduc_poker(suit_isomorphism=true)"  # OpenSpiel game string
LEDUC_ALPHA = 100.0      # Dirichlet concentration: high alpha -> small perturbation around uniform
LEDUC_NUM_RANKS = 3     # Number of card ranks (J, Q, K)

# Cube settings (only used when GAME_MODE = "cubes")
# Each reset samples one of three pathological cube families (Klee-Minty,
# Jeroslow, Goldfarb-Sit) with the parameter drawn from the matching range.
CUBE_N = 10
# Tuned from the empirical sweep in docs/cubes_experiment.md — these
# parameter ranges produce distinguishable tableau patterns. Only the KM
# range actually hits 2^n pathology; JL and GS are currently KM-pathological
# variants with different tableau shapes (see cubes.py for caveats).
CUBE_EPS_RANGE = (0.6, 1.0)     # Klee-Minty tilt
CUBE_BETA_RANGE = (0.30, 0.45)  # Jeroslow objective decay
CUBE_DELTA_RANGE = (0.1, 0.3)   # Goldfarb-Sit row scaling

# Training parameters
M = 40
N = 40


TIMESTEPS = 20_000_000

# Checkpoint settings: save model every CHECKPOINT_FREQ steps after CHECKPOINT_START
CHECKPOINT_START = 1_000_000
CHECKPOINT_FREQ = 1_000_000
LOAD_MODEL = False
MATRIX_MODE = "toeplitz" # toeplitz, uniform

# UNIFORM MODE SETTINGS
MIN_VAL = -1
MAX_VAL = 1


# TOEPLITZ Matrices
TOEPLITZ_RHO = 0.95
TOEPLITZ_SIGNED = False
TOEPLITZ_ANTISYMMETRIC = False
TOEPLITZ_BAND = None

# EPSILON used in both
EPSILON = 0.001

# Perturbation magnitude used by experiment.py when generating in-distribution
# test matrices. Defaults to EPSILON so the test set matches the training
# distribution. Set to a different value to probe how the agent generalizes
# to larger or smaller perturbations than it was trained on (the saved model's
# filename / training epsilon is unaffected).
TEST_EPSILON = EPSILON

PREFERRED_ACTION_ID = 2
INITIAL_BIAS = 3.0

# Training settings
N_ENVS = 4

# (MODEL_NAME_TEMPLATE is defined further down, after USE_COMPACT_OBS and
# USE_WEIGHTED_STEP_PENALTY are set, so it can include obs/penalty tags
# automatically.)

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
USE_COMPACT_OBS = False  # Wrap env with CompactObsWrapper (31 size-independent features)
# Leduc tableau is ~(483, 965); a Dict-obs policy would have ~120M params just
# in the input layer and a 500MB+ saved model. Force compact obs for Leduc.
if GAME_MODE == "leduc":
    USE_COMPACT_OBS = True
USE_BASELINE_REWARD = False  # Shape reward by difference from baseline (steepest_edge) iter count
BASELINE_REWARD_COEF = 2.0  # Multiplier for (baseline_nit - agent_nit) terminal bonus — amplified to push past "tie with steepest"
BASELINE_REWARD_WINS_ONLY = False  # If True, only reward strict wins vs baseline (no penalty for losing) — removes the "imitate steepest to avoid losses" equilibrium
USE_FULL_PIVOT = False  # Agent plays BOTH phases (True) or just phase 2 (False); ignored for cubes
USE_LR_DECAY = False  # Linear decay of learning rate (1e-4 -> 1e-5) over the run
ENT_COEF = 0.05  # PPO entropy coefficient (higher = more exploration / action diversity)
SWITCH_BONUS = 0  # Per-step reward for picking a different action than previous — encourages strategy diversity
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
USE_TWO_PHASE = False

# Weighted step penalty: scale the per-step penalty by the empirical cost of
# the chosen pivot rule (calibrate via benchmark_pivot_cost.py /
# benchmark_pivot_cost_leduc.py). When False, every step costs -1 regardless
# of strategy.
USE_WEIGHTED_STEP_PENALTY = True

# Per-rule wallclock weights, normalized so largest_coefficient = 1.0.
# Re-run the matching benchmark if you change PIVOT_MAP or the simplex
# internals.
#
# Calibrated by benchmark_pivot_cost.py at 40x40 (matrix mode, phase 2):
STEP_PENALTY_WEIGHTS_MATRIX = {
    'largest_coefficient': 1.00,
    'steepest_edge':       1.50,
    'largest_increase':    2.50,
    'random_edge':         0.80,
    'blands_rule':         0.35,
}
# Calibrated by benchmark_pivot_cost_leduc.py at the Leduc poker phase-2
# tableau (~483x965). Steepest_edge and largest_increase are far more
# expensive on Leduc than at 40x40 because the per-column work scales with
# row count.
STEP_PENALTY_WEIGHTS_LEDUC = {
    'largest_coefficient': 1.00,
    'steepest_edge':       9.5,
    'largest_increase':   20,
    'random_edge':         0.8,
    'blands_rule':         0.4,
}
# Active dict — auto-picked from GAME_MODE so envs.py / experiment scripts
# don't need to know the mode.
STEP_PENALTY_WEIGHTS = (STEP_PENALTY_WEIGHTS_LEDUC if GAME_MODE == "leduc"
                       else STEP_PENALTY_WEIGHTS_MATRIX)

# Scale the per-step penalty linearly with tableau row count so the same
# weights generalize across LP sizes. Reference is set to a 40x40 matrix-mode
# phase-2 tableau (≈ 41 rows). Adjust if you train at a different size.
SCALE_PENALTY_BY_SIZE = False
REFERENCE_TABLEAU_ROWS = M + 1


# Run tag derived from observation-space and step-penalty flags so each
# training configuration writes to a unique model filename. Examples:
#   USE_COMPACT_OBS=False, USE_WEIGHTED_STEP_PENALTY=True  -> "dict_weighted"
#   USE_COMPACT_OBS=True,  USE_WEIGHTED_STEP_PENALTY=False -> "compact_unweighted"
_OBS_TAG = "compact" if USE_COMPACT_OBS else "dict"
_PEN_TAG = "weighted" if USE_WEIGHTED_STEP_PENALTY else "unweighted"
MODEL_RUN_TAG = f"{_OBS_TAG}_{_PEN_TAG}"

# Model save path template — uses MODEL_RUN_TAG so different obs/penalty
# combinations don't overwrite each other's saved models.
MODEL_NAME_TEMPLATE = (
    f"models/ppo_simplex_random_{{steps}}_matrix{{m}}x{{n}}"
    f"_min{{min}}_max{{max}}_epsilon{{eps}}_{MODEL_RUN_TAG}.zip"
)

