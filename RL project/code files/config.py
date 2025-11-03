# --- config.py ---

# Grid World Dimensions
GRID_WIDTH = 10
GRID_HEIGHT = 10

# Robot Start and Goal Positions
START_POS = (0, 0)
GOAL_POS = (9, 9)

# Obstacle Locations (Shelves)
# Format: list of (row, col) tuples
OBSTACLES = [
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
    (5, 0), (5, 1), (5, 2), (5, 3),
    (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9),
    (8, 4)
]

# Rewards
REWARD_GOAL = 100.0
REWARD_OBSTACLE = -10.0
REWARD_STEP = -1.0

# MDP Parameters
DISCOUNT_FACTOR = 0.99  # Gamma
CONVERGENCE_THRESHOLD = 1e-4 # Theta for value/policy iteration convergence

# Transition Model Configuration
# 'deterministic' or 'stochastic'
TRANSITION_MODEL_TYPE = 'stochastic'

# Probabilities for Stochastic Model
PROB_FORWARD = 0.8  # Probability of moving in the intended direction
PROB_LEFT = 0.1     # Probability of moving 90 degrees left of intended
PROB_RIGHT = 0.1    # Probability of moving 90 degrees right of intended

# Simulation & Visualization
ANIMATION_SPEED_MS = 150 # Speed of robot animation in milliseconds
MAX_PATH_LENGTH = 50 # Max steps for simulation runs
NUM_SIMULATIONS = 1000 # Number of simulations for evaluation
