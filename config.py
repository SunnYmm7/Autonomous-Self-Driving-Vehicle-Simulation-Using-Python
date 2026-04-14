# ─────────────────────────────────────────────
#  config.py – Central configuration
#  
#  All simulation parameters in one place for easy tuning.
# ─────────────────────────────────────────────

# ── Display ───────────────────────────────────
SCREEN_WIDTH: int = 1200
SCREEN_HEIGHT: int = 800
FPS: int = 60

# ── Car / Sensors ─────────────────────────────
SENSOR_COUNT: int = 7  # must stay odd so one ray points dead ahead
SENSOR_LENGTH: int = 150  # max ray length in pixels
CAR_RADIUS: int = 8
MAX_SPEED: float = 6.0
MIN_SPEED: float = 1.5
STEER_POWER: float = 4.5  # degrees per frame at full output

# ── Neural Network ────────────────────────────
HIDDEN_NODES: int = 10

# GENOME_SIZE: computed from network topology
# W1(input*hidden) + b1(hidden) + W2(hidden*output) + b2(output)
GENOME_SIZE: int = (SENSOR_COUNT * HIDDEN_NODES + HIDDEN_NODES) + \
                   (HIDDEN_NODES * 2 + 2)

# ── Genetic Algorithm ─────────────────────────
POPULATION_SIZE: int = 30
MUTATION_RATE: float = 0.08  # probability each weight mutates
MUTATION_STD: float = 0.15  # gaussian std for mutation noise
ELITE_COUNT: int = 4  # genomes preserved unchanged each generation
SELECTION_TOP: int = 10  # best genomes used for breeding

# ── Simulation limits ─────────────────────────
MAX_TICKS_PER_GEN: int = 1800  # 30 s at 60 fps – prevents stuck-car stalls

# ── Colours (R, G, B) ─────────────────────────
COL_BG: tuple = (15, 15, 25)
COL_ROAD: tuple = (55, 60, 70)
COL_ROAD_EDGE: tuple = (70, 75, 85)
COL_INFIELD: tuple = (15, 15, 25)
COL_LANE_MARK: tuple = (200, 185, 80)
COL_OBSTACLE: tuple = (220, 60, 60)
COL_CAR_ALIVE: tuple = (50, 220, 120)
COL_CAR_DEAD: tuple = (180, 40, 40)
COL_BEST_CAR: tuple = (255, 220, 0)
COL_SENSOR: tuple = (80, 160, 220)
COL_SENSOR_HIT: tuple = (255, 100, 60)
COL_UI_BG: tuple = (20, 22, 35)
COL_UI_ACCENT: tuple = (80, 140, 255)
COL_TEXT: tuple = (220, 225, 240)
COL_TEXT_DIM: tuple = (120, 125, 140)